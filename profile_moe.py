import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from tqdm import tqdm

from vllm.model_executor.layers.fused_moe.fused_moe import *
from vllm.platforms import current_platform
from vllm.utils import FlexibleArgumentParser


def main(args):
    os.environ["HIP_VISIBLE_DEVICES"] = args.GPUID
    os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
    os.environ["DEBUG_CLR_GRAPH_PACKET_CAPTURE"] = "1"
    os.environ["OPTIMIZE_EPILOGUE"] = "1"

    for num_tokens in [
            1,
            # 2,
            # 4,
            # 8,
            # 16,
            # 24,
            # 32,
            # 48,
            # 64,
            # 96,
            # 128,
            # 256,
            # 512,
            # 1024,
            # 1536,
            # 2048,
            # 3072,
            # 4096,
    ]:
        run_grid(num_tokens, model=args.model, TP=args.TP)


## Utilize method from rocm/Triton tuning script
def get_full_tuning_space():
    configs = []

    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256]
    # split_k_range = [1] #, 2, 4, 5, 6, 8, 10, 12, 16, 18, 24]
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 4, 8, 16, 32]
    # For now we see better perf with num_stages=0 for all gemm configs we care
    # But keep this explicit so that we do not forget we may need to set it to
    # other values in the future
    num_stage_range = [0]
    waves_per_eu_range = [0]
    matrix_instr_nonkdim_range = [16, 32]
    kpack_range = [1, 2]

    for block_m in block_mn_range:
        for block_n in block_mn_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        # for split_k in split_k_range:
                        for num_stages in num_stage_range:
                            for waves_per_eu in waves_per_eu_range:
                                for (matrix_instr_nonkdim
                                     ) in matrix_instr_nonkdim_range:
                                    for kpack in kpack_range:
                                        configs.append({
                                            "BLOCK_SIZE_M": block_m,
                                            "BLOCK_SIZE_N": block_n,
                                            "BLOCK_SIZE_K": block_k,
                                            "GROUP_SIZE_M": group_m,
                                            "num_warps": num_warps,
                                            "num_stages": num_stages,
                                            "waves_per_eu": waves_per_eu,
                                            "matrix_instr_nonkdim":
                                            matrix_instr_nonkdim,
                                            "kpack": kpack,
                                        })

    return configs


## Utilize method from rocm/Triton tuning script
def prune_configs(M, N, K, configs):
    pruned_configs = []
    elemBytes_a = 2  # [DV Note] Hard-coded for float16 (2 bytes)
    elemBytes_b = 2  # [DV Note] Hard-coded for float16 (2 bytes)

    mfma = 16 if M < 32 or N < 32 else 32

    # TODO (zhanglx): figure out the boundary between large and small gemms
    large_gemm = False
    if M >= 2048 and N >= 2048:
        large_gemm = True

    for config in configs:
        BLOCK_SIZE_M = config.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.get("BLOCK_SIZE_N")
        BLOCK_SIZE_K = config.get("BLOCK_SIZE_K")
        num_warps = config.get("num_warps")
        matrix_instr_nonkdim = config.get("matrix_instr_nonkdim")
        # kpack = config.get("kpack")
        if matrix_instr_nonkdim > mfma:
            continue
        if mfma == 4 and BLOCK_SIZE_K < 64:
            continue
        # some layouts could not work properly in case
        # number elements per thread is less 1
        if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
            continue
        SPLIT_K = 1  # config.get("SPLIT_K")
        GROUP_M = config.get("GROUP_SIZE_M")
        if (matrix_instr_nonkdim > BLOCK_SIZE_M
                or matrix_instr_nonkdim > BLOCK_SIZE_N):
            continue
        if matrix_instr_nonkdim >= M and matrix_instr_nonkdim != BLOCK_SIZE_M:
            continue
        if matrix_instr_nonkdim >= N and matrix_instr_nonkdim != BLOCK_SIZE_N:
            continue
        # Skip BLOCK_SIZE that is too large compare to M/N
        # unless BLOCK_SIZE is already small enough
        if M * 2 < BLOCK_SIZE_M and BLOCK_SIZE_M != 16:
            continue
        if N * 2 < BLOCK_SIZE_N and BLOCK_SIZE_N != 16:
            continue
        # skip large split_k when not necessary
        if SPLIT_K != 1 and not need_split_k(M, N, K):
            continue
        # skip split_k that leads to EVEN_K = false
        leap = SPLIT_K * BLOCK_SIZE_K
        modv = K % leap
        if modv != 0:
            continue
        # skip large GROUP_M
        if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
            continue
        # out of shared memory resource
        # TODO (zhanglx): This does not consider the LDS usage in the epilogue
        LDS = (BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a +
               BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b)
        if LDS > 65536:
            continue
        # Skip small block sizes and num_warps for large gemm
        # For fp16 and f8, we want to only use BLOCK_SIZE >= 64
        if large_gemm:
            if BLOCK_SIZE_M < 64 or BLOCK_SIZE_N < 64:
                continue
            if BLOCK_SIZE_K < 64:
                continue
            if num_warps < 4:
                continue

        pruned_configs.append(config)

    return pruned_configs


def union_of_list_of_dicts(l1, l2):
    result = []
    temp_list = l1.copy()
    temp_list.extend(l2)
    for myDict in temp_list:
        if myDict not in result:
            result.append(myDict)

    return result


def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024


def run_grid(num_tokens, model, TP):
    if model == '8x7B':
        hidden_size = 4096
        intermediate_size = 14336
    elif model == '8x22B':
        hidden_size = 6144
        intermediate_size = 16384
         
    else:
        raise ValueError(f'Unsupported Mixtral model {model}')

    num_experts = 8
    top_k = 2
    tp_size = TP
    shard_intermediate_size = 2 * intermediate_size // args.tp_size

    num_warmup_trials = 1
    num_trials = 1

    full_configs = get_full_tuning_space()
    M1 = num_tokens * 2
    N1 = intermediate_size * 2 // tp_size
    K1 = hidden_size
    prune_configs_1 = prune_configs(M1, N1, K1, full_configs)

    M2 = num_tokens * 2
    N2 = hidden_size
    K2 = intermediate_size // tp_size
    prune_configs_2 = prune_configs(M2, N2, K2, full_configs)

    configs = union_of_list_of_dicts(prune_configs_1, prune_configs_2)
    print(f"{num_tokens=} || {len(full_configs)=} | {len(prune_configs_1)=} | \
            {len(prune_configs_2)=} | {len(configs)=}")

    best_config = None
    best_time = float("inf")

    for config in tqdm(configs):
        # Benchmark
        try:
            kernel_time = benchmark_config(
                config=config,
                num_tokens=num_tokens,
                num_experts=NUM_EXPERTS,
                shard_intermediate_size=shard_intermediate_size,
                hidden_size=hidden_size,
                top_k=top_k,
                num_iters=10,
            )
            print(f"[DONE KERNEL]: {kernel_time=.1f}")

            if kernel_dur_us < best_time_us:
                best_config = config
                best_time_us = kernel_dur_us
        except triton.runtime.autotuner.OutOfResources:
            continue

    # holds Dict[str, Dict[str, int]]
    filename = get_config_file_name(NUM_EXPERTS,
                                    shard_intermediate_size,
                                    dtype=None)
    print(f"[DONE CONFIG {num_tokens=}]: Writing to {filename}.")
    existing_content = {}
    if os.path.exists(filename):
        with open(filename) as f:
            existing_content = json.load(f)
    existing_content[str(bs)] = best_config
    with open(filename, "w") as f:
        json.dump(existing_content, f, indent=4)
        f.write("\n")


def benchmark_config(
    config,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    top_k: int,
    num_iters: int,
) -> float:

    dtype = torch.float16
    fp8_dtype = torch.float8_e4m3fnuz

    x = torch.randn(num_tokens, hidden_size, dtype=dtype)

    w1 = torch.randn(
        num_experts,
        shard_intermediate_size,
        hidden_size,
        dtype=dtype)

     w2 = torch.randn(
        num_experts,
        hidden_size,
        shard_intermediate_size // 2,
        dtype=dtype)

    gating_output = torch.randn(
        num_iters,
        num_tokens,
        num_experts,
        dtype=torch.float32)

    w1_scale = torch.randn(num_experts, dtype=torch.float32)
    w2_scale = torch.randn(num_experts, dtype=torch.float32)
    a1_scale = torch.randn(1, dtype=torch.float32)
    a2_scale = torch.randn(1, dtype=torch.float32)

    w1 = w1.to(fp8_dtype)
    w2 = w2.to(fp8_dtype)

    input_gating = torch.empty(num_tokens, num_experts, dtype=torch.float32)

    def run():
        from vllm.model_executor.layers.fused_moe import override_config
        with override_config(config):
            fused_moe(
                x,
                w1,
                w2,
                input_gating,
                topk,
                renormalize=True,
                inplace=True,
                use_fp8_w8a8=True,
                use_int8_w8a16=use_int8_w8a16,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
            )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: List[float] = []
    for i in range(num_iters):
        prepare(i)
        torch.cuda.synchronize()

        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="benchmark_mixtral_moe_rocm",
        description="Tune the fused_moe kernel for mixtral.")
    parser.add_argument(
        "--TP",
        type=int,
        choices=[8, 4, 2, 1],
        help="Specify the TP value that the actual model will run on",
        required=True,
    )
    parser.add_argument(
        "--GPUID",
        type=str,
        help="This script uses single GPU. Specify the GPU to use for tuning",
        default="0",
    )
    parser.add_argument('--model',
                        type=str,
                        choices=['8x7B', '8x22B'],
                        help='The Mixtral model to benchmark')

    args = parser.parse_args()

    print(f"Running tuning for {args.model} model")
    print(f"TP is set to: {args.TP}")
    print(f"GPU-ID being used for tuning: {args.GPUID}")
    sys.exit(main(args))