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

    torch.set_default_device("cuda")
    for num_tokens in [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
    ]:
        run_grid(num_tokens, model=args.model, TP=args.TP)


## Utilize method from rocm/Triton tuning script
def get_full_tuning_space():
    configs = []

    for num_stages in [0]:
        for block_m in [16, 32, 64, 128, 256]:
            for block_n in [32, 64, 128, 256]:
                for block_k in [64, 128, 256]:
                    for num_warps in [1, 2, 4, 8]:
                        for group_size in [1, 16, 32, 64]:
                            for waves_per_eu in [0]:
                                configs.append({
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": group_size,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                    "waves_per_eu": waves_per_eu,
                                })
    return configs

def run_grid(num_tokens, model, TP):
    if model == '8x7B':
        hidden_size = 4096
        intermediate_size = 14336
    elif model == '8x22B':
        hidden_size = 6144
        intermediate_size = 16384
         
    else:
        raise ValueError(f"Unsupported Mixtral model {model}")

    num_experts = 8
    top_k = 2
    tp_size = TP
    shard_intermediate_size = 2 * intermediate_size // tp_size

    configs = get_full_tuning_space()
    print(f"{num_tokens=} | {len(configs)=}")
    best_config = None
    best_time = float("inf")

    baseline_time = benchmark_config(
        config=None,
        num_tokens=num_tokens,
        num_experts=num_experts,
        shard_intermediate_size=shard_intermediate_size,
        hidden_size=hidden_size,
        top_k=top_k,
        num_iters=10,
    )
    for config in tqdm(configs):
        # Benchmark
        try:
            kernel_time = benchmark_config(
                config=config,
                num_tokens=num_tokens,
                num_experts=num_experts,
                shard_intermediate_size=shard_intermediate_size,
                hidden_size=hidden_size,
                top_k=top_k,
                num_iters=10,
            )

            if kernel_time < best_time:
                best_config = config
                best_time = kernel_time
        except triton.runtime.autotuner.OutOfResources:
            continue

    # holds Dict[str, Dict[str, int]]
    filename = get_config_file_name(num_experts,
                                    shard_intermediate_size // 2,
                                    "fp8_w8a8")
    print(f"[DONE {num_tokens=} | {best_config=}]: {best_time=:0.1f} | {baseline_time=:0.1f} | Writing to {filename}")
    existing_content = {}
    if os.path.exists(filename):
        with open(filename) as f:
            existing_content = json.load(f)
    existing_content[str(num_tokens)] = best_config
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

    def prepare(i: int):
        input_gating.copy_(gating_output[i])

    def run():
        fused_moe(
            x,
            w1,
            w2,
            input_gating,
            top_k,
            renormalize=True,
            inplace=True,
            use_fp8_w8a8=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            override_config=config,
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