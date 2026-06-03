"""
benchmark.py —— 朴素生成 vs KV-cache 生成 的速度对比
============================================================
对同一个模型 / 同一段 prompt / 同样的采样数，分别用两种方法各跑 N 轮，
打印平均耗时、tokens/s、加速比。

用法：
    python benchmark.py
    python benchmark.py --ckpt ../step5/out/ckpt.pt --n_tokens 200 --n_runs 3
"""
import argparse
import os
import time
import torch
import tiktoken

from model import GPT, GPTConfig


def time_generate(fn, *, n_warmup=1, n_runs=3, device="cuda"):
    """返回 (mean_seconds, std_seconds, last_output)。"""
    out = None
    # 预热（CUDA 第一次跑通常有 kernel 编译开销）
    for _ in range(n_warmup):
        out = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    mean = sum(times) / len(times)
    var  = sum((t - mean) ** 2 for t in times) / len(times)
    return mean, var ** 0.5, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default=os.path.join(os.path.dirname(__file__), "..", "step7", "out", "ckpt.pt"))
    ap.add_argument("--prompt", default="Once upon a time, in a quiet little village,")
    ap.add_argument("--n_tokens", type=int, default=200)
    ap.add_argument("--n_runs",   type=int, default=3)
    ap.add_argument("--seed",     type=int, default=1337)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg  = GPTConfig(**ckpt["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Device: {device}")
    print(f"Model : {model.num_params()/1e6:.2f}M params,  block_size={cfg.block_size}")
    print(f"Ckpt  : {args.ckpt}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Generating {args.n_tokens} new tokens × {args.n_runs} runs each\n")

    ids = enc.encode_ordinary(args.prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    def naive():
        torch.manual_seed(args.seed)
        return model.generate(x, args.n_tokens, temperature=1.0, top_k=None)

    def cached():
        torch.manual_seed(args.seed)
        return model.generate_cached(x, args.n_tokens, temperature=1.0, top_k=None)

    print("--- Naive (no cache, O(T^2) per step) ---")
    t_naive, s_naive, out_naive = time_generate(naive, n_runs=args.n_runs, device=device)
    tps_naive = args.n_tokens / t_naive
    print(f"mean {t_naive*1000:.1f} ms ± {s_naive*1000:.1f} ms   "
          f"throughput {tps_naive:.1f} tok/s\n")

    print("--- Cached (KV-cache, O(1) per step) ---")
    t_cached, s_cached, out_cached = time_generate(cached, n_runs=args.n_runs, device=device)
    tps_cached = args.n_tokens / t_cached
    print(f"mean {t_cached*1000:.1f} ms ± {s_cached*1000:.1f} ms   "
          f"throughput {tps_cached:.1f} tok/s\n")

    print(f"=== Speedup: {t_naive / t_cached:.2f}x ===")

    # 正确性检查：两种方法在相同 seed 下应输出相同 token 序列
    same = torch.equal(out_naive, out_cached)
    print(f"\nOutputs identical: {same}")
    if not same:
        diff = (out_naive != out_cached).nonzero()
        print(f"  first diff at position: {diff[0].tolist() if len(diff) else 'N/A'}")


if __name__ == "__main__":
    main()
