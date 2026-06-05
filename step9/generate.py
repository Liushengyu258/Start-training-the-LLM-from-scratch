"""
generate.py —— 用训练好的 ckpt 生成文本
============================================================
用法：
    python generate.py --prompt "Once upon a time" --max_new_tokens 300
    python generate.py --prompt "The history of" --temperature 0.6 --top_k 50

同 step3~5 的接口，使用 BPE 编码/解码。
"""
import argparse
import os
import torch

from model import GPT, GPTConfig
from data  import load_meta, get_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default=os.path.join(os.path.dirname(__file__), "out", "ckpt.pt"))
    ap.add_argument("--prompt",         default="\n")
    ap.add_argument("--max_new_tokens", type=int,   default=300)
    ap.add_argument("--temperature",    type=float, default=0.8)
    ap.add_argument("--top_k",          type=int,   default=50)
    ap.add_argument("--seed",           type=int,   default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc = get_encoder()

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg  = GPTConfig(**ckpt["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded ckpt @ iter {ckpt['iter']} val_loss={ckpt['val_loss']:.4f}")
    print(f"Model: {model.num_params()/1e6:.1f}M params, block_size={cfg.block_size}")

    # prompt → token ids
    ids = enc.encode_ordinary(args.prompt)
    if len(ids) == 0:
        ids = [enc.encode_ordinary("\n")[0]]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    # 自回归采样
    out = model.generate(
        x,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    text = enc.decode(out[0].tolist())
    print("=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
