"""
generate.py —— 用 SFT 后的模型按 Alpaca 模板做指令推理
============================================================
用法：
    python generate.py --instruction "Translate to French: I love cats."
    python generate.py --instruction "Summarize this." --input "..."

会自动拼成 Alpaca 标准 prompt，模型生成完“### Response:\\n”之后的内容。
"""
import argparse
import os
import torch

from model import GPT, GPTConfig
from data  import get_encoder, build_prompt, EOS_ID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default=os.path.join(os.path.dirname(__file__), "out", "ckpt.pt"))
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--input",       default="")
    ap.add_argument("--max_new_tokens", type=int,   default=200)
    ap.add_argument("--temperature",    type=float, default=0.7)
    ap.add_argument("--top_k",          type=int,   default=40)
    ap.add_argument("--seed",           type=int,   default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc = get_encoder()

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg  = GPTConfig(**ckpt["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded ckpt @ iter {ckpt['iter']}  val_loss={ckpt['val_loss']:.4f}")

    # ---- 拼 prompt ----
    prompt = build_prompt(args.instruction, args.input)
    print("=" * 60)
    print(prompt, end="")           # 把 prompt 先打印出来
    prompt_ids = enc.encode_ordinary(prompt)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # ---- 自回归采样，遇到 EOS 提前停 ----
    for _ in range(args.max_new_tokens):
        idx_cond = x[:, -cfg.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / args.temperature
        if args.top_k:
            v, _ = torch.topk(logits, args.top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, nxt], dim=1)
        # 流式打印新字符
        new_id = int(nxt.item())
        print(enc.decode([new_id]), end="", flush=True)
        if new_id == EOS_ID:
            break

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
