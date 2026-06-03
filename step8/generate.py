"""
generate.py —— 用 KV-cache 加速的生成
============================================================
直接加载 step5 / step6 / step7 任一个 ckpt，权重 100% 兼容。
默认走 KV-cache 路径（--no-cache 可切回朴素方式做对比）。

用法示例：
    # 加载 step7 (DPO) 模型，按 HH 对话模板生成
    python generate.py --message "How do I learn Rust?"

    # 加载 step6 (SFT) 模型，按 Alpaca 模板
    python generate.py --ckpt ../step6/out/ckpt.pt --template alpaca --message "Translate to French: Hello"

    # 加载 step5 (base) 模型，做续写
    python generate.py --ckpt ../step5/out/ckpt.pt --template none --message "Once upon a time"

    # 关闭 KV-cache 做对比
    python generate.py --no-cache --message "Tell me a joke."
"""
import argparse
import os
import torch
import tiktoken

from model import GPT, GPTConfig

EOS_ID = 50256

ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{msg}\n\n### Response:\n"
)
HH_TEMPLATE = "\n\nHuman: {msg}\n\nAssistant:"


def build_prompt(template: str, msg: str) -> str:
    if template == "alpaca":
        return ALPACA_NO_INPUT.format(msg=msg)
    if template == "hh":
        return HH_TEMPLATE.format(msg=msg)
    return msg                              # template=none → 原样


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default=os.path.join(os.path.dirname(__file__), "..", "step7", "out", "ckpt.pt"))
    ap.add_argument("--message", required=True)
    ap.add_argument("--template", choices=["hh", "alpaca", "none"], default="hh")
    ap.add_argument("--max_new_tokens", type=int,   default=200)
    ap.add_argument("--temperature",    type=float, default=0.7)
    ap.add_argument("--top_k",          type=int,   default=40)
    ap.add_argument("--seed",           type=int,   default=1337)
    ap.add_argument("--no-cache", action="store_true",
                    help="使用朴素的 O(T^2) 生成做对比")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc = tiktoken.get_encoding("gpt2")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg  = GPTConfig(**ckpt["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded ckpt: {args.ckpt}")
    print(f"  iter={ckpt['iter']}  val_loss={ckpt['val_loss']:.4f}")

    prompt = build_prompt(args.template, args.message)
    ids = enc.encode_ordinary(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    print("=" * 60)
    print(prompt, end="", flush=True)

    if args.no_cache:
        # 朴素：一次性算完再打印（也可以改成逐步采样并打印，但意义不大）
        out = model.generate(x, args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k)
        new_ids = out[0].tolist()[len(ids):]
        # 遇到 EOS 截断
        if EOS_ID in new_ids:
            new_ids = new_ids[: new_ids.index(EOS_ID) + 1]
        print(enc.decode(new_ids), end="", flush=True)
    else:
        # KV-cache：流式打印每个 token
        stop = {"flag": False}
        def on_token(tid: int):
            if stop["flag"]:
                return
            print(enc.decode([tid]), end="", flush=True)
            if tid == EOS_ID:
                stop["flag"] = True

        # 由于 generate_cached 没法中途 break（要保持函数纯净），
        # 用一个小 wrapper 在到达 EOS 后停止打印（仍会算完剩余 token，
        # 但用户视觉上看到的是“到 EOS 就结束了”）
        model.generate_cached(x, args.max_new_tokens,
                              temperature=args.temperature, top_k=args.top_k,
                              on_token=on_token)

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
