"""
generate.py —— 用训练好的 ckpt 生成文本
=====================================
用法：
    python generate.py --prompt "ROMEO:" --max_new_tokens 500

参数说明：
    --ckpt           ：模型权重文件路径
    --prompt         ：起始文本（模型从这里开始续写）
    --max_new_tokens ：往后再生成多少个字符
    --temperature    ：>1 更随机/发散，<1 更保守/确定
    --top_k          ：每步只从概率最高的 k 个候选里抽，过滤低概率噪声
    --seed           ：随机种子（同种子 + 同 prompt → 同结果）
"""
import argparse
import os
import torch

from model import GPT, GPTConfig
from data  import load_meta, get_encoder


def main():
    # ---- 1) 解析命令行参数 ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default=os.path.join(os.path.dirname(__file__), "out", "ckpt.pt"))
    ap.add_argument("--prompt",         default="\n")
    ap.add_argument("--max_new_tokens", type=int,   default=500)
    ap.add_argument("--temperature",    type=float, default=0.8)
    ap.add_argument("--top_k",          type=int,   default=40)
    ap.add_argument("--seed",           type=int,   default=1337)
    args = ap.parse_args()

    # ---- 2) 设种子 + 选设备 ----
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 3) 加载 BPE 编码器（负责 token <-> 字符串 的互转）----
    enc = get_encoder()

    # ---- 4) 加载 ckpt 并重建模型 ----
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg  = GPTConfig(**ckpt["cfg"])           # 用保存的配置重建一模一样的结构
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])      # 灌入训练好的权重
    model.eval()                              # 推理模式
    print(f"Loaded ckpt @ iter {ckpt['iter']} val_loss={ckpt['val_loss']:.4f}")

    # ---- 5) prompt → token id 序列（BPE 编码）----
    ids = enc.encode_ordinary(args.prompt)
    if len(ids) == 0:
        ids = [enc.encode_ordinary("\n")[0]]   # 防止空 prompt
    x = torch.tensor([ids], dtype=torch.long, device=device)   # (1, T)

    # ---- 6) 自回归采样 ----
    out = model.generate(
        x,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # ---- 7) BPE 解码：token id 列表 → 字符串 ----
    text = enc.decode(out[0].tolist())
    print("=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
