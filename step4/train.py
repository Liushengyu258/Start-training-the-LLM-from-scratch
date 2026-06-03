"""
train.py —— 训练脚本
==========================
做的事情：
    1) 读数据 + 词表
    2) 建模型 + 优化器
    3) 主循环：取一个 batch → 前向 → 反向 → 更新参数
    4) 每隔若干步在训练/验证集上评估 loss，保存到 CSV，并保存最佳 ckpt

新手要理解的几个关键概念：
- iter（迭代步）：处理一个 mini-batch 算一步
- batch（批）：一次同时处理多少条样本（这里 64 条，每条长 128 字符）
- loss：交叉熵越小代表预测越准；初始 loss ≈ ln(vocab_size)
- learning rate (lr)：参数更新的步幅。常用“warmup + 余弦退火”策略
- AdamW：带权重衰减的 Adam 优化器，是 Transformer 的标配
- gradient clipping：把梯度的总长度限制住，防止梯度爆炸
"""
import os
import csv
import time
import math
import torch
import numpy as np

from model import GPT, GPTConfig
from data  import load_meta, load_split, prepare, TRAIN_BIN, META_PATH

# ============================================================
#   超参（你可以直接在这里修改后重跑）
# ============================================================
OUT_DIR        = os.path.join(os.path.dirname(__file__), "out")
BATCH_SIZE     = 32        # 4060Ti 8GB 在 12M + block=512 下够用
BLOCK_SIZE     = 512       # RoPE 打开后上下文再加倍
MAX_ITERS      = 10000     # 数据变多了，多训一倍
EVAL_INTERVAL  = 250       # 每多少步做一次评估 + 存 ckpt
EVAL_ITERS     = 50        # 评估 batch 数（略调低减评估耗时）
LR             = 3e-4      # 峰值学习率
MIN_LR         = 3e-5      # 退火到的最小学习率
WARMUP_ITERS   = 100       # 前多少步线性升温到 LR
LR_DECAY_ITERS = MAX_ITERS # 余弦退火在多少步内完成
WEIGHT_DECAY   = 0.1       # AdamW 的权重衰减
GRAD_CLIP      = 1.0       # 梯度裁剪阈值（梯度 L2 范数上限）
SEED           = 1337      # 随机种子，便于复现
# ============================================================


def get_batch(data, block_size, batch_size, device):
    """
    随机从长 token 序列里采 batch_size 条长 block_size 的子串：
        x = data[i : i+block_size]          ← 输入
        y = data[i+1 : i+1+block_size]      ← 输出（每个位置预测下一个）
    这种“偏移 1 位”的对齐方式就是语言模型的标准做法。
    """
    # 随机起点：保证 i + block_size + 1 不越界
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    # 把每条样本组成张量 (B, T)；data 是 uint16，要转成 int64 给 nn.Embedding 用
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64))         for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    # non_blocking=True 可以让 CPU→GPU 拷贝异步进行（配合 pin_memory 时更明显）
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()                        # 评估不需要算梯度
def estimate_loss(model, splits, block_size, batch_size, device):
    """在 train 和 val 两个集合上各采 EVAL_ITERS 个 batch，分别求平均 loss。"""
    model.eval()                        # 切到推理模式（关掉 Dropout）
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()                       # 切回训练模式
    return out


def get_lr(it):
    """
    学习率调度：
        阶段 1 (warmup)：从 0 线性升到 LR    —— 防止刚开始大步长把权重打飞
        阶段 2 (cosine)：从 LR 余弦退火到 MIN_LR —— 后期小步长精细打磨
        阶段 3 (固定)  ：到了 LR_DECAY_ITERS 之后保持 MIN_LR
    """
    if it < WARMUP_ITERS:
        return LR * it / max(1, WARMUP_ITERS)
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # 1 → 0 的余弦曲线
    return MIN_LR + coeff * (LR - MIN_LR)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    # 1) 准备数据：第一次运行会自动下载
    if not (os.path.exists(TRAIN_BIN) and os.path.exists(META_PATH)):
        prepare()
    meta       = load_meta()
    train_data = load_split("train")
    val_data   = load_split("val")

    # 2) 选设备：有 GPU 就用 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 3) 建模型
    # ---- 约 12M 参数 + RoPE ----
    cfg = GPTConfig(
        vocab_size = meta["vocab_size"],
        block_size = BLOCK_SIZE,
        n_layer    = 6,
        n_head     = 6,
        n_embd     = 192,
        dropout    = 0.1,
        rope_theta = 10000.0,
    )
    model = GPT(cfg).to(device)
    n_params = model.num_params()
    print(f"Model parameters: {n_params:,} (~{n_params/1e6:.2f}M)")

    # 4) 优化器：AdamW = Adam + decoupled weight decay
    #    betas: (0.9, 0.95) 是 GPT 系列常用配置
    optim = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY,
    )

    # 5) 准备损失日志 CSV
    csv_path = os.path.join(OUT_DIR, "losses.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["iter", "train_loss", "val_loss", "lr"])  # 表头
    csv_file.flush()

    # 6) 训练主循环
    splits   = {"train": train_data, "val": val_data}
    t0       = time.time()
    best_val = float("inf")

    for it in range(1, MAX_ITERS + 1):
        # —— 设置当前步学习率 ——
        lr = get_lr(it)
        for pg in optim.param_groups:
            pg["lr"] = lr

        # —— 取一个 batch ——
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)

        # —— 前向 + 计算 loss ——
        _, loss = model(x, y)

        # —— 反向传播 ——
        optim.zero_grad(set_to_none=True)   # 清掉上一步的梯度（更省内存的写法）
        loss.backward()                      # 自动求导，把梯度填进每个参数的 .grad

        # —— 梯度裁剪：防止个别 batch 梯度过大把权重打飞 ——
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # —— 用优化器把梯度应用到参数上 ——
        optim.step()

        # —— 周期性评估 + 写日志 + 存 ckpt ——
        if it % EVAL_INTERVAL == 0 or it == 1:
            losses = estimate_loss(model, splits, BLOCK_SIZE, BATCH_SIZE, device)
            dt = time.time() - t0
            print(
                f"iter {it:5d} | lr {lr:.2e} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | elapsed {dt:.1f}s"
            )
            # 写一行到 losses.csv，flush 保证训练中途也能用 plot.py 实时画图
            csv_writer.writerow([it, losses["train"], losses["val"], lr])
            csv_file.flush()

            # 验证集 loss 创新低就保存 checkpoint
            if losses["val"] < best_val:
                best_val = losses["val"]
                ckpt_path = os.path.join(OUT_DIR, "ckpt.pt")
                torch.save(
                    {
                        "model":    model.state_dict(),   # 所有参数
                        "cfg":      cfg.__dict__,          # 配置（生成时要重建模型）
                        "iter":     it,
                        "val_loss": best_val,
                    },
                    ckpt_path,
                )
                print(f"  ↳ saved checkpoint to {ckpt_path}")

    csv_file.close()
    print(f"Done. Best val loss = {best_val:.4f}")
    print(f"损失日志: {csv_path}")
    print(f"运行 `python plot.py` 可绘制曲线。")


if __name__ == "__main__":
    main()
