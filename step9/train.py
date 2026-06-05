"""
train.py —— 120M 模型训练脚本
============================================================
相对 step5 的 train.py 新增：

1) **梯度累积**（Gradient Accumulation）：
   - 16GB 显存无法一次放下大 batch
   - 内循环跑 GRAD_ACCUM 个 micro-batch，累积梯度后再 step
   - 有效 batch = MICRO_BATCH × GRAD_ACCUM

2) **断点续训**（Resume from checkpoint）：
   - 启动时检查 out/ckpt.pt，若存在则自动恢复 model + optimizer + iter
   - 长时间训练（12~20h）中断后可无缝继续

3) **训练监控增强**：
   - tokens/sec 吞吐量
   - 显存占用打印
   - 进度百分比

4) **compile（可选）**：
   - torch.compile 在支持的环境下加速 10~30%
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
#   超参
# ============================================================
OUT_DIR        = os.path.join(os.path.dirname(__file__), "out")

# ---- 数据 & batch ----
BLOCK_SIZE     = 1024       # 上下文窗口
MICRO_BATCH    = 8          # 每个 micro-step 的 batch size
GRAD_ACCUM     = 4          # 梯度累积步数 → 有效 batch = 8×4 = 32

# ---- 训练步数 ----
MAX_ITERS      = 50000      # 总迭代步（每步 = 一次 optimizer.step）
EVAL_INTERVAL  = 500        # 每多少步评估一次
EVAL_ITERS     = 20         # 评估时采多少个 batch 求平均

# ---- 学习率 ----
LR             = 3e-4       # 峰值学习率
MIN_LR         = 3e-5       # 退火到的最小学习率
WARMUP_ITERS   = 500        # 线性 warmup 步数
LR_DECAY_ITERS = MAX_ITERS  # 余弦退火在多少步内完成

# ---- 正则化 ----
WEIGHT_DECAY   = 0.1        # AdamW 权重衰减
GRAD_CLIP      = 1.0        # 梯度 L2 范数裁剪

# ---- 速度优化 ----
USE_AMP        = True       # 混合精度
AMP_DTYPE      = torch.bfloat16   # Ada/Ampere 用 bf16；老卡可改 float16
USE_COMPILE    = False      # torch.compile（Windows 支持有限，可设 True 尝试）

# ---- 其它 ----
SEED           = 1337
RESUME         = True       # 是否自动从 ckpt 恢复

# ---- 模型配置 ----
MODEL_CFG = dict(
    block_size = BLOCK_SIZE,
    n_layer    = 12,
    n_head     = 12,
    n_embd     = 768,
    dropout    = 0.0,
    rope_theta = 10000.0,
)
# ============================================================


def get_batch(data, block_size, batch_size, device):
    """向量化采样。"""
    n = len(data) - block_size - 1
    ix = np.random.randint(0, n, size=batch_size)
    rng = np.arange(block_size, dtype=np.int64)
    x_np = data[ix[:, None] + rng].astype(np.int64, copy=False)
    y_np = data[ix[:, None] + rng + 1].astype(np.int64, copy=False)
    x = torch.from_numpy(x_np).pin_memory() if device == "cuda" else torch.from_numpy(x_np)
    y = torch.from_numpy(y_np).pin_memory() if device == "cuda" else torch.from_numpy(y_np)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device):
    """在 train/val 上各采 EVAL_ITERS 个 batch 估计 loss。"""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(EVAL_ITERS, device=device)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, block_size, batch_size, device)
            if USE_AMP:
                with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            losses[k] = loss.detach()
        out[name] = losses.mean().item()
    model.train()
    return out


def get_lr(it):
    """Warmup + 余弦退火。"""
    if it < WARMUP_ITERS:
        return LR * it / max(1, WARMUP_ITERS)
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LR - MIN_LR)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # 1) 准备数据
    if not (os.path.exists(TRAIN_BIN) and os.path.exists(META_PATH)):
        prepare()
    meta       = load_meta()
    train_data = load_split("train")
    val_data   = load_split("val")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val   tokens: {len(val_data):,}")

    # 2) 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 3) 建模型
    cfg = GPTConfig(vocab_size=meta["vocab_size"], **MODEL_CFG)
    model = GPT(cfg).to(device)
    n_params = model.num_params()
    print(f"Model parameters: {n_params:,} (~{n_params/1e6:.1f}M)")

    if USE_COMPILE and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    # 4) 优化器
    # 区分需要 weight_decay 和不需要的参数（bias、norm 不做衰减）
    decay_params = []
    no_decay_params = []
    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    optim_groups = [
        {"params": decay_params,    "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(optim_groups, lr=LR, betas=(0.9, 0.95), fused=True)

    # 5) 断点续训
    start_iter = 1
    best_val = float("inf")
    ckpt_path = os.path.join(OUT_DIR, "ckpt.pt")
    if RESUME and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"] + 1
        best_val = ckpt["val_loss"]
        print(f"  Resumed at iter={start_iter}, best_val={best_val:.4f}")

    # 6) 损失日志 CSV
    csv_path = os.path.join(OUT_DIR, "losses.csv")
    csv_mode = "a" if (RESUME and start_iter > 1) else "w"
    csv_file = open(csv_path, csv_mode, newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if csv_mode == "w":
        csv_writer.writerow(["iter", "train_loss", "val_loss", "lr", "tokens_per_sec"])
    csv_file.flush()

    # 7) 训练主循环
    splits = {"train": train_data, "val": val_data}
    tokens_per_iter = MICRO_BATCH * GRAD_ACCUM * BLOCK_SIZE
    print(f"\nTokens per iteration: {tokens_per_iter:,} "
          f"(micro_batch={MICRO_BATCH} × grad_accum={GRAD_ACCUM} × block_size={BLOCK_SIZE})")
    print(f"Total training tokens over {MAX_ITERS} iters: "
          f"~{tokens_per_iter * MAX_ITERS / 1e9:.2f}B")
    print(f"\n{'='*70}")
    print(f"  Starting training from iter {start_iter} to {MAX_ITERS}")
    print(f"{'='*70}\n")

    t0 = time.time()
    local_t0 = time.time()

    for it in range(start_iter, MAX_ITERS + 1):
        # 设置学习率
        lr = get_lr(it)
        for pg in optim.param_groups:
            pg["lr"] = lr

        # ---- 梯度累积循环 ----
        optim.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(GRAD_ACCUM):
            x, y = get_batch(train_data, BLOCK_SIZE, MICRO_BATCH, device)
            if USE_AMP:
                with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            # 梯度除以累积步数，等价于在大 batch 上算平均
            scaled_loss = loss / GRAD_ACCUM
            scaled_loss.backward()
            accum_loss += loss.item()

        # 梯度裁剪 + 优化器更新
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim.step()

        accum_loss /= GRAD_ACCUM  # 平均 loss

        # ---- 周期性评估 ----
        if it % EVAL_INTERVAL == 0 or it == start_iter:
            losses = estimate_loss(model, splits, BLOCK_SIZE, MICRO_BATCH, device)
            dt = time.time() - t0
            local_dt = time.time() - local_t0
            tps = tokens_per_iter * EVAL_INTERVAL / local_dt if it > start_iter else 0
            local_t0 = time.time()

            # 显存信息
            if device == "cuda":
                mem_used = torch.cuda.max_memory_allocated() / 1e9
                mem_str = f"mem={mem_used:.1f}GB"
            else:
                mem_str = ""

            progress = it / MAX_ITERS * 100
            print(
                f"iter {it:6d}/{MAX_ITERS} ({progress:5.1f}%) | "
                f"lr {lr:.2e} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"{tps:,.0f} tok/s | {mem_str} | "
                f"elapsed {dt:.0f}s"
            )

            csv_writer.writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}",
                                 f"{lr:.2e}", f"{tps:.0f}"])
            csv_file.flush()

            # 保存最佳 ckpt
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model":     model.state_dict(),
                        "optimizer": optim.state_dict(),
                        "cfg":       cfg.__dict__,
                        "iter":      it,
                        "val_loss":  best_val,
                    },
                    ckpt_path,
                )
                print(f"  ↳ saved checkpoint (val_loss={best_val:.4f})")

    csv_file.close()
    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Best val loss: {best_val:.4f}")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"  Losses CSV: {csv_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
