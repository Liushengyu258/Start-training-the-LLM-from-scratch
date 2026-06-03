"""
train.py —— SFT（Supervised Fine-Tuning）训练
============================================================
关键差别 vs 预训练 (step3~5)：
  1) 从 step5 的预训练 ckpt 加载权重（不是从随机初始化开始）
  2) 学习率小很多（3e-5 ~ 1e-4），避免“遗忘”预训练学到的语言能力
  3) loss 只在 response 的 token 上算（用 prompt mask 把 prompt 段的标签置 -100）
  4) 数据是定长 (N, block_size) 的，按 batch 抽行；不再像预训练那样随机切窗口
"""
import os
import csv
import time
import math
import torch
import numpy as np

from model import GPT, GPTConfig
from data  import load_meta, load_split, prepare, IDS_BIN, META_PATH

# ============================================================
#   超参
# ============================================================
OUT_DIR        = os.path.join(os.path.dirname(__file__), "out")
# 预训练 ckpt 路径：默认指向 step5 的产物。没有就报错或随机初始化（看 LOAD_PRETRAIN）
PRETRAIN_CKPT  = os.path.join(os.path.dirname(__file__), "..", "step5", "out", "ckpt.pt")
LOAD_PRETRAIN  = True        # False 时表示直接从随机初始化做 SFT（效果会很差，仅用于对照）

BATCH_SIZE     = 16          # SFT 数据 padding 较多，bs 可以小一点
MAX_ITERS      = 3000        # Alpaca 52K / bs16 ≈ 3250 iter/epoch，约 1 epoch
EVAL_INTERVAL  = 200
EVAL_ITERS     = 30
LR             = 3e-5        # SFT 一般用比预训练小一个量级的 lr
MIN_LR         = 3e-6
WARMUP_ITERS   = 50
LR_DECAY_ITERS = MAX_ITERS
WEIGHT_DECAY   = 0.0         # SFT 通常不用或调小，避免“拉回”预训练的 weight 分布
GRAD_CLIP      = 1.0
SEED           = 1337
# ============================================================


def get_batch(ids, mask, batch_size, device):
    """
    从 (N, T) 的二维数组里随机抽 batch_size 行。
    构造：
        x = ids[:, :-1]                 输入
        y = ids[:, 1:]                  目标（下一个 token）
        loss_mask = mask[:, 1:]         哪些位置参与 loss（response 段）
    然后把 loss_mask=0 的目标位置替换成 -100，让 F.cross_entropy 忽略。
    """
    N = ids.shape[0]
    ix = torch.randint(N, (batch_size,)).numpy()

    batch_ids  = ids[ix].astype(np.int64)   # (B, T)
    batch_mask = mask[ix].astype(np.int64)  # (B, T)

    x = torch.from_numpy(batch_ids[:, :-1]).contiguous()
    y = torch.from_numpy(batch_ids[:, 1:]).contiguous()
    m = torch.from_numpy(batch_mask[:, 1:]).contiguous()

    # 把不参与 loss 的位置替换成 -100（cross_entropy 默认 ignore_index=-100）
    y = torch.where(m.bool(), y, torch.full_like(y, -100))

    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, splits, batch_size, device):
    model.eval()
    out = {}
    for name, (ids, mask) in splits.items():
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(ids, mask, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def get_lr(it):
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

    # 1) 准备数据
    if not (os.path.exists(IDS_BIN) and os.path.exists(META_PATH)):
        prepare()
    meta = load_meta()
    train_ids,  train_mask = load_split("train")
    val_ids,    val_mask   = load_split("val")
    print(f"train samples: {len(train_ids)}    val samples: {len(val_ids)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 2) 构建模型（与 step5 完全同结构）
    cfg = GPTConfig(
        vocab_size = meta["vocab_size"],
        block_size = meta["block_size"],
        n_layer    = 6,
        n_head     = 6,
        n_embd     = 192,
        dropout    = 0.0,    # SFT 通常关 dropout 或调小
        rope_theta = 10000.0,
        norm_eps   = 1e-5,
    )
    model = GPT(cfg).to(device)
    n_params = model.num_params()
    print(f"Model parameters: {n_params:,} (~{n_params/1e6:.2f}M)")

    # 3) 加载预训练权重
    if LOAD_PRETRAIN:
        assert os.path.exists(PRETRAIN_CKPT), (
            f"找不到预训练 ckpt: {PRETRAIN_CKPT}\n"
            "请先运行 step5/train.py 训练一个 base 模型，或设置 LOAD_PRETRAIN=False"
        )
        ck = torch.load(PRETRAIN_CKPT, map_location=device)
        # 注意 step5 的 cfg 和这里的一致才能直接 load
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"Loaded pretrain from {PRETRAIN_CKPT}")
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("WARNING: 从随机初始化做 SFT（仅用于对比，效果会很差）")

    # 4) 优化器
    optim = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY,
    )

    # 5) 日志
    csv_path = os.path.join(OUT_DIR, "losses.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["iter", "train_loss", "val_loss", "lr"])
    csv_file.flush()

    splits = {"train": (train_ids, train_mask), "val": (val_ids, val_mask)}
    t0 = time.time()
    best_val = float("inf")

    for it in range(1, MAX_ITERS + 1):
        lr = get_lr(it)
        for pg in optim.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_ids, train_mask, BATCH_SIZE, device)
        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim.step()

        if it % EVAL_INTERVAL == 0 or it == 1:
            losses = estimate_loss(model, splits, BATCH_SIZE, device)
            dt = time.time() - t0
            print(
                f"iter {it:5d} | lr {lr:.2e} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | elapsed {dt:.1f}s"
            )
            csv_writer.writerow([it, losses["train"], losses["val"], lr])
            csv_file.flush()

            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model":    model.state_dict(),
                        "cfg":      cfg.__dict__,
                        "iter":     it,
                        "val_loss": best_val,
                    },
                    os.path.join(OUT_DIR, "ckpt.pt"),
                )
                print(f"  -> saved ckpt (val {best_val:.4f})")

    csv_file.close()
    print(f"Done. Best val loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
