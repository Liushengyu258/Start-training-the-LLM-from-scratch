"""
train.py —— DPO（Direct Preference Optimization）训练
============================================================
核心思想（一句话）：
    让 policy 模型相对 reference 模型，给“好回答”更高的对数概率，
    给“坏回答”更低的对数概率。

数学（一条 prompt 的损失）：
    logits = β · ( (logπ_θ(y_w|x) - logπ_ref(y_w|x))
                  -(logπ_θ(y_l|x) - logπ_ref(y_l|x)) )
    L      = -log σ(logits)

实现要点：
    - 2 个模型在显存里：policy（可训练）+ reference（冻结、no_grad）
    - 每个 batch 做 4 次 forward：
        policy(chosen), policy(rejected), ref(chosen), ref(rejected)
      其中 ref 的 2 次包在 torch.no_grad() 里
    - 取 token 级 log-prob 后用 response mask 求和，得到序列级 log-prob
    - β（beta）越大，policy 越被允许偏离 reference；常用 0.1~0.5
"""
import os
import csv
import time
import math
import torch
import torch.nn.functional as F
import numpy as np

from model import GPT, GPTConfig
from data  import load_meta, load_split, prepare, CHO_IDS, META_PATH

# ============================================================
#   超参
# ============================================================
OUT_DIR        = os.path.join(os.path.dirname(__file__), "out")
# DPO 的“起点”应该是 SFT 后的模型（step6 的 ckpt），不是 base
SFT_CKPT       = os.path.join(os.path.dirname(__file__), "..", "step6", "out", "ckpt.pt")

BATCH_SIZE     = 4           # 双模型 + 4 次 forward，bs 要小
MAX_ITERS      = 2000
EVAL_INTERVAL  = 100
EVAL_ITERS     = 20
LR             = 1e-6        # DPO 用极小的 lr，否则容易把模型“打飞”
MIN_LR         = 1e-7
WARMUP_ITERS   = 50
LR_DECAY_ITERS = MAX_ITERS
WEIGHT_DECAY   = 0.0
GRAD_CLIP      = 1.0
BETA           = 0.1         # DPO 温度系数
SEED           = 1337
# ============================================================


def get_batch(splits_data, batch_size, device):
    """随机从 (cho_ids, cho_mask, rej_ids, rej_mask) 抽 batch_size 行。"""
    cho_ids, cho_mask, rej_ids, rej_mask = splits_data
    N = cho_ids.shape[0]
    ix = torch.randint(N, (batch_size,)).numpy()
    def to_t(arr, dtype):
        return torch.from_numpy(arr[ix].astype(dtype)).to(device, non_blocking=True)
    return (to_t(cho_ids, np.int64), to_t(cho_mask, np.int64),
            to_t(rej_ids, np.int64), to_t(rej_mask, np.int64))


def sequence_logprob(model, ids, mask):
    """
    计算每条序列在 response 段上的 log-prob 总和。
    ids:  (B, T) 完整 token 序列（包含 prompt + response + pad）
    mask: (B, T) 1 表示该位置是 response token（应计入 logprob）
    返回: (B,) 每条序列的总 log-prob
    """
    # 标准语言模型：第 t 个位置的输出预测第 t+1 个 token
    inp     = ids [:, :-1]                # (B, T-1)
    targets = ids [:, 1:]                 # (B, T-1)
    m       = mask[:, 1:].float()         # (B, T-1)

    logits, _ = model(inp)                # (B, T-1, V)
    log_probs = F.log_softmax(logits, dim=-1)
    # gather 每个位置的目标 log-prob
    tok_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)   # (B, T-1)
    # 只在 response 位置求和
    return (tok_lp * m).sum(dim=-1)       # (B,)


def dpo_loss(policy_chosen_lp, policy_rejected_lp,
             ref_chosen_lp,    ref_rejected_lp, beta):
    """
    返回:
      loss       : 标量
      stats dict : 包含 reward 间距、acc 等诊断指标
    """
    pi_logits = beta * ((policy_chosen_lp   - ref_chosen_lp)
                       -(policy_rejected_lp - ref_rejected_lp))
    loss = -F.logsigmoid(pi_logits).mean()

    # 诊断：reward = β·(logπ_θ - logπ_ref)
    with torch.no_grad():
        r_chosen   = beta * (policy_chosen_lp   - ref_chosen_lp)
        r_rejected = beta * (policy_rejected_lp - ref_rejected_lp)
        margin = (r_chosen - r_rejected).mean().item()
        acc    = (r_chosen > r_rejected).float().mean().item()
    return loss, {"reward_margin": margin, "acc": acc}


def get_lr(it):
    if it < WARMUP_ITERS:
        return LR * it / max(1, WARMUP_ITERS)
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LR - MIN_LR)


def build_model_from_ckpt(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    return model, cfg


@torch.no_grad()
def estimate_loss(policy, reference, splits, batch_size, device):
    policy.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(EVAL_ITERS)
        margins = torch.zeros(EVAL_ITERS)
        accs   = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            cho_ids, cho_mask, rej_ids, rej_mask = get_batch(data, batch_size, device)
            pi_c = sequence_logprob(policy,    cho_ids, cho_mask)
            pi_r = sequence_logprob(policy,    rej_ids, rej_mask)
            rf_c = sequence_logprob(reference, cho_ids, cho_mask)
            rf_r = sequence_logprob(reference, rej_ids, rej_mask)
            loss, stats = dpo_loss(pi_c, pi_r, rf_c, rf_r, BETA)
            losses[k]  = loss.item()
            margins[k] = stats["reward_margin"]
            accs[k]    = stats["acc"]
        out[name] = (losses.mean().item(), margins.mean().item(), accs.mean().item())
    policy.train()
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    # 1) 数据
    if not (os.path.exists(CHO_IDS) and os.path.exists(META_PATH)):
        prepare()
    meta = load_meta()
    train_data = load_split("train")
    val_data   = load_split("val")
    print(f"train pairs: {train_data[0].shape[0]}  val pairs: {val_data[0].shape[0]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 2) 加载 policy 和 reference（都从 SFT ckpt 起步，reference 冻结）
    assert os.path.exists(SFT_CKPT), (
        f"找不到 SFT ckpt: {SFT_CKPT}\n请先跑 step6/train.py 得到 SFT 模型"
    )
    print(f"Loading policy   from {SFT_CKPT}")
    policy,    _   = build_model_from_ckpt(SFT_CKPT, device)
    print(f"Loading reference from {SFT_CKPT}")
    reference, cfg = build_model_from_ckpt(SFT_CKPT, device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    n_params = policy.num_params()
    print(f"Policy parameters: {n_params:,} (~{n_params/1e6:.2f}M)")

    # 3) 优化器（只优化 policy）
    optim = torch.optim.AdamW(
        policy.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY,
    )

    # 4) 日志
    csv_path = os.path.join(OUT_DIR, "losses.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["iter", "train_loss", "val_loss",
                         "train_margin", "val_margin",
                         "train_acc", "val_acc", "lr"])
    csv_file.flush()

    splits = {"train": train_data, "val": val_data}
    t0 = time.time()
    best_val = float("inf")

    for it in range(1, MAX_ITERS + 1):
        lr = get_lr(it)
        for pg in optim.param_groups:
            pg["lr"] = lr

        cho_ids, cho_mask, rej_ids, rej_mask = get_batch(train_data, BATCH_SIZE, device)

        # ---- 4 次 forward ----
        pi_c = sequence_logprob(policy, cho_ids, cho_mask)
        pi_r = sequence_logprob(policy, rej_ids, rej_mask)
        with torch.no_grad():
            rf_c = sequence_logprob(reference, cho_ids, cho_mask)
            rf_r = sequence_logprob(reference, rej_ids, rej_mask)

        loss, _ = dpo_loss(pi_c, pi_r, rf_c, rf_r, BETA)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        optim.step()

        if it % EVAL_INTERVAL == 0 or it == 1:
            res = estimate_loss(policy, reference, splits, BATCH_SIZE, device)
            tr_l, tr_m, tr_a = res["train"]
            va_l, va_m, va_a = res["val"]
            dt = time.time() - t0
            print(
                f"iter {it:5d} | lr {lr:.2e} | "
                f"train loss {tr_l:.4f}  acc {tr_a:.2f} | "
                f"val loss {va_l:.4f}  acc {va_a:.2f} | "
                f"{dt:.1f}s"
            )
            csv_writer.writerow([it, tr_l, va_l, tr_m, va_m, tr_a, va_a, lr])
            csv_file.flush()

            if va_l < best_val:
                best_val = va_l
                torch.save(
                    {
                        "model":    policy.state_dict(),
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
