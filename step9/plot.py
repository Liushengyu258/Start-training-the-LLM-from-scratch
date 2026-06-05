"""
plot.py —— 绘制训练损失曲线
============================================================
从 out/losses.csv 读取训练日志，绘制 train/val loss 曲线。
支持训练进行中实时绘制（CSV 是边训练边写的）。

用法：
    python plot.py
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")   # 无 GUI 环境也能跑
import matplotlib.pyplot as plt

OUT_DIR  = os.path.join(os.path.dirname(__file__), "out")
CSV_PATH = os.path.join(OUT_DIR, "losses.csv")
PNG_PATH = os.path.join(OUT_DIR, "loss_curve.png")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"找不到 {CSV_PATH}，请先训练。")
        return

    iters, train_losses, val_losses, lrs = [], [], [], []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row["iter"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            lrs.append(float(row["lr"]))

    if not iters:
        print("CSV 文件为空。")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 上图：Loss
    ax1.plot(iters, train_losses, label="train loss", alpha=0.8)
    ax1.plot(iters, val_losses,   label="val loss",   alpha=0.8)
    ax1.set_ylabel("Loss")
    ax1.set_title("Step 9: 120M Model Training Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 下图：Learning Rate
    ax2.plot(iters, lrs, color="green", alpha=0.8)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Learning Rate")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    print(f"Saved plot to {PNG_PATH}")


if __name__ == "__main__":
    main()
