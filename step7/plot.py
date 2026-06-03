"""DPO 训练曲线：loss / reward_margin / accuracy / lr。"""
import csv
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CSV_PATH = os.path.join(HERE, "out", "losses.csv")
PNG_PATH = os.path.join(HERE, "out", "loss_curve.png")


def main():
    iters = []
    tr_loss, va_loss = [], []
    tr_mar,  va_mar  = [], []
    tr_acc,  va_acc  = [], []
    lrs = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row["iter"]))
            tr_loss.append(float(row["train_loss"]))
            va_loss.append(float(row["val_loss"]))
            tr_mar .append(float(row["train_margin"]))
            va_mar .append(float(row["val_margin"]))
            tr_acc .append(float(row["train_acc"]))
            va_acc .append(float(row["val_acc"]))
            lrs    .append(float(row["lr"]))

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(iters, tr_loss, label="train", marker="o")
    axes[0].plot(iters, va_loss, label="val",   marker="s")
    axes[0].set_ylabel("DPO loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title("DPO Training Curves")

    axes[1].plot(iters, tr_mar, label="train", marker="o")
    axes[1].plot(iters, va_mar, label="val",   marker="s")
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("reward margin\n(chosen - rejected)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(iters, tr_acc, label="train", marker="o")
    axes[2].plot(iters, va_acc, label="val",   marker="s")
    axes[2].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("accuracy\n(chosen > rejected)")
    axes[2].set_ylim(0, 1)
    axes[2].legend(); axes[2].grid(alpha=0.3)

    axes[3].plot(iters, lrs, color="orange")
    axes[3].set_ylabel("lr"); axes[3].set_xlabel("iteration")
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=120)
    print(f"saved: {PNG_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
