"""读取训练时保存的 out/losses.csv，绘制 train / val loss 曲线。

用法：
    python plot.py            # 显示并保存 out/loss_curve.png
"""
import csv
import os
import matplotlib.pyplot as plt

# ---- 文件路径 ----
HERE = os.path.dirname(__file__)            # 当前脚本所在目录
CSV_PATH = os.path.join(HERE, "out", "losses.csv")    # 损失日志输入
PNG_PATH = os.path.join(HERE, "out", "loss_curve.png")  # 图像输出


def main():
    # 1) 读 CSV：每行 = 一次评估，字段 iter / train_loss / val_loss / lr
    iters, train_losses, val_losses, lrs = [], [], [], []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)          # 自动按表头解析成字典
        for row in reader:
            iters.append(int(row["iter"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            lrs.append(float(row["lr"]))

    # 2) 画两张子图：上=loss，下=学习率
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(iters, train_losses, label="train loss", marker="o")
    ax1.plot(iters, val_losses,   label="val loss",   marker="s")
    ax1.set_ylabel("loss (cross-entropy)")
    ax1.set_title("Training Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, lrs, color="orange", label="learning rate")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("lr")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=120)
    print(f"已保存曲线到: {PNG_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
