"""
data.py —— 数据准备脚本
================================
作用：
1) 从网上下载 TinyShakespeare（莎士比亚剧本，纯文本，约 1MB）
2) 构建“字符级”词表（vocab）：把每个唯一字符映射到一个整数 id
3) 把整本文本切成 90% 训练集 / 10% 验证集
4) 把字符 id 序列以 uint16 二进制保存到磁盘（方便训练时用 memmap 高速读取）
5) 把词表（stoi / itos）和 vocab_size 用 pickle 保存

什么是“字符级”分词？
    最简单的分词法：一个字符就是一个 token。
    比如 "Hi!" → ['H', 'i', '!'] → [23, 47, 5]
    优点：实现极简，词表很小（这里只有 65）。
    缺点：序列变长、模型要从字符学语法，效率不如 BPE/WordPiece。
    对 1M 小模型 + 玩具数据来说足够。
"""
import os                              # 操作文件路径
import pickle                          # 把 Python 对象（词表）序列化保存到磁盘
import requests                        # 发 HTTP 请求下载数据
import numpy as np                     # 高速数值数组，用来存 token id

# ---- 关键路径常量 ----
# karpathy 大神 char-rnn 仓库里的 tinyshakespeare 文本 raw 链接
DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")  # 当前脚本同级的 data/ 目录
RAW_PATH  = os.path.join(DATA_DIR, "input.txt")              # 原始文本
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")              # 训练集 token id（二进制）
VAL_BIN   = os.path.join(DATA_DIR, "val.bin")                # 验证集 token id（二进制）
META_PATH = os.path.join(DATA_DIR, "meta.pkl")               # 词表元信息（pickle）


def prepare():
    """下载 & 预处理。重复调用会跳过已下载的文件。"""
    os.makedirs(DATA_DIR, exist_ok=True)        # 确保 data/ 存在

    # 1) 下载原始文本
    if not os.path.exists(RAW_PATH):
        print(f"Downloading TinyShakespeare to {RAW_PATH} ...")
        r = requests.get(DATA_URL, timeout=30)  # 发 GET 请求
        r.raise_for_status()                    # 下载失败抛异常
        with open(RAW_PATH, "w", encoding="utf-8") as f:
            f.write(r.text)                     # 保存为 UTF-8 文本

    # 2) 读文本到内存（仅 ~1MB，无压力）
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Dataset length (chars): {len(text)}")  # ~1,115,394 个字符

    # 3) 构建字符级词表
    chars = sorted(list(set(text)))             # 所有不重复字符并排序，保证可复现
    vocab_size = len(chars)                     # 这里是 65
    print(f"Vocab size: {vocab_size}")
    stoi = {ch: i for i, ch in enumerate(chars)}    # string → int   '!' -> 5
    itos = {i: ch for i, ch in enumerate(chars)}    # int → string   5  -> '!'

    # 4) 划分训练 / 验证 集（前 90% 训练，后 10% 验证）
    n = len(text)
    train_text = text[: int(n * 0.9)]
    val_text   = text[int(n * 0.9):]

    # 5) 字符 → id 序列。uint16 足够装下 65 这么小的词表（最多 65535）
    train_ids = np.array([stoi[c] for c in train_text], dtype=np.uint16)
    val_ids   = np.array([stoi[c] for c in val_text],   dtype=np.uint16)

    # 6) 二进制保存。用 .tofile 写出原始字节流，训练时再用 np.memmap 映射读取。
    #    这样几百万 token 也能秒读，且不占 RAM（按需加载页）。
    train_ids.tofile(TRAIN_BIN)
    val_ids.tofile(VAL_BIN)

    # 7) 词表元信息（推理时也要用，要解码 id→char）
    with open(META_PATH, "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)

    print(f"Saved train={len(train_ids)} val={len(val_ids)} tokens.")


def load_meta():
    """读出词表元信息。返回 dict: {vocab_size, stoi, itos}"""
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


def load_split(split: str) -> np.ndarray:
    """
    用 memmap 打开 train.bin / val.bin。
    memmap 不会一次把整个文件读进内存，而是按需读；
    对几 MB 的数据其实差别不大，但这是大数据集的通用模式。
    """
    path = TRAIN_BIN if split == "train" else VAL_BIN
    return np.memmap(path, dtype=np.uint16, mode="r")


# 直接 `python data.py` 时运行 prepare()
if __name__ == "__main__":
    prepare()
