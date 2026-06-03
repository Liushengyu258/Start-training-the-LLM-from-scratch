"""
data.py —— 数据准备（BPE 分词版）
====================================
和 step1 的差别：分词从“字符级”升级为 BPE（GPT-2 用的同一套）。

什么是 BPE？
    Byte-Pair Encoding。先把语料按字节拆开，统计最常出现的相邻对，
    把它们合并成一个新 token；不断迭代，最后得到一份“子词”词表。
    例子：'tokenization' 可能被拆成 ['token', 'ization'] 而不是逐字符。

为什么换 BPE？
    1) 序列更短：1 个 token 约等于 3~4 个字符，相同 block_size 看的上下文更长。
    2) 语义更好：常见词是一整个 token，模型不用从字母拼起来。
    3) 兼容 GPT-2：可以直接用 tiktoken 的预训练词表，省得自己训分词器。

代价：
    词表从 65 暴涨到 50257，token 嵌入参数量随之大涨。
"""
import os
import pickle
import requests
import numpy as np
import tiktoken                          # OpenAI 官方 BPE 实现，速度很快

# ---- 路径常量 ----
DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
RAW_PATH  = os.path.join(DATA_DIR, "input.txt")
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN   = os.path.join(DATA_DIR, "val.bin")
META_PATH = os.path.join(DATA_DIR, "meta.pkl")

# 使用 GPT-2 的 BPE。tiktoken 第一次调用会自动下载 vocab 文件并缓存。
ENCODING_NAME = "gpt2"


def prepare():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) 下载原始文本（与 step1 相同）
    if not os.path.exists(RAW_PATH):
        print(f"Downloading TinyShakespeare to {RAW_PATH} ...")
        r = requests.get(DATA_URL, timeout=30)
        r.raise_for_status()
        with open(RAW_PATH, "w", encoding="utf-8") as f:
            f.write(r.text)
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Dataset length (chars): {len(text)}")

    # 2) 加载 BPE 分词器
    enc = tiktoken.get_encoding(ENCODING_NAME)
    vocab_size = enc.n_vocab            # gpt2 = 50257
    print(f"Tokenizer: {ENCODING_NAME}, vocab_size={vocab_size}")

    # 3) 切分 train/val
    n = len(text)
    train_text = text[: int(n * 0.9)]
    val_text   = text[int(n * 0.9):]

    # 4) 编码：字符串 -> token id 列表
    #    encode_ordinary 表示不处理特殊 token（速度更快），普通文本用它就够了
    train_ids = enc.encode_ordinary(train_text)
    val_ids   = enc.encode_ordinary(val_text)
    print(f"train tokens: {len(train_ids):,}    val tokens: {len(val_ids):,}")

    # 5) 用 uint16 存储（够用：50257 < 65535）
    np.array(train_ids, dtype=np.uint16).tofile(TRAIN_BIN)
    np.array(val_ids,   dtype=np.uint16).tofile(VAL_BIN)

    # 6) 保存 meta：只需要 vocab_size 和 encoding 名称（解码时再起 tiktoken 即可）
    with open(META_PATH, "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "encoding": ENCODING_NAME}, f)
    print("Done.")


def load_meta():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


def load_split(split: str) -> np.ndarray:
    path = TRAIN_BIN if split == "train" else VAL_BIN
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_encoder():
    """返回 tiktoken 编码器，generate.py 解码时会用。"""
    return tiktoken.get_encoding(ENCODING_NAME)


if __name__ == "__main__":
    prepare()
