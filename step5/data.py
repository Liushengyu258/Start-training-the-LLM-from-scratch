"""
data.py —— Shakespeare + TinyStories 混合数据集 (BPE)
============================================================
两份原始数据：
  1) TinyShakespeare （~1MB，莎士比亚戏剧）
  2) TinyStories     （~1.5GB 原始 txt，GPT-3.5/GPT-4 生成的简单英文小故事，
                       专门为训练小模型而生：词汇浅、句子短、覆盖儿童故事题材）

为了控制下载和编码耗时，TinyStories 默认只取前 MAX_STORIES_BYTES 字节
（默认 100MB ≈ 2500 万 BPE token，对 10~20M 模型已绰绰有余）。
你可以调大它甚至设为 None 取全量。
"""
import os
import pickle
import requests
import numpy as np
import tiktoken
from tqdm import tqdm

# ---- 路径 ----
DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")
SHAKE_PATH      = os.path.join(DATA_DIR, "shakespeare.txt")
STORIES_PATH    = os.path.join(DATA_DIR, "tinystories.txt")
TRAIN_BIN       = os.path.join(DATA_DIR, "train.bin")
VAL_BIN         = os.path.join(DATA_DIR, "val.bin")
META_PATH       = os.path.join(DATA_DIR, "meta.pkl")

# ---- 下载源 ----
SHAKE_URL   = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# HuggingFace 上 TinyStories 的官方原始 txt（roneneldan/TinyStories）
STORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"

# 截取上限：100MB。设为 None 表示全量下载（~1.5GB）
MAX_STORIES_BYTES = 100 * 1024 * 1024

ENCODING_NAME = "gpt2"


def _stream_download(url: str, dst: str, max_bytes: int | None):
    """流式下载到本地文件，可选限制最大字节数。带 tqdm 进度条。"""
    print(f"Downloading {url}")
    print(f"  -> {dst}  (cap={max_bytes/1e6 if max_bytes else 'full'} MB)")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        if max_bytes:
            total = min(total, max_bytes) if total else max_bytes
        downloaded = 0
        with open(dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):    # 1MB / chunk
                if not chunk:
                    continue
                if max_bytes and downloaded + len(chunk) > max_bytes:
                    # 把这一块截掉一部分，正好达到上限就停
                    chunk = chunk[: max_bytes - downloaded]
                    f.write(chunk)
                    pbar.update(len(chunk))
                    break
                f.write(chunk)
                downloaded += len(chunk)
                pbar.update(len(chunk))


def _ensure_downloads():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SHAKE_PATH):
        _stream_download(SHAKE_URL, SHAKE_PATH, max_bytes=None)
    if not os.path.exists(STORIES_PATH):
        _stream_download(STORIES_URL, STORIES_PATH, max_bytes=MAX_STORIES_BYTES)


def prepare():
    _ensure_downloads()

    # 读两份文本
    with open(SHAKE_PATH, "r", encoding="utf-8") as f:
        shake = f.read()
    with open(STORIES_PATH, "r", encoding="utf-8", errors="ignore") as f:
        stories = f.read()
    print(f"Shakespeare chars : {len(shake):,}")
    print(f"TinyStories chars : {len(stories):,}")

    # 拼接：故事在前（量大、风格简单），莎翁在后；中间加分隔
    full_text = stories + "\n\n" + shake
    print(f"Combined chars    : {len(full_text):,}")

    # BPE 编码
    enc = tiktoken.get_encoding(ENCODING_NAME)
    vocab_size = enc.n_vocab
    print(f"Tokenizer: {ENCODING_NAME}, vocab_size={vocab_size}")
    print("BPE encoding (may take a minute) ...")
    ids = enc.encode_ordinary(full_text)
    print(f"Total BPE tokens  : {len(ids):,}")

    # 90/10 切分
    n = len(ids)
    split = int(n * 0.9)
    train_ids = np.array(ids[:split], dtype=np.uint16)
    val_ids   = np.array(ids[split:], dtype=np.uint16)
    print(f"train tokens={len(train_ids):,}    val tokens={len(val_ids):,}")

    train_ids.tofile(TRAIN_BIN)
    val_ids.tofile(VAL_BIN)
    with open(META_PATH, "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "encoding": ENCODING_NAME}, f)
    print("Saved bin files & meta.")


def load_meta():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


def load_split(split: str) -> np.ndarray:
    path = TRAIN_BIN if split == "train" else VAL_BIN
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_encoder():
    return tiktoken.get_encoding(ENCODING_NAME)


if __name__ == "__main__":
    prepare()
