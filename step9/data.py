"""
data.py —— TinyStories 全量 + WikiText-103 混合数据集 (BPE)
============================================================
两份数据源：

1) TinyStories 全量（~1.5GB 文本）：
   GPT-3.5/GPT-4 生成的简单英文小故事，词汇浅、句子短。
   对训练小模型写故事非常有效。

2) WikiText-103-raw（~500MB 文本）：
   维基百科长文章，内容涵盖历史、科技、地理等。
   让模型学到更多世界知识和复杂句式。

合计 ~500M+ BPE tokens，是 step5 的 20 倍，对 120M 模型充分训练。

编码策略：
- 使用 tiktoken GPT-2 BPE（vocab=50257）
- 分块编码，避免一次性 encode 几 GB 文本导致内存爆炸
- 每个文档/故事之间插入 <|endoftext|> 分隔符
"""
import os
import pickle
import requests
import numpy as np
import tiktoken
from tqdm import tqdm

# ---- 路径 ----
DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")
STORIES_PATH    = os.path.join(DATA_DIR, "tinystories.txt")
WIKI_PATH       = os.path.join(DATA_DIR, "wikitext103.txt")
TRAIN_BIN       = os.path.join(DATA_DIR, "train.bin")
VAL_BIN         = os.path.join(DATA_DIR, "val.bin")
META_PATH       = os.path.join(DATA_DIR, "meta.pkl")

# ---- 下载源 ----
STORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"

# TinyStories：全量下载（设为整数可限制大小，如 500*1024*1024 = 500MB）
MAX_STORIES_BYTES = None

ENCODING_NAME = "gpt2"

# 分块编码：每次读取的字符数（避免大文件一次性加载到内存）
CHUNK_CHARS = 10 * 1024 * 1024  # 10MB per chunk


def _stream_download(url: str, dst: str, max_bytes: int | None = None):
    """流式下载到本地文件，可选限制最大字节数。"""
    print(f"Downloading {url}")
    limit_str = f"{max_bytes/1e6:.0f}MB" if max_bytes else "full"
    print(f"  -> {dst}  (cap={limit_str})")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        if max_bytes and total:
            total = min(total, max_bytes)
        downloaded = 0
        with open(dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                if max_bytes and downloaded + len(chunk) > max_bytes:
                    chunk = chunk[: max_bytes - downloaded]
                    f.write(chunk)
                    pbar.update(len(chunk))
                    break
                f.write(chunk)
                downloaded += len(chunk)
                pbar.update(len(chunk))


def _download_stories():
    """下载 TinyStories 全量。"""
    if os.path.exists(STORIES_PATH):
        print(f"TinyStories already exists: {STORIES_PATH}")
        return
    _stream_download(STORIES_URL, STORIES_PATH, max_bytes=MAX_STORIES_BYTES)


def _download_wiki():
    """通过 HuggingFace datasets 库下载 WikiText-103-raw 训练集。"""
    if os.path.exists(WIKI_PATH):
        print(f"WikiText-103 already exists: {WIKI_PATH}")
        return
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "需要安装 datasets 库来下载 WikiText-103：\n"
            "  pip install datasets\n"
            "或者你可以手动下载 wikitext-103 的 train split 放到：\n"
            f"  {WIKI_PATH}"
        )
    print("Downloading WikiText-103 via HuggingFace datasets ...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    print(f"  Got {len(ds):,} paragraphs, writing to {WIKI_PATH} ...")
    with open(WIKI_PATH, "w", encoding="utf-8") as f:
        for item in tqdm(ds, desc="Writing WikiText-103"):
            text = item["text"]
            if text.strip():  # 跳过空行
                f.write(text + "\n")
    print(f"  Saved: {WIKI_PATH} ({os.path.getsize(WIKI_PATH)/1e6:.0f}MB)")


def _encode_file_chunked(filepath: str, enc, desc: str = "") -> list[int]:
    """
    分块读取并编码大文件，避免一次性占用过多内存。
    返回完整的 token id 列表。
    """
    file_size = os.path.getsize(filepath)
    all_ids = []
    eot = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc=desc)
        while True:
            chunk = f.read(CHUNK_CHARS)
            if not chunk:
                break
            pbar.update(len(chunk.encode("utf-8", errors="ignore")))
            ids = enc.encode_ordinary(chunk)
            all_ids.extend(ids)
        pbar.close()

    # 文件末尾加 EOT 作为文档分隔
    all_ids.append(eot)
    return all_ids


def prepare():
    """下载数据 + BPE 编码 + 保存二进制文件。"""
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) 下载
    _download_stories()
    _download_wiki()

    # 2) 分块编码
    enc = tiktoken.get_encoding(ENCODING_NAME)
    vocab_size = enc.n_vocab
    print(f"\nTokenizer: {ENCODING_NAME}, vocab_size={vocab_size}")

    print("\n[1/2] Encoding TinyStories ...")
    stories_ids = _encode_file_chunked(STORIES_PATH, enc, desc="TinyStories")
    print(f"  TinyStories tokens: {len(stories_ids):,}")

    print("\n[2/2] Encoding WikiText-103 ...")
    wiki_ids = _encode_file_chunked(WIKI_PATH, enc, desc="WikiText-103")
    print(f"  WikiText-103 tokens: {len(wiki_ids):,}")

    # 3) 合并：故事在前，维基在后
    all_ids = stories_ids + wiki_ids
    total_tokens = len(all_ids)
    print(f"\nTotal BPE tokens: {total_tokens:,}")
    print(f"  = {total_tokens/1e6:.1f}M tokens")

    # 4) 90/10 切分
    split = int(total_tokens * 0.9)
    train_ids = np.array(all_ids[:split], dtype=np.uint16)
    val_ids   = np.array(all_ids[split:], dtype=np.uint16)
    print(f"Train: {len(train_ids):,} tokens")
    print(f"Val  : {len(val_ids):,} tokens")

    # 5) 保存
    train_ids.tofile(TRAIN_BIN)
    val_ids.tofile(VAL_BIN)
    with open(META_PATH, "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "encoding": ENCODING_NAME}, f)

    print(f"\nSaved: {TRAIN_BIN}")
    print(f"       {VAL_BIN}")
    print(f"       {META_PATH}")
    print("Done!")


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
