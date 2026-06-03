"""
data.py —— Alpaca 指令微调数据准备
============================================================
做的事：
  1) 下载 Stanford Alpaca（52K 英文指令样本，约 25MB JSON）
  2) 用 Alpaca 官方 prompt 模板把 (instruction, input, output) 拼成一段文本
  3) 用 GPT-2 BPE 编码
  4) 关键：生成 loss_mask，prompt 部分置 0、response 部分置 1
         训练时只在 response 上计算 loss（这是 SFT 与预训练的本质区别）
  5) 保存到 ids.bin / mask.bin（二维数组：N 条样本 × block_size）

为什么只在 response 上算 loss？
    预训练阶段：模型“预测下一个 token”，无论上下文是什么。
    SFT 阶段：只想让模型在看到 prompt 后生成正确的 response，
              所以让 prompt 不贡献梯度，模型不会被强迫去“背诵”prompt 写法。
"""
import os
import json
import pickle
import requests
import numpy as np
import tiktoken
from tqdm import tqdm

# ---- 路径 ----
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
RAW_PATH  = os.path.join(DATA_DIR, "alpaca_data.json")
IDS_BIN   = os.path.join(DATA_DIR, "ids.bin")     # (N, block_size) uint16
MASK_BIN  = os.path.join(DATA_DIR, "mask.bin")    # (N, block_size) uint8
META_PATH = os.path.join(DATA_DIR, "meta.pkl")

# ---- 下载源 ----
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

# ---- 配置 ----
BLOCK_SIZE    = 512                # 与预训练一致；超长样本截断
ENCODING_NAME = "gpt2"
EOS_ID        = 50256              # GPT-2 BPE 中 endoftext 的 id，用作 EOS / pad
VAL_RATIO     = 0.02               # 验证集比例（监控过拟合）

# Alpaca 官方 prompt 模板（两种：有 / 无 input）
PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def build_prompt(instruction: str, input_text: str) -> str:
    """根据是否有 input 选模板，并填充。"""
    if input_text and input_text.strip():
        return PROMPT_WITH_INPUT.format(instruction=instruction.strip(),
                                        input=input_text.strip())
    return PROMPT_NO_INPUT.format(instruction=instruction.strip())


def _download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_PATH):
        return
    print(f"Downloading Alpaca to {RAW_PATH} ...")
    r = requests.get(ALPACA_URL, timeout=60)
    r.raise_for_status()
    with open(RAW_PATH, "wb") as f:
        f.write(r.content)


def prepare():
    _download()
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} Alpaca samples")

    enc = tiktoken.get_encoding(ENCODING_NAME)

    all_ids  = []
    all_mask = []
    dropped  = 0

    for s in tqdm(samples, desc="Encoding"):
        prompt   = build_prompt(s["instruction"], s.get("input", ""))
        response = s["output"].strip()

        # 编码
        prompt_ids   = enc.encode_ordinary(prompt)
        response_ids = enc.encode_ordinary(response) + [EOS_ID]   # 末尾加 EOS

        full_ids = prompt_ids + response_ids
        # 截断
        if len(full_ids) > BLOCK_SIZE:
            full_ids = full_ids[:BLOCK_SIZE]

        # 如果截断后 response 一个 token 都不剩，丢弃
        if len(prompt_ids) >= len(full_ids):
            dropped += 1
            continue

        # 构造 loss_mask：prompt 段 0，response 段 1
        mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))

        # 右侧 pad 到 BLOCK_SIZE（pad 也设为 EOS_ID，mask 设 0 不算 loss）
        pad_len = BLOCK_SIZE - len(full_ids)
        full_ids = full_ids + [EOS_ID] * pad_len
        mask     = mask     + [0]      * pad_len

        all_ids .append(full_ids)
        all_mask.append(mask)

    print(f"Kept {len(all_ids)} samples, dropped {dropped} too-long ones.")

    ids_arr  = np.array(all_ids,  dtype=np.uint16)   # (N, BLOCK_SIZE)
    mask_arr = np.array(all_mask, dtype=np.uint8)    # (N, BLOCK_SIZE)

    # 随机打乱并切 train / val
    rng = np.random.default_rng(1337)
    perm = rng.permutation(len(ids_arr))
    ids_arr, mask_arr = ids_arr[perm], mask_arr[perm]

    n_val = max(1, int(len(ids_arr) * VAL_RATIO))
    val_ids,   train_ids  = ids_arr[:n_val],  ids_arr[n_val:]
    val_mask,  train_mask = mask_arr[:n_val], mask_arr[n_val:]

    # 保存：把 train/val 拼成一个文件，前 n_train 行为 train，后 n_val 行为 val
    np.concatenate([train_ids,  val_ids ]).tofile(IDS_BIN)
    np.concatenate([train_mask, val_mask]).tofile(MASK_BIN)

    with open(META_PATH, "wb") as f:
        pickle.dump({
            "vocab_size": enc.n_vocab,
            "encoding":   ENCODING_NAME,
            "block_size": BLOCK_SIZE,
            "n_train":    len(train_ids),
            "n_val":      len(val_ids),
            "eos_id":     EOS_ID,
        }, f)

    # 平均 response 长度 / mask 利用率（看看 loss 计算的有效 token 占比）
    avg_resp = mask_arr.sum(axis=1).mean()
    print(f"Avg response tokens per sample: {avg_resp:.1f} / {BLOCK_SIZE}")
    print(f"Saved: train={len(train_ids)}  val={len(val_ids)}")


def load_meta():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


def load_split(split: str):
    """返回 (ids, mask)，都是 (N, block_size) 的 memmap。"""
    meta = load_meta()
    block_size = meta["block_size"]
    n_train, n_val = meta["n_train"], meta["n_val"]

    ids_full  = np.memmap(IDS_BIN,  dtype=np.uint16, mode="r").reshape(-1, block_size)
    mask_full = np.memmap(MASK_BIN, dtype=np.uint8,  mode="r").reshape(-1, block_size)

    if split == "train":
        return ids_full[:n_train], mask_full[:n_train]
    return ids_full[n_train:], mask_full[n_train:]


def get_encoder():
    return tiktoken.get_encoding(ENCODING_NAME)


if __name__ == "__main__":
    prepare()
