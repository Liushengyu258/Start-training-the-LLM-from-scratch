"""
data.py —— Anthropic HH-RLHF 偏好对数据准备 (用于 DPO)
============================================================
HH-RLHF 是 Anthropic 公开的人类反馈数据集，每条样本是一对：
    {
      "chosen":   "...Human: Q\\n\\nAssistant: 好答案",
      "rejected": "...Human: Q\\n\\nAssistant: 差答案"
    }
chosen 和 rejected 共享同样的对话历史（prompt），只在最后的回答上不同。

我们要做的：
  1) 下载 helpful-base 训练集（jsonl.gz, ~50MB）
  2) 对每条 (chosen, rejected) 用“最长公共前缀”切出 prompt
  3) 分别拼成 prompt+chosen 和 prompt+rejected 两条序列
  4) BPE 编码 + 生成 mask（只在最后回答的 token 上算 loss）
  5) 存成 4 个二进制：chosen_ids/chosen_mask + rejected_ids/rejected_mask
"""
import os
import gzip
import json
import pickle
import requests
import numpy as np
import tiktoken
from tqdm import tqdm

# ---- 路径 ----
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
RAW_GZ    = os.path.join(DATA_DIR, "hh_helpful_train.jsonl.gz")
CHO_IDS   = os.path.join(DATA_DIR, "chosen_ids.bin")
CHO_MASK  = os.path.join(DATA_DIR, "chosen_mask.bin")
REJ_IDS   = os.path.join(DATA_DIR, "rejected_ids.bin")
REJ_MASK  = os.path.join(DATA_DIR, "rejected_mask.bin")
META_PATH = os.path.join(DATA_DIR, "meta.pkl")

# ---- 下载源 ----
HH_URL = "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/helpful-base/train.jsonl.gz"

# ---- 配置 ----
BLOCK_SIZE    = 512
ENCODING_NAME = "gpt2"
EOS_ID        = 50256
VAL_RATIO     = 0.02
MAX_SAMPLES   = 20000   # 取前 N 条，避免编码太久。设 None 取全量
MIN_RESPONSE  = 4       # 截断后最少要剩多少个 response token，否则丢弃


def _download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_GZ):
        return
    print(f"Downloading HH-RLHF to {RAW_GZ} ...")
    with requests.get(HH_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(RAW_GZ, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                pbar.update(len(chunk))


def _longest_common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _split_prompt_response(chosen: str, rejected: str):
    """用最长公共前缀切出共享 prompt，剩下分别是 chosen / rejected 的回答。"""
    k = _longest_common_prefix(chosen, rejected)
    prompt = chosen[:k]
    cho_resp = chosen[k:]
    rej_resp = rejected[k:]
    return prompt, cho_resp, rej_resp


def _encode_one(enc, prompt_ids, response_text):
    """把一条 (prompt, response) 编码 + pad，返回 (ids, mask)。
       mask 只在 response 段为 1。截断后若 response 不足 MIN_RESPONSE，返回 None 表示丢弃。"""
    resp_ids = enc.encode_ordinary(response_text) + [EOS_ID]
    full = prompt_ids + resp_ids
    if len(full) > BLOCK_SIZE:
        full = full[:BLOCK_SIZE]
    response_kept = len(full) - len(prompt_ids)
    if response_kept < MIN_RESPONSE:
        return None
    mask = [0] * len(prompt_ids) + [1] * response_kept
    pad = BLOCK_SIZE - len(full)
    full = full + [EOS_ID] * pad
    mask = mask + [0] * pad
    return full, mask


def prepare():
    _download()
    enc = tiktoken.get_encoding(ENCODING_NAME)

    cho_ids_list,  cho_mask_list  = [], []
    rej_ids_list,  rej_mask_list  = [], []
    dropped = 0

    with gzip.open(RAW_GZ, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(tqdm(f, desc="Encoding pairs")):
            if MAX_SAMPLES and line_no >= MAX_SAMPLES:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            chosen   = obj.get("chosen",   "")
            rejected = obj.get("rejected", "")
            if not chosen or not rejected or chosen == rejected:
                continue

            prompt, cho_resp, rej_resp = _split_prompt_response(chosen, rejected)
            if not cho_resp.strip() or not rej_resp.strip():
                continue

            # prompt 只编码一次共用
            prompt_ids = enc.encode_ordinary(prompt)
            if len(prompt_ids) >= BLOCK_SIZE - MIN_RESPONSE:
                dropped += 1
                continue

            cho = _encode_one(enc, prompt_ids, cho_resp)
            rej = _encode_one(enc, prompt_ids, rej_resp)
            if cho is None or rej is None:
                dropped += 1
                continue

            cho_ids_list.append(cho[0]);  cho_mask_list.append(cho[1])
            rej_ids_list.append(rej[0]);  rej_mask_list.append(rej[1])

    print(f"Kept {len(cho_ids_list)} pairs, dropped {dropped}")

    cho_ids  = np.array(cho_ids_list,  dtype=np.uint16)
    cho_mask = np.array(cho_mask_list, dtype=np.uint8)
    rej_ids  = np.array(rej_ids_list,  dtype=np.uint16)
    rej_mask = np.array(rej_mask_list, dtype=np.uint8)

    # 打乱并切 train/val
    rng = np.random.default_rng(1337)
    perm = rng.permutation(len(cho_ids))
    cho_ids, cho_mask = cho_ids[perm], cho_mask[perm]
    rej_ids, rej_mask = rej_ids[perm], rej_mask[perm]

    n_val = max(1, int(len(cho_ids) * VAL_RATIO))
    n_train = len(cho_ids) - n_val

    # 保存：前 n_train 为 train，后 n_val 为 val
    cho_ids.tofile(CHO_IDS);   cho_mask.tofile(CHO_MASK)
    rej_ids.tofile(REJ_IDS);   rej_mask.tofile(REJ_MASK)

    with open(META_PATH, "wb") as f:
        pickle.dump({
            "vocab_size": enc.n_vocab,
            "encoding":   ENCODING_NAME,
            "block_size": BLOCK_SIZE,
            "n_train":    n_train,
            "n_val":      n_val,
            "eos_id":     EOS_ID,
        }, f)

    avg_cho = cho_mask.sum(1).mean()
    avg_rej = rej_mask.sum(1).mean()
    print(f"Avg response tokens — chosen: {avg_cho:.1f}  rejected: {avg_rej:.1f}")
    print(f"Saved: train={n_train}  val={n_val}")


def load_meta():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


def load_split(split: str):
    meta = load_meta()
    T = meta["block_size"]
    n_train = meta["n_train"]
    cho_ids  = np.memmap(CHO_IDS,  dtype=np.uint16, mode="r").reshape(-1, T)
    cho_mask = np.memmap(CHO_MASK, dtype=np.uint8,  mode="r").reshape(-1, T)
    rej_ids  = np.memmap(REJ_IDS,  dtype=np.uint16, mode="r").reshape(-1, T)
    rej_mask = np.memmap(REJ_MASK, dtype=np.uint8,  mode="r").reshape(-1, T)
    if split == "train":
        return cho_ids[:n_train], cho_mask[:n_train], rej_ids[:n_train], rej_mask[:n_train]
    return cho_ids[n_train:], cho_mask[n_train:], rej_ids[n_train:], rej_mask[n_train:]


def get_encoder():
    return tiktoken.get_encoding(ENCODING_NAME)


if __name__ == "__main__":
    prepare()
