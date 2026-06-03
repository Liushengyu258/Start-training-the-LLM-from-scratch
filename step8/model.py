"""
model.py —— GPT + RoPE + RMSNorm + SwiGLU + KV-cache
============================================================
KV-cache 是什么？
    在自回归生成时，我们一个 token 一个 token 地往外吐：
        step t :  喂 [t0, t1, ..., t_{t-1}] 进模型，拿最后一位 logits 采样
        step t+1: 喂 [t0, t1, ..., t_t]     进模型，又重头算一次
    每层 attention 里，过去那些位置的 K、V 已经算过了，没必要每次重算。

    思路：把每层 attention 算出来的 K、V 缓存下来，
    下一步只需要：
        1) 给最新 1 个 token 算它的 Q、K、V
        2) 把新 K、V 拼到缓存末尾
        3) 用新 Q 与“缓存 + 新 K、V”做注意力

    速度收益：把每次 O(T) 变成 O(1)，整段生成从 O(T^2) 降到 O(T)。
    显存代价：每层多存 (B, n_head, T, head_dim) 的 K 和 V。

代码层面与 step7 model 的差别：
    1) CausalSelfAttention.forward 多两个参数：past_kv、use_cache，可返回 new_kv
    2) Block.forward 把 cache 透传到 attention
    3) GPT.forward 多参数 past_kvs (list, 每层一个) 和 use_cache，可返回 new_kvs
    4) RoPE 需要知道当前 token 的【绝对位置】，所以从 GPT.forward 里按
       start_pos = past 的长度 切对应的 freqs_cis
    5) 新增 GPT.generate_cached：基于 KV-cache 的高速生成
       原 generate（无 cache，逐步重算）仍保留以便对比

权重完全兼容 step5/6/7 的 ckpt —— KV-cache 是【推理时的运行机制】，不增删任何参数。
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size:    int   = 50257
    block_size:    int   = 512
    n_layer:       int   = 6
    n_head:        int   = 6
    n_embd:        int   = 192
    ffn_hidden:    int | None = None
    multiple_of:   int   = 64
    dropout:       float = 0.1
    rope_theta:    float = 10000.0
    norm_eps:      float = 1e-5


# ---------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


# ---------------------------------------------------------------------
# RoPE 工具
# ---------------------------------------------------------------------
def precompute_freqs_cis(head_dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    fc = freqs_cis.view(1, 1, freqs_cis.shape[0], -1)
    xq_out = torch.view_as_real(xq_c * fc).flatten(-2)
    xk_out = torch.view_as_real(xk_c * fc).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------
# Attention（带 KV-cache）
# ---------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head   = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_embd   = cfg.n_embd

        self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop  = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # 用于 prefill 阶段的下三角因果 mask
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)) \
                    .view(1, 1, cfg.block_size, cfg.block_size)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, freqs_cis, past_kv=None, use_cache=False):
        """
        x:        (B, T_new, C) —— 本次要喂进去的新 token（prefill 时是整段 prompt）
        freqs_cis:(T_new, head_dim/2) —— 对应这 T_new 个位置的旋转
        past_kv:  None 或 (K_cache, V_cache)，K_cache shape (B, H, T_past, head_dim)
        use_cache:是否返回新的 KV-cache
        """
        B, T_new, C = x.shape

        # 1) 算新 Q、K、V
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T_new, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T_new, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T_new, self.n_head, self.head_dim).transpose(1, 2)

        # 2) RoPE：q、k 都要旋转（v 不旋转）
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # 3) 如有 cache，拼接到时间维
        if past_kv is not None:
            k_cache, v_cache = past_kv
            k = torch.cat([k_cache, k], dim=2)      # (B, H, T_past+T_new, head_dim)
            v = torch.cat([v_cache, v], dim=2)

        new_kv = (k, v) if use_cache else None

        # 4) 注意力分数
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 5) Mask 规则：
        #    - 增量解码 (T_new == 1)：query 在最末位置，可看所有过去 + 自己，不需要 mask
        #    - prefill / 训练 (past_kv is None)：标准下三角 causal mask
        #    - 同时既有 cache 又一次喂多个 token 的情况，本实现暂不支持
        if past_kv is None:
            att = att.masked_fill(self.mask[:, :, :T_new, :T_new] == 0, float("-inf"))
        # 否则不加 mask（保证调用方满足 T_new == 1）

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v                                 # (B, H, T_new, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T_new, C)
        return self.resid_drop(self.proj(y)), new_kv


# ---------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.n_embd
        if cfg.ffn_hidden is not None:
            hidden = cfg.ffn_hidden
        else:
            hidden = int(2 * 4 * d / 3)
            m = cfg.multiple_of
            hidden = ((hidden + m - 1) // m) * m
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up   = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
        self.drop   = nn.Dropout(cfg.dropout)
        self.hidden = hidden

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ---------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.mlp  = SwiGLU(cfg)

    def forward(self, x, freqs_cis, past_kv=None, use_cache=False):
        h, new_kv = self.attn(self.ln1(x), freqs_cis, past_kv=past_kv, use_cache=use_cache)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


# ---------------------------------------------------------------------
# 顶层 GPT
# ---------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f   = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.head   = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        head_dim  = cfg.n_embd // cfg.n_head
        freqs_cis = precompute_freqs_cis(head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, past_kvs=None, use_cache=False):
        """
        idx:       (B, T_new)  本次喂入的 token
        past_kvs:  list[(K, V)] 或 None；list 长度 = n_layer
        use_cache: 是否返回新的 past_kvs
        返回：
          - 训练 / 普通推理（use_cache=False）：(logits, loss)
          - 带 cache：               (logits, loss, new_past_kvs)
        """
        B, T_new = idx.shape
        # 当前段的起始位置 = 已缓存的 token 数
        start_pos = 0 if past_kvs is None else past_kvs[0][0].shape[2]
        assert start_pos + T_new <= self.cfg.block_size, \
            f"序列总长 {start_pos + T_new} 超过 block_size {self.cfg.block_size}"

        x = self.drop(self.tok_emb(idx))
        # 只取当前段对应的旋转角度
        freqs_cis = self.freqs_cis[start_pos: start_pos + T_new]

        new_past_kvs = [] if use_cache else None
        for i, blk in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            x, new_kv = blk(x, freqs_cis, past_kv=past, use_cache=use_cache)
            if use_cache:
                new_past_kvs.append(new_kv)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        if use_cache:
            return logits, loss, new_past_kvs
        return logits, loss

    # -----------------------------------------------------------------
    # 朴素生成：每步都把整个序列重头算（O(T^2) 计算量）
    # 与 step1~7 的 generate 等价，用于对照
    # -----------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    # -----------------------------------------------------------------
    # KV-cache 生成：prefill + 逐 token 增量解码
    # -----------------------------------------------------------------
    @torch.no_grad()
    def generate_cached(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                        on_token=None):
        """
        on_token: 可选回调 fn(new_token_id_int) -> None，用于流式打印
        """
        B = idx.shape[0]

        # 1) Prefill：一次性把 prompt 全喂进去，建立每层 KV-cache
        logits, _, past_kvs = self(idx, use_cache=True)
        next_token = self._sample(logits[:, -1, :], temperature, top_k)
        out = torch.cat([idx, next_token], dim=1)
        if on_token is not None:
            on_token(int(next_token[0].item()))

        # 2) Decode：每次只喂 1 个 token，复用 cache
        for _ in range(max_new_tokens - 1):
            # 注意：如果 cache 长度即将超过 block_size，需要截断或滑窗。
            # 这里做最简单的处理：直接超限报错 / 截断到 block_size-1（保留最近）
            if past_kvs[0][0].shape[2] >= self.cfg.block_size:
                # 简单的滑窗：丢弃最早的若干位置（仅作演示，未处理 RoPE 位置漂移）
                # 生产用法应采用 sliding-window attention 等方案
                past_kvs = [(k[:, :, 1:, :], v[:, :, 1:, :]) for (k, v) in past_kvs]
            logits, _, past_kvs = self(next_token, past_kvs=past_kvs, use_cache=True)
            next_token = self._sample(logits[:, -1, :], temperature, top_k)
            out = torch.cat([out, next_token], dim=1)
            if on_token is not None:
                on_token(int(next_token[0].item()))
        return out

    @staticmethod
    def _sample(logits, temperature, top_k):
        logits = logits / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
