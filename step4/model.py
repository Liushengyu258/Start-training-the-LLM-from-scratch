"""
model.py —— GPT + RoPE + RMSNorm
============================================================
相对 step3 的唯一改动：把所有 nn.LayerNorm 替换为自定义的 RMSNorm。

什么是 RMSNorm？
    LayerNorm 公式（对每个 token 的特征向量 x 做归一化）：
        mean = x.mean()
        var  = x.var()
        x_hat = (x - mean) / sqrt(var + eps)
        y = gamma * x_hat + beta            # 两个可学习参数：gamma、beta

    RMSNorm 公式：
        rms   = sqrt( mean(x^2) + eps )
        x_hat = x / rms
        y = gamma * x_hat                    # 只有 gamma，没有 beta，也不减均值

直观差别：
    1) 省掉减均值的步骤 → 计算更快
    2) 没有 beta 偏置参数 → 更少参数
    3) 实验上效果与 LayerNorm 持平或更好
    4) LLaMA、Qwen、Mistral 等现代 LLM 全部默认 RMSNorm

代码层面：替换文件里所有 nn.LayerNorm(d) → RMSNorm(d) 即可，其它结构原封不动。
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
    vocab_size: int   = 50257
    block_size: int   = 512
    n_layer:    int   = 6
    n_head:     int   = 6
    n_embd:     int   = 192
    dropout:    float = 0.1
    rope_theta: float = 10000.0
    norm_eps:   float = 1e-5       # RMSNorm 数值稳定项


# ---------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------
class RMSNorm(nn.Module):
    """
    LLaMA 风格的 RMSNorm。
        y = x * rsqrt( mean(x^2, dim=-1) + eps ) * weight
    其中 weight (gamma) 是可学习的缩放向量，形状 = (dim,)。
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 初始化为 1，等价于一开始相当于纯归一化（没有缩放偏置）
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # rsqrt(x) = 1 / sqrt(x)，比写两步更快、更稳
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先在 float32 下做归一化避免半精度数值不稳，再转回原 dtype
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ---------------------------------------------------------------------
# RoPE 工具（与 step3 相同）
# ---------------------------------------------------------------------
def precompute_freqs_cis(head_dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    fc = freqs_cis[: xq.shape[-2]].view(1, 1, xq.shape[-2], -1)
    xq_out = torch.view_as_real(xq_c * fc).flatten(-2)
    xk_out = torch.view_as_real(xk_c * fc).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------
# 注意力（不变）
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

        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)) \
                    .view(1, 1, cfg.block_size, cfg.block_size)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, freqs_cis):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


# ---------------------------------------------------------------------
# MLP（不变）
# ---------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


# ---------------------------------------------------------------------
# Block —— LayerNorm → RMSNorm
# ---------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)   # ← 改了这里
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)   # ← 改了这里
        self.mlp  = MLP(cfg)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln1(x), freqs_cis)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------
# 顶层 GPT —— 最后一层 LayerNorm → RMSNorm
# ---------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f   = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)     # ← 改了这里
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

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"T={T} > block_size={self.cfg.block_size}"

        x = self.drop(self.tok_emb(idx))
        freqs_cis = self.freqs_cis[:T]
        for blk in self.blocks:
            x = blk(x, freqs_cis)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

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
