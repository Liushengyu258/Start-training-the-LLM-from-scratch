"""
model.py —— GPT + RoPE + RMSNorm + SwiGLU
============================================================
相对 step4 的唯一改动：把 MLP 由
        Linear(d, 4d) → GELU → Linear(4d, d)
替换成 **SwiGLU**（LLaMA / Qwen / Mistral 同款）：
        x1 = W_gate(x)
        x2 = W_up  (x)
        y  = W_down( SiLU(x1) * x2 )
其中 SiLU(x) = x · sigmoid(x)，又名 Swish。

直观理解：
    - 多出一个 “gate” 分支：让网络学习 “这条通路要不要打开”
    - 比 ReLU/GELU 表达力更强，长期实验中收敛更快、效果略好
    - 是“门控线性单元 GLU 家族”里目前最常用的一员

参数对齐：
    标准 MLP 有 2 个 (d, 4d) 大矩阵 → 8·d² 个参数。
    SwiGLU 有 3 个 (d, h) 矩阵 → 3·d·h 个参数。
    为保持参数量相当，常取 h ≈ (2/3)·4d ≈ 2.67 d，
    通常向上取整到 256/64 的倍数。这里 d=192，
    (2/3)*4*192 = 512 正好整齐，于是 h = 512。
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
    # FFN 隐藏维度。若设为 None 会按 (2/3)*4*n_embd 自动算并向上取整到 multiple_of。
    ffn_hidden:    int | None = None
    multiple_of:   int   = 64
    dropout:       float = 0.1
    rope_theta:    float = 10000.0
    norm_eps:      float = 1e-5


# ---------------------------------------------------------------------
# RMSNorm（同 step4）
# ---------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


# ---------------------------------------------------------------------
# RoPE 工具（同 step3/4）
# ---------------------------------------------------------------------
def precompute_freqs_cis(head_dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    fc = freqs_cis[: xq.shape[-2]].view(1, 1, xq.shape[-2], -1)
    xq_out = torch.view_as_real(xq_c * fc).flatten(-2)
    xk_out = torch.view_as_real(xk_c * fc).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------
# 注意力（同 step3/4）
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
# SwiGLU FFN（核心改动）
# ---------------------------------------------------------------------
class SwiGLU(nn.Module):
    """
    LLaMA 风格的门控 FFN：
        out = W_down( SiLU(W_gate(x)) * W_up(x) )
    无 bias，参数效率高。
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.n_embd

        # 计算隐藏维度：默认 (2/3)*4d，向上对齐到 multiple_of
        if cfg.ffn_hidden is not None:
            hidden = cfg.ffn_hidden
        else:
            hidden = int(2 * 4 * d / 3)
            m = cfg.multiple_of
            hidden = ((hidden + m - 1) // m) * m

        # 三个 Linear（注意均不带 bias，符合 LLaMA 习惯）
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up   = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
        self.drop   = nn.Dropout(cfg.dropout)
        self.hidden = hidden

    def forward(self, x):
        # SiLU(x) = x * sigmoid(x)，PyTorch 里就是 F.silu
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ---------------------------------------------------------------------
# Block：MLP → SwiGLU
# ---------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.mlp  = SwiGLU(cfg)                              # ← 改了这里

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln1(x), freqs_cis)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------
# 顶层 GPT（其它无变化）
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
