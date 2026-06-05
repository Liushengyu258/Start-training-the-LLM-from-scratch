"""
model.py —— GPT + RoPE + RMSNorm + SwiGLU + Flash-Attention (SDPA)
============================================================
相对 step5 的核心改动：

1. **Flash-Attention**（通过 PyTorch 2.0+ 内置的 scaled_dot_product_attention）：
   - 自动选择最优后端（Flash / Memory-Efficient / Math）
   - 不再需要预计算因果 mask buffer
   - 显存从 O(T²) 降到 O(T)，训练长序列时节省巨大

2. **模型规模提升到 ~120M 参数**（GPT-2 Small 级别）：
   - n_embd: 192 → 768
   - n_layer: 6 → 12
   - n_head: 6 → 12
   - block_size: 512 → 1024

3. **初始化改进**：
   - 残差路径输出投影 std 按 1/sqrt(2*n_layer) 缩放，防深网梯度不稳

4. **dropout=0.0**：
   - 数据量充足（500M+ tokens >> 120M params），不再需要正则化

其它保留不变：RoPE + RMSNorm + SwiGLU + 权重共享。
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
    vocab_size:    int   = 50257      # GPT-2 BPE 词表
    block_size:    int   = 1024       # 上下文窗口（比 step5 的 512 翻倍）
    n_layer:       int   = 12         # Transformer 层数
    n_head:        int   = 12         # 注意力头数
    n_embd:        int   = 768        # 隐藏维度
    # SwiGLU 隐藏维度。None 时自动按 (2/3)*4*n_embd 对齐到 multiple_of。
    ffn_hidden:    int | None = None
    multiple_of:   int   = 64
    dropout:       float = 0.0        # 大数据训练不需要 dropout
    rope_theta:    float = 10000.0
    norm_eps:      float = 1e-5


# ---------------------------------------------------------------------
# RMSNorm（同 step4/5）
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
# RoPE 工具（同 step3/4/5）
# ---------------------------------------------------------------------
def precompute_freqs_cis(head_dim: int, end: int, theta: float = 10000.0):
    """预计算 RoPE 的复数旋转因子。"""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """对 Q、K 施加旋转位置编码。"""
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    fc = freqs_cis[: xq.shape[-2]].view(1, 1, xq.shape[-2], -1)
    xq_out = torch.view_as_real(xq_c * fc).flatten(-2)
    xk_out = torch.view_as_real(xk_c * fc).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------
# 因果自注意力 + Flash-Attention (SDPA)
# ---------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """
    多头自注意力，使用 PyTorch 2.0+ 的 scaled_dot_product_attention（SDPA）。
    SDPA 会自动选择最优后端：
      - FlashAttention-2（Ampere/Ada 以上 GPU）
      - Memory-Efficient Attention（xformers 风格）
      - Math（兜底）
    好处：显存 O(T) 而非 O(T²)，速度快 2~4x，且无需手写 mask。
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head   = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_embd   = cfg.n_embd
        self.dropout  = cfg.dropout

        self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 施加 RoPE（只对 Q、K，不对 V）
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Flash-Attention via SDPA：is_causal=True 自动处理因果 mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


# ---------------------------------------------------------------------
# SwiGLU FFN（同 step5）
# ---------------------------------------------------------------------
class SwiGLU(nn.Module):
    """
    LLaMA 风格的门控 FFN：
        out = W_down( SiLU(W_gate(x)) * W_up(x) )
    """
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
# Transformer Block
# ---------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.mlp  = SwiGLU(cfg)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln1(x), freqs_cis)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------
# 顶层 GPT 模型
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

        # 权重共享
        self.head.weight = self.tok_emb.weight

        # 预计算 RoPE 旋转因子
        head_dim  = cfg.n_embd // cfg.n_head
        freqs_cis = precompute_freqs_cis(head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # 初始化
        self.apply(self._init_weights)
        # GPT-2 做法：残差路径输出投影按 1/sqrt(2*n_layer) 缩放，防深网梯度发散
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight") or pn.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """返回总参数量（权重共享不重复计算）。"""
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
        """自回归采样（朴素版，用于短生成）。"""
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
