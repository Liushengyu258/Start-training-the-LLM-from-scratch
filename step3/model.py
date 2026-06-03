"""
model.py —— GPT + RoPE
============================================================
相对 step1/step2 的关键改动：把可学习的【绝对位置嵌入】替换成
【RoPE = Rotary Position Embedding】（旋转位置编码，LLaMA / Qwen / 多数现代 LLM 都在用）。

什么是 RoPE？
    思路：不再单独学一个位置向量加到 token 嵌入上，而是
    在 attention 内部，对 Q 和 K 的每一对相邻维度做一次“二维旋转”。
    每个位置 m 用一个频率 θ_i 旋转角度 m·θ_i：
        x' = [ cos(mθ)·x_even - sin(mθ)·x_odd ,
               sin(mθ)·x_even + cos(mθ)·x_odd ]
    这样得到的 q_m·k_n 只与相对位置 (m-n) 有关 —— 即相对位置编码。

为什么改？
    1) 相对位置：天然适配语言建模“关心相对距离”的需求。
    2) 可外推：训练 256 长度，推理时可用 512 也大致可用（无需重新学位置）。
    3) 不增加参数：完全是确定性的 cos/sin。
    4) 现代 LLM 标配。

整体数据流（其余照旧）：
    输入 id (B,T)
       │
    Token Embedding              ← 仍保留
       │
    Dropout
       │
    Transformer Block × N
       │   每个 block 里：
       │     LN -> Attention[ apply_rope(Q,K) ] -> 残差
       │     LN -> MLP                            -> 残差
       │
    LN_f -> Linear(head, tied) -> logits
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
    block_size: int   = 512        # RoPE 支持更长，提到 512
    n_layer:    int   = 6
    n_head:     int   = 6
    n_embd:     int   = 192
    dropout:    float = 0.1
    rope_theta: float = 10000.0    # RoPE 的基础周期 θ


# ---------------------------------------------------------------------
# RoPE 工具函数
# ---------------------------------------------------------------------
def precompute_freqs_cis(head_dim: int, end: int, theta: float = 10000.0):
    """
    预计算所有 (位置, 频率) 对应的复数 e^{i·m·θ_i} = cos + i·sin。
    返回 shape = (end, head_dim/2) 的 complex 张量。

    解释：
        head_dim 维特征被两两配对，共 head_dim/2 对。每对对应一个频率 θ_i：
            θ_i = theta ** (-2i / head_dim)        i=0,1,...,head_dim/2 - 1
        位置 m 处第 i 对的旋转角度 = m · θ_i。
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()          # (end, head_dim/2)
    # 用复数形式存：模长 1，幅角 = m·θ_i
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    对 Q、K 应用旋转。
    xq, xk: (B, H, T, head_dim)
    freqs_cis: (T, head_dim/2)   复数
    """
    # 把每两个相邻实数看作 (实部, 虚部)，转成复数：
    #   shape (B, H, T, head_dim) -> (B, H, T, head_dim/2) complex
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 广播 freqs_cis: (T, d/2) -> (1, 1, T, d/2)
    fc = freqs_cis[: xq.shape[-2]].view(1, 1, xq.shape[-2], -1)
    # 复数相乘 = 旋转，再把复数拆回实数并 flatten 最后两维
    xq_out = torch.view_as_real(xq_c * fc).flatten(-2)
    xk_out = torch.view_as_real(xk_c * fc).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------
# 注意力（在 Q/K 上做 RoPE）
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

        # 因果 mask（下三角）
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)) \
                    .view(1, 1, cfg.block_size, cfg.block_size)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, freqs_cis):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        # 切多头：(B, T, C) -> (B, H, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # >>> 关键：在 Q、K 上应用 RoPE <<<
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # 经典缩放点积注意力
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v                                     # (B, H, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


# ---------------------------------------------------------------------
# MLP（无变化）
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
# Block：注意 forward 多了 freqs_cis 参数
# ---------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.n_embd)
        self.mlp  = MLP(cfg)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln1(x), freqs_cis)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------
# 顶层 GPT
# ---------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # 没有 pos_emb 了！位置信息由 RoPE 提供
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f   = nn.LayerNorm(cfg.n_embd)
        self.head   = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight     # 权重共享

        # RoPE 频率缓存（不可训练）
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

        x = self.drop(self.tok_emb(idx))           # (B, T, C)
        freqs_cis = self.freqs_cis[:T]              # 只取前 T 个位置的旋转
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
