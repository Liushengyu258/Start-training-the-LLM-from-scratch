"""
model.py —— GPT 风格 Decoder-only Transformer 的最小实现
============================================================

整体结构（数据如何在网络里流动）：

    输入 token id (B, T)
        │   B = batch size, T = 序列长度
        ▼
    [Token Embedding] + [Position Embedding]   ← 把整数变成 d 维向量并叠加位置信息
        │
        ▼  shape = (B, T, d_model)
    ┌──────── Transformer Block × N ────────┐
    │  x ← x + CausalSelfAttention(LN(x))  │   ← 残差连接 + 因果自注意力
    │  x ← x + MLP(LN(x))                   │   ← 残差连接 + 前馈网络
    └───────────────────────────────────────┘
        │
        ▼
    LayerNorm
        │
        ▼
    Linear → logits (B, T, vocab_size)         ← 每个位置预测下一个 token 的概率分布
        │
        ▼
    CrossEntropy(targets)                       ← 训练时计算 loss

关键概念速查：
- Decoder-only：只有解码器（像 GPT），不像 BERT 那种 encoder-only。
- Causal（因果）注意力：第 t 个 token 只能看到前 t 个，不能偷看未来 → 用下三角 mask 屏蔽。
- 自注意力 self-attention：每个位置去 “询问” 序列里所有位置（受 mask 限制），
  并加权聚合它们的信息。Q/K/V 三个矩阵就是 Query / Key / Value。
- 残差连接 (residual / skip connection)：y = x + f(x)，让梯度更易传播，深网更稳。
- LayerNorm：对每个 token 的特征向量做归一化，让训练稳定。
- 权重共享 (weight tying)：输出层的权重和 token embedding 共享同一个矩阵，
  既省参数又有理论依据（输入和输出都是同一个词表空间）。
"""
from dataclasses import dataclass    # 用来简洁地定义“配置类”
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# 配置：把所有超参打包成一个对象，方便传递和保存到 ckpt
# ---------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 65      # 词表大小（字符级=65）
    block_size: int = 128     # 上下文窗口长度（一次最多看多少个 token）
    n_layer:    int = 5       # Transformer Block 堆几层
    n_head:     int = 4       # 多头注意力的头数
    n_embd:     int = 128     # 隐藏向量维度（也就是 d_model）
    dropout:    float = 0.1   # 随机丢弃比例，防过拟合


# ---------------------------------------------------------------------
# 因果自注意力（Causal Self-Attention）
# ---------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """
    多头自注意力 + 因果 mask。
    数学公式： Attention(Q,K,V) = softmax( Q·K^T / sqrt(d_k) + mask ) · V
    “多头”就是把 d_model 维分成 n_head 份，每份独立做注意力，最后拼回来。
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # d_model 必须能被 n_head 整除，因为要平均切分
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head   = cfg.n_head
        self.n_embd   = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head     # 每个头的维度

        # 一次性算 Q、K、V：用一个大 Linear 输出 3*n_embd，再切三块。比分三次 Linear 更快。
        self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        # 注意力输出之后再过一层投影，融合多个头
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        # 两个 dropout：注意力权重上 & 残差路径上
        self.attn_drop  = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # 因果 mask：下三角全 1，上三角全 0。形状 (1,1,T,T) 便于广播到 (B,H,T,T)
        # tril = lower triangle，保留对角线及以下
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)) \
                    .view(1, 1, cfg.block_size, cfg.block_size)
        # register_buffer：让这个张量随模型 .to(device) 一起转到 GPU，但不算可训练参数
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        # x.shape = (B, T, C)   B=batch, T=序列长, C=n_embd
        B, T, C = x.shape

        # 1) 一次算出 q/k/v，然后沿最后一维切成三块
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        # 2) reshape 成多头形状 (B, n_head, T, head_dim)
        #    这样不同头之间的注意力是相互独立的
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3) 注意力分数 = Q·K^T / sqrt(d_k)，shape (B, H, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 4) 因果 mask：把上三角位置（未来）填成 -inf，softmax 后就变成 0 → 看不到未来
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        # 5) softmax 沿“被关注的位置”维归一化，得到注意力权重
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # 6) 用注意力权重加权 V，得到聚合后的表示，shape (B, H, T, head_dim)
        y = att @ v
        # 7) 把多头拼回去 (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 8) 输出投影 + dropout
        return self.resid_drop(self.proj(y))


# ---------------------------------------------------------------------
# 前馈网络（FeedForward / MLP）：每个位置上独立做的 2 层 MLP
# ---------------------------------------------------------------------
class MLP(nn.Module):
    """
    Transformer 里的 “Position-wise FFN”：
        Linear(d → 4d) → GELU → Linear(4d → d) → Dropout
    维度先扩大 4 倍再缩回，是 GPT/BERT 的标准做法（参数量大头之一）。
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc   = nn.Linear(cfg.n_embd,     4 * cfg.n_embd)   # 升维
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)        # 降维
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # GELU：Gaussian Error Linear Unit，比 ReLU 更平滑，GPT 系列默认激活函数
        return self.drop(self.proj(F.gelu(self.fc(x))))


# ---------------------------------------------------------------------
# 一个 Transformer Block：Attention + MLP，各带 LayerNorm 和残差
# ---------------------------------------------------------------------
class Block(nn.Module):
    """
    现代 GPT 用的是 “Pre-Norm”：先 LayerNorm 再进入子层。
    公式：
        x = x + attn( LN(x) )
        x = x + mlp ( LN(x) )
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.n_embd)
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # 残差 + 注意力
        x = x + self.mlp (self.ln2(x))   # 残差 + 前馈
        return x


# ---------------------------------------------------------------------
# 顶层 GPT 模型
# ---------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        # 1) Token 嵌入：把 id (整数) 映射为 n_embd 维向量。形状 (vocab, d)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # 2) 位置嵌入：第 0、1、2、… 位置各对应一个可学习向量。形状 (block_size, d)
        #    Transformer 自身没有位置感，需要这个告诉它“谁在前谁在后”。
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)

        # 3) N 个 Transformer Block 串起来
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])

        # 4) 最后一层 LayerNorm（GPT 论文里有的“final_layer_norm”）
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # 5) 语言模型头：把 d 维特征投影到 vocab_size 维，得到每个 token 的 logit
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # 权重共享 (weight tying)：让 head 的权重就是 tok_emb 的权重
        # 直观理解：嵌入是“词→向量”，head 是“向量→词”，本来就是一对反向操作。
        # 好处：少一份参数（vocab*d 个），通常还能小幅提升效果。
        self.head.weight = self.tok_emb.weight

        # 用统一的初始化方法（小权重的正态分布）初始化全部子模块
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """按照 GPT-2 论文做小标准差正态分布初始化。"""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """返回总参数量（含权重共享时不会重复算）。"""
        return sum(p.numel() for p in self.parameters())

    # -----------------------------------------------------------------
    # 前向：训练时传 targets 计算 loss；推理时不传，只要 logits
    # -----------------------------------------------------------------
    def forward(self, idx, targets=None):
        # idx: (B, T)  里面是 token id
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"序列长度 {T} 超过 block_size {self.cfg.block_size}"

        # 位置序号 [0,1,2,...,T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # token 向量 + 位置向量 → (B, T, d_model)
        # pos_emb(pos) 形状是 (T, d)，加 None 变成 (1, T, d) 才能广播到每个 batch
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        # 串行通过所有 Transformer Block
        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.head(x)              # (B, T, vocab_size)

        # 训练时计算交叉熵：每个位置预测的是“下一个 token”
        # logits.view(-1, V) 把 (B,T,V) 拉平成 (B*T, V)，targets 同样拉平
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    # -----------------------------------------------------------------
    # 文本生成：逐 token 自回归采样
    # -----------------------------------------------------------------
    @torch.no_grad()                         # 推理不需要算梯度，关掉省显存
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        idx: (B, T0)  起始 prompt 的 token id
        每一步：
          1) 取最近 block_size 个 token 作为输入
          2) 前向得到最后一个位置的 logits
          3) 用 temperature 平滑 / 锐化分布
          4) 可选 top_k：只保留概率最高的 k 个，其它置 -inf
          5) softmax 得到概率，按概率采样下一个 token
          6) 拼到序列尾部，重复
        """
        for _ in range(max_new_tokens):
            # 上下文最多 block_size 这么长
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # 只看最后一个位置
            if top_k is not None:
                # 找 top-k 的阈值，把比它低的全部置 -inf（softmax 后概率为 0）
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)   # 按概率随机抽 1 个
            idx = torch.cat([idx, nxt], dim=1)
        return idx
