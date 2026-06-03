# 第五步：SwiGLU 激活

在 step4 基础上做一个改动：把 MLP（Linear→GELU→Linear）换成 **SwiGLU**（LLaMA / Qwen / Mistral 同款）。

| 项目 | step4 | step5 |
|---|---|---|
| FFN | Linear(d→4d)→GELU→Linear(4d→d) | **SwiGLU**：W_down( SiLU(W_gate(x)) * W_up(x) ) |
| 其它 | RoPE + RMSNorm | 同左 |

## SwiGLU 是什么

公式：
```
gate = W_gate(x)
up   = W_up  (x)
y    = W_down( SiLU(gate) * up )
```

其中 `SiLU(x) = x · sigmoid(x)`（又叫 Swish）。

为什么好：
- **门控机制**：`SiLU(gate)` 起到“阀门”作用，控制 `up` 的每一维通过多少
- **比 GELU/ReLU 表达力更强**：多一条独立的线性通路
- **LLaMA 论文实证更优**：在同参数量下 loss 更低
- **现已成事实标准**：LLaMA / Qwen / Mistral / Gemma 等清一色用 SwiGLU

## 参数对齐

标准 MLP：2 个 `(d, 4d)` 矩阵 → **8·d²** 参数
SwiGLU ：3 个 `(d, h)` 矩阵 → **3·d·h** 参数

要保持参数量相当：`h ≈ (2/3) · 4d ≈ 2.67d`，通常向上对齐到 64/256 的倍数。
本项目 `d=192`：自动算得 `hidden = 512`（正好整齐，对齐到 64 倍数）。

每个 block 的 FFN 参数：
- step4：2 × 192 × 768 = **294,912**
- step5：3 × 192 × 512 = **294,912** ← 完全相等

## 代码改动

只动了 `model.py`：
1. 新增 `class SwiGLU`（约 15 行）
2. 配置加 `ffn_hidden` 和 `multiple_of`
3. `Block.mlp = SwiGLU(cfg)` 替换原来的 `MLP(cfg)`

其它文件全部沿用 step4。

## 快速开始

```powershell
pip install -r step5/requirements.txt

# 数据可直接复用 step3/4 已下载的
Copy-Item -Recurse step3\data step5\data

D:/Users/A/anaconda3/envs/py310_llm/python.exe step5/train.py
D:/Users/A/anaconda3/envs/py310_llm/python.exe step5/generate.py --prompt "Once upon a time" --max_new_tokens 300
D:/Users/A/anaconda3/envs/py310_llm/python.exe step5/plot.py
```

## 预期现象

- 总参数量与 step4 几乎完全一致（~12.31M）
- val loss 通常比 step4 略低（同步数下 SwiGLU 更高效）
- 训练耗时略增（多了一个 Linear 分支，但 PyTorch 能并行化大半）
