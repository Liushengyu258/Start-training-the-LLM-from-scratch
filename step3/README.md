# 第三步：RoPE + 更大数据集

在 step2 基础上做两个升级：

| 项目 | step2 | step3 |
|---|---|---|
| 位置编码 | 可学习的绝对位置嵌入 | **RoPE（旋转位置编码）** |
| 数据集 | TinyShakespeare (~1MB) | **Shakespeare + TinyStories (~100MB 默认)** |
| 上下文长度 | 256 | **512** |
| 训练步数 | 5000 | **10000** |
| 模型大小 | ~12M | ~12M（结构不变） |

## RoPE 简介

把 token 嵌入加上"位置向量"的做法换成：**在 attention 内部对 Q、K 的相邻维度做二维旋转**。
位置 m 的第 i 对维度旋转角度 = m·θ_i，θ_i = 10000^(-2i/d_head)。

直观好处：
- **相对位置**：q_m · k_n 只依赖 (m-n)，符合语言"关心相对距离"的本性
- **可外推**：训练 512、推理 1024 仍能用
- **不增参数**：纯 cos/sin 计算
- **现代标配**：LLaMA、Qwen、ChatGLM 等都用它

实现见 `model.py` 里：
- `precompute_freqs_cis()`：预计算每个 (位置, 频率) 的 e^{i·m·θ_i}
- `apply_rotary_emb()`：把它乘到 Q、K 上（用复数乘法表达旋转）
- `CausalSelfAttention.forward()`：在算注意力分数之前调用 `apply_rotary_emb`
- `GPT.__init__`：去掉 `pos_emb`，注册 `freqs_cis` 为 buffer

## 数据集

- **TinyShakespeare**：1MB 莎翁戏剧，沿用 step1/2
- **TinyStories**：HuggingFace `roneneldan/TinyStories` 的官方原始 txt，
  GPT-3.5/GPT-4 生成的简单英文小故事，专为训练小模型设计。

为控制下载耗时，`data.py` 里默认 `MAX_STORIES_BYTES = 100 * 1024 * 1024`（100MB）。
对应 ~2500 万 BPE token，对 12M 模型已经很充足。如果你想用全量（~1.5GB）：

```python
# 改 data.py 顶部
MAX_STORIES_BYTES = None
```

## 快速开始

```powershell
pip install -r step3/requirements.txt

# 1) 准备数据（首次会下载 ~100MB，几分钟）
D:/Users/A/anaconda3/envs/py310_llm/python.exe step3/data.py

# 2) 训练（10000 步，4060Ti 约 15~25 分钟）
D:/Users/A/anaconda3/envs/py310_llm/python.exe step3/train.py

# 3) 生成
D:/Users/A/anaconda3/envs/py310_llm/python.exe step3/generate.py --prompt "Once upon a time" --max_new_tokens 300

# 4) 画曲线
D:/Users/A/anaconda3/envs/py310_llm/python.exe step3/plot.py
```

## 预期现象

- **val loss 不再严重过拟合**（数据多了 100 倍），最终大致在 2.0~2.5 区间（BPE token 空间）
- 生成"once upon a time …"风格的小故事，能保持基本人物 / 情节连贯
- 给 Shakespeare 风格 prompt 也能输出半古英语对白（莎翁数据被混在最后）
