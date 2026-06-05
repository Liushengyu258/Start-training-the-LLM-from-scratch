# 第九步：100M 级模型 + 大数据集 + Flash-Attention

把 step5 的 12M 模型**扩大 10 倍**到 ~120M 参数（GPT-2 Small 级别），配合 20 倍的训练数据，并引入 Flash-Attention 和梯度累积等工程优化。

## 与 step5 的关键对比

| 维度 | step5 (12M) | step9 (120M) |
|------|-------------|-------------|
| n_embd | 192 | **768** |
| n_layer | 6 | **12** |
| n_head | 6 | **12** |
| head_dim | 32 | **64** |
| SwiGLU hidden | 512 | **2048** |
| block_size | 512 | **1024** |
| dropout | 0.1 | **0.0** |
| 参数量 | ~12M | **~120M** |
| 训练数据 | ~25M tokens | **~500M tokens** |
| Attention | 手写 O(T²) 显存 | **SDPA (Flash-Attention)** |
| 梯度累积 | 无 | **4 步累积** |
| 断点续训 | 无 | **有** |

## Flash-Attention (PyTorch SDPA)

使用 PyTorch 2.0+ 内置的 `F.scaled_dot_product_attention`：
- **零额外安装**：不需要 `flash-attn` 第三方库
- **Windows 完美兼容**
- **自动选择最优后端**：Flash-2 / Memory-Efficient / Math
- **显存节省**：从 O(T²) 降到 O(T)，1024 上下文节省 ~40% 显存

```python
# 旧方式（step5）：手写 attention，需要预存 mask
att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
att = F.softmax(att, dim=-1)
y = att @ v

# 新方式（step9）：一行搞定，自动 Flash
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

## 数据集

| 数据源 | 大小 | 内容 | BPE tokens |
|--------|------|------|-----------|
| TinyStories 全量 | ~1.5GB | GPT-3.5/4 生成的简单英文故事 | ~400M |
| WikiText-103-raw | ~500MB | 维基百科长文章 | ~100M+ |
| **合计** | **~2GB** | | **~500M+** |

## 梯度累积

16GB 显存无法一次跑大 batch，通过梯度累积等效放大：

```
有效 batch = micro_batch_size × grad_accumulation_steps
           = 8 × 4 = 32 序列/步
           = 32 × 1024 = 32,768 tokens/步
```

## 文件结构

```
step9/
├── data.py          # 下载 TinyStories 全量 + WikiText-103 + BPE 编码
├── model.py         # 120M 模型 + Flash-Attention (SDPA)
├── train.py         # 梯度累积 + AMP + 断点续训 + tokens/sec 监控
├── generate.py      # 文本续写（同 step5 接口）
├── plot.py          # 训练曲线
└── requirements.txt
```

## 快速开始

```powershell
pip install -r step9/requirements.txt

# 1) 准备数据（首次下载 ~2GB，编码约 5~10 分钟）
D:/Users/A/anaconda3/envs/py310_llm/python.exe step9/data.py

# 2) 训练（50K 步，4060Ti 16GB 约 12~20 小时）
D:/Users/A/anaconda3/envs/py310_llm/python.exe step9/train.py

# 3) 中途查看曲线（训练不用停）
D:/Users/A/anaconda3/envs/py310_llm/python.exe step9/plot.py

# 4) 生成
D:/Users/A/anaconda3/envs/py310_llm/python.exe step9/generate.py --prompt "Once upon a time" --max_new_tokens 300
D:/Users/A/anaconda3/envs/py310_llm/python.exe step9/generate.py --prompt "The history of artificial intelligence" --max_new_tokens 200
```

## 断点续训

训练自动保存 checkpoint 到 `out/ckpt.pt`（含 model + optimizer + iter）。

如果训练中断（Ctrl+C / 断电 / 系统重启），只需重新运行：
```powershell
D:/Users/A/anaconda3/envs/py310_llm/python.exe step9/train.py
```
会自动检测并恢复，从上次保存的位置继续训练。

## 显存与速度估算（4060 Ti 16GB）

| 项目 | 估算 |
|------|------|
| 模型 (bf16) | ~250MB |
| 优化器 (fp32 m+v) | ~1GB |
| 梯度 | ~250MB |
| 激活值 (Flash-Attn) | ~2~4GB |
| **总显存** | **~4~6GB** |
| 每步耗时 | ~0.5~1s |
| tokens/sec | ~30K~60K |
| 50K 步总时间 | **~12~20h** |

## 预期效果

- **val loss**：预计从 ~10（初始）降到 **1.5~2.0** 区间
- **生成质量**：
  - 故事结构完整、逻辑连贯
  - 人物名字/情节保持一致
  - 能写较长的段落（1024 上下文）
  - 维基百科风格的事实性文本（质量有限但格式到位）
- **对比 step5**：同样 prompt 下，文本流畅度和连贯性应有明显提升

## 超参调节建议

| 参数 | 调大效果 | 调小效果 |
|------|----------|----------|
| MICRO_BATCH | 训练更快，需更多显存 | 省显存 |
| GRAD_ACCUM | 有效 batch 更大，训练更稳 | 更新更频繁 |
| LR | 学得快，但可能不稳定 | 更稳，但慢 |
| MAX_ITERS | 训练更充分 | 节省时间 |
| BLOCK_SIZE | 更长上下文，更多显存 | 省显存，但短视 |

## 后续可做

- **SFT / DPO**：和 step6/7 一样，在 120M base 上做微调对齐
- **中文支持**：换用支持中文的 tokenizer（如 Qwen 的 BPE）
- **多 GPU (DDP)**：多卡并行加速训练
- **更长上下文**：RoPE 外推 + 滑动窗口注意力
