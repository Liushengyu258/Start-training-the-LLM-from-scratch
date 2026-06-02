# 从零训练一个 1M 参数的 LLM

最简化的 GPT 风格 Decoder-only Transformer，在 TinyShakespeare 上做字符级语言建模。

## 项目结构

- `data.py` —— 下载 TinyShakespeare、构建字符级词表、写入 `data/{train,val}.bin`
- `model.py` —— GPT 模型（Causal Self-Attention + MLP + LayerNorm + 权重共享）
- `train.py` —— 训练循环（AdamW + 余弦退火 + 梯度裁剪 + 验证集 ckpt）
- `generate.py` —— 加载 ckpt 采样

## 模型规模（默认）

| 超参 | 值 |
|---|---|
| n_layer | 4 |
| n_head | 4 |
| n_embd | 128 |
| block_size | 128 |
| vocab_size | 65 (字符级) |

参数量约 **0.8M ~ 1.0M**（含位置嵌入；token 嵌入与输出层权重共享）。

## 快速开始

```powershell
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据（约 1MB，自动下载）
python data.py

# 3. 训练（CPU 也能跑，GPU 几分钟即可）
python train.py

# 4. 生成
python generate.py --prompt "ROMEO:" --max_new_tokens 500

# 5. 画训练曲线（读 out/losses.csv，输出 out/loss_curve.png）
python plot.py
```

## 训练日志

训练时每次评估都会把 `iter / train_loss / val_loss / lr` 追加到 `out/losses.csv`，
随时可以用 `python plot.py` 绘制曲线（甚至训练中途也行，CSV 是边训练边 flush 的）。

## 预期效果

- 初始 loss ≈ ln(65) ≈ 4.17
- 训练 5000 步后 val loss 约 1.5 ~ 1.7
- 生成结果：能写出像模像样的"伪莎士比亚"对白（拼写大致正确、有角色名和换行节奏，但语义随机）

## 后续可玩

- 换数据集（中文古诗、代码语料……）
- 升级到 BPE 分词（如 `tiktoken`）
- 加 RoPE / KV-cache / Flash-Attention
- 扩到 10M / 100M 参数
