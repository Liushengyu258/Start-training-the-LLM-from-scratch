# 第二步：BPE 分词 + 10M 模型

在 step1 的基础上做两个升级：

| 项目 | step1 | step2 |
|---|---|---|
| 分词 | 字符级（vocab=65） | **BPE / GPT-2**（vocab=50257） |
| 上下文 | 128 tokens | **256 tokens** |
| 层数 | 5 | **6** |
| 隐藏维度 | 128 | **192** |
| 注意力头 | 4 | **6** |
| 总参数量 | ~1.0M | **~12M**（其中 token 嵌入占 ~9.6M） |

## 为什么用 BPE？

- **序列更短**：1 个 BPE token ≈ 3~4 个字符，相同 `block_size` 看的上下文更多。
- **语义更好**：常见词被打成 1 个 token，模型不必从字母拼起来。
- **直接复用 GPT-2 词表**：`tiktoken` 第一次跑会自动下载并缓存。

## 文件结构

```
step2/
├── data.py        # 改用 tiktoken('gpt2') 编码 TinyShakespeare
├── model.py       # 与 step1 完全一致（GPT 架构本身没变）
├── train.py       # 模型放大到 ~12M
├── generate.py    # 解码改用 enc.decode()
├── plot.py        # 损失曲线（同 step1）
└── requirements.txt
```

## 快速开始

```powershell
pip install -r step2/requirements.txt

# 1) 准备数据：下载 TinyShakespeare + 用 tiktoken 编码
D:/Users/A/anaconda3/envs/py310_llm/python.exe step2/data.py

# 2) 训练（4060 Ti 大约 5~10 分钟）
D:/Users/A/anaconda3/envs/py310_llm/python.exe step2/train.py

# 3) 生成
D:/Users/A/anaconda3/envs/py310_llm/python.exe step2/generate.py --prompt "ROMEO:" --max_new_tokens 200

# 4) 画曲线
D:/Users/A/anaconda3/envs/py310_llm/python.exe step2/plot.py
```

## 关于过拟合

**注意：** TinyShakespeare 编码完只有约 30 万个 BPE token，对 12M 参数的模型来说**严重不足**。
预期现象：

- train loss 会一路降到很低（甚至 < 1）
- val loss 会先降后升，明显过拟合
- 生成结果可能出现“背诵原文”

这正是教学价值所在 —— 真切感受“数据不够 vs 模型够大”的矛盾。
**step3** 会换更大的数据集来解决这个问题。

## 损失值不能直接和 step1 比

字符级和 BPE 的 cross-entropy 是不同空间的：
- step1：每个 token = 1 字符，loss 单位是 “nats / 字符”
- step2：每个 token ≈ 3~4 字符，loss 单位是 “nats / token”

要公平比较，应换算到 “bits per character”（bpc）。直观上看生成质量更靠谱。
