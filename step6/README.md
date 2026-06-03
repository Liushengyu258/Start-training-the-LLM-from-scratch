# 第六步：SFT（指令微调）

在 step5 训好的 base 模型上做 **SFT (Supervised Fine-Tuning)**：用 Stanford Alpaca 52K 指令样本，让模型学会“按指令回答”。

## 与预训练的关键差别

| 维度 | 预训练 (step1~5) | SFT (step6) |
|---|---|---|
| 起点 | 随机初始化 | **加载 step5 ckpt** |
| 数据 | 大段语料随机切窗口 | **(instruction, input, output) 三元组** |
| Loss 范围 | 所有 token | **只在 response 上算**（prompt 的 label 置 -100） |
| 学习率 | 3e-4 | **3e-5**（小一个量级，防遗忘） |
| Dropout | 0.1 | 0.0 |
| Weight decay | 0.1 | 0.0 |
| 训练量 | 数千~上万步 | 约 1 个 epoch |

## Alpaca prompt 模板

```
Below is an instruction that describes a task. Write a response ...

### Instruction:
{instruction}

### Input:                  (可选，没有就用另一个模板)
{input}

### Response:
{output}
```

每条样本拼成上面的文本后 BPE 编码，并生成一个 `loss_mask`：
- prompt 段所有 token 对应 mask=0，目标 label 会被置为 -100（PyTorch CE 默认忽略）
- response 段 mask=1，正常算 loss
- response 末尾追加 EOS（GPT-2 BPE 的 id=50256，即 endoftext 特殊 token）
- 长度统一 pad 到 `block_size=512`（pad 部分 mask 也为 0）

## 文件结构

```
step6/
├── data.py        # 下载 Alpaca + 模板拼接 + 编码 + 生成 mask
├── model.py       # 同 step5（结构没变）
├── train.py       # SFT 训练循环；加载 step5 ckpt + 带 mask 的 CE loss
├── generate.py    # 按 Alpaca 模板生成；遇到 EOS 自动停
├── plot.py        # 同 step1~5
└── requirements.txt
```

## 快速开始

```powershell
# 0) 前置：先把 step5 训练完，生成 step5/out/ckpt.pt
D:/Users/A/anaconda3/envs/py310_llm/python.exe step5/train.py

# 1) 准备 SFT 数据（首次会下载 ~25MB Alpaca JSON）
D:/Users/A/anaconda3/envs/py310_llm/python.exe step6/data.py

# 2) SFT 训练（4060Ti 约 5~10 分钟）
D:/Users/A/anaconda3/envs/py310_llm/python.exe step6/train.py

# 3) 推理：按指令生成
D:/Users/A/anaconda3/envs/py310_llm/python.exe step6/generate.py --instruction "Tell me a short story about a brave cat."

# 4) 画曲线
D:/Users/A/anaconda3/envs/py310_llm/python.exe step6/plot.py
```

## 预期与限制

- val loss 应在前几百步快速下降（模型迅速学到 Alpaca 模板）
- **指令理解能力很有限**：12M 参数 + TinyStories 风格 base 远不足以做真正的"小助手"
- 你会看到：
  - 输出**格式**像模像样（按 `### Response:` 开头、有 EOS 结尾）
  - 简单指令偶尔能回答（翻译、续写故事）
  - 复杂推理 / 数学题基本胡说
- 这是预期的——SFT 不能凭空造出能力，只能"激活"base 已学到的东西

## 想要更强效果？

- 把 step5 的预训练扩到更大数据（中文/代码/wiki）和更多步数
- 把 base 扩大到 50M+ 参数（n_embd=384, n_layer=12 之类）
- 换更高质量的指令数据集（如 OpenOrca、UltraChat 子集）
