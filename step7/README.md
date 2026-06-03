# 第七步：DPO（Direct Preference Optimization）

在 step6 的 SFT 模型上做**偏好对齐**。用人类标注的"chosen vs rejected"对子让模型学会"更喜欢哪种回答"。

## 为什么 DPO 而不是 RLHF/PPO？

| | RLHF (PPO) | DPO |
|---|---|---|
| 需要奖励模型 | ✅ 需要单独训一个 | ❌ 直接用偏好对 |
| 需要 RL | ✅ PPO 复杂、不稳定 | ❌ 普通监督训练 |
| 实现难度 | 高 | 低 |
| 训练稳定性 | 较差 | 较好 |
| 效果 | 公认优秀 | 多数场景与 PPO 持平/更好 |

DPO 由 Stanford 2023 年提出，现在已是开源社区的主流偏好对齐方法（Llama-2/3-chat、Mistral、Qwen 等的开源复现版都在用）。

## DPO 损失推导（简版）

每条样本：`(prompt x, chosen y_w, rejected y_l)`。

```
logits = β · ( (logπ_θ(y_w|x) - logπ_ref(y_w|x))
              -(logπ_θ(y_l|x) - logπ_ref(y_l|x)) )

L      = -log σ(logits)
```

- `π_θ` = policy（可训练）
- `π_ref` = reference（冻结，就是 SFT 模型）
- `β`（beta）= 温度系数，控制 policy 偏离 reference 的程度，0.1 ~ 0.5 之间常见
- σ = sigmoid

**直觉**：把 chosen 的"相对优势"减去 rejected 的"相对优势"，希望差越大越好 → sigmoid 越接近 1 → loss 越接近 0。

## 数据：Anthropic HH-RLHF

- 来自 `Anthropic/hh-rlhf` 的 `helpful-base/train.jsonl.gz`
- 每条 = `{chosen, rejected}` 两段完整对话，只在最后回答上不同
- 我们用"最长公共前缀"切出共享 prompt，分别得到 chosen / rejected response
- 默认取前 **20000** 条（可在 `data.py` 改 `MAX_SAMPLES`）
- BPE 编码 + 生成 mask（只在 response 段算 logprob）+ pad 到 block_size=512

## 文件结构

```
step7/
├── data.py        # 下载 HH-RLHF + 切 prompt + 双路编码（chosen / rejected）
├── model.py       # 同 step5/6（架构不变）
├── train.py       # DPO 训练；同时持有 policy + reference 两个模型
├── generate.py    # 用 HH 对话模板 "Human: ... Assistant:" 推理
├── plot.py        # 比 step1~6 多了 reward_margin / acc 两条曲线
└── requirements.txt
```

## 训练时同时跟踪的指标

- **loss** ↓：DPO 损失，越小越好
- **reward margin** ↑：`β·(logπ_θ - logπ_ref)` 在 chosen vs rejected 上的差值；应从 0 起步逐渐上升
- **accuracy** ↑：chosen 比 rejected 得到更高 reward 的比例；从 ~50% 起步，逐步上升到 60~70%+

## 快速开始

```powershell
# 0) 前置：必须已有 step6/out/ckpt.pt （SFT 后的模型）
D:/Users/A/anaconda3/envs/py310_llm/python.exe step6/train.py

# 1) 准备 HH-RLHF（首次下载 ~50MB gzip）
D:/Users/A/anaconda3/envs/py310_llm/python.exe step7/data.py

# 2) DPO 训练（双模型 + 4 次 forward，4060Ti 约 15~25 分钟）
D:/Users/A/anaconda3/envs/py310_llm/python.exe step7/train.py

# 3) 推理
D:/Users/A/anaconda3/envs/py310_llm/python.exe step7/generate.py --message "What is the best way to learn Python?"

# 4) 曲线
D:/Users/A/anaconda3/envs/py310_llm/python.exe step7/plot.py
```

## 显存与速度

- policy + reference 两个模型同时在 GPU：~250MB 显存
- 每步 4 次 forward + 1 次 backward → 比 SFT 慢约 4 倍
- `BATCH_SIZE=4` 默认值是为 8GB 显存设计；24GB 卡可调到 16

## 预期现象

- **loss** 从 ≈0.693（=−ln(0.5)，相当于随机猜）逐步下降到 0.5 以下
- **accuracy** 从 ~50% 上升到 60~70%
- **reward margin** 从 0 起步逐渐为正
- 生成结果：**风格**会向"helpful、礼貌、长篇"偏移；**事实正确性**仍受 12M 模型容量限制

## 重要超参经验

- **LR 一定要小**：1e-6 量级。大了会把 policy 拉跑偏，loss 反而上升
- **β 太大**容易让 policy 过度偏离 reference（模型胡说）；太小则学不动
- **训练步数不要太多**：DPO 容易过拟合到偏好对，过训会让生成质量变差

## 下一步可以做

- **Online DPO / IPO / KTO**：DPO 的各种改进版
- **RLHF / PPO**：经典路线，需要先训奖励模型
- **多轮 SFT + DPO 迭代**：现代开源对齐套路
