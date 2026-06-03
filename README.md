# 从零训练 LLM —— 学习路线

本仓库按“分步进阶”的方式记录从零训练大语言模型的过程。每一步都是一个独立的子文件夹，可单独运行，循序加深。

## 目录

- [`step1/`](./step1/) —— **第一步：最小可用的 GPT**
  - 字符级分词 + 5 层 Transformer + ~1M 参数
  - TinyShakespeare 数据集
  - 训练 / 生成 / 损失曲线
  - 详见 [`step1/README.md`](./step1/README.md)

- [`step2/`](./step2/) —— **第二步：BPE 分词 + 10M 模型**
  - `tiktoken` GPT-2 BPE（vocab=50257）
  - 6 层 Transformer，n_embd=192，约 12M 参数
  - 同样在 TinyShakespeare 上训练（数据偏小，会过拟合，下一步解决）
  - 详见 [`step2/README.md`](./step2/README.md)

- [`step3/`](./step3/) —— **第三步：RoPE 位置编码 + 更大数据集**
  - 旋转位置编码（LLaMA/Qwen 同款）替代可学习位置嵌入
  - Shakespeare + TinyStories（默认 ~100MB，可调）
  - 上下文 512、训练 10000 步
  - 详见 [`step3/README.md`](./step3/README.md)

- [`step4/`](./step4/) —— **第四步：RMSNorm**
  - 把 LayerNorm 替换为 LLaMA / Qwen 同款的 RMSNorm
  - 详见 [`step4/README.md`](./step4/README.md)

- [`step5/`](./step5/) —— **第五步：SwiGLU 激活**
  - 把标准 GELU MLP 替换为门控的 SwiGLU（LLaMA / Qwen / Mistral 同款）
  - 详见 [`step5/README.md`](./step5/README.md)

- [`step6/`](./step6/) —— **第六步：SFT 指令微调**
  - 加载 step5 的 base 权重，在 Stanford Alpaca 上做监督指令微调
  - prompt 模板 + loss 仅在 response 上计算
  - 详见 [`step6/README.md`](./step6/README.md)

- [`step7/`](./step7/) —— **第七步：DPO 偏好对齐**
  - 基于 step6 SFT 模型，在 Anthropic HH-RLHF 偏好对上做 DPO
  - 同时持有 policy（可训练）+ reference（冻结）两个模型
  - 跟踪 loss / reward margin / accuracy 三个指标
  - 详见 [`step7/README.md`](./step7/README.md)

- [`step8/`](./step8/) —— **第八步：推理 KV-cache 加速**
  - 纯推理优化，权重 100% 兼容 step5/6/7 的 ckpt
  - attention 加 `past_kv` 支持；`generate_cached` 把 O(N²) 降到 O(N)
  - `benchmark.py` 对比朴素 vs 缓存的速度，校验数学等价
  - 详见 [`step8/README.md`](./step8/README.md)

## 后续计划（待完善）

- **step9**：扩到 50M+ 大模型 / 中文支持 / Flash-Attention

## 一键按顺序训练

根目录提供了 `run_all.py`：

```powershell
# 默认跑 step1 ~ step7
D:/Users/A/anaconda3/envs/py310_llm/python.exe run_all.py

# 只跑 step3 ~ step7（接力做现代 GPT + SFT + DPO）
D:/Users/A/anaconda3/envs/py310_llm/python.exe run_all.py --steps 3-7

# 任意子集
D:/Users/A/anaconda3/envs/py310_llm/python.exe run_all.py --steps 1,3,5
D:/Users/A/anaconda3/envs/py310_llm/python.exe run_all.py --steps 5

# 数据已经下载好，跳过 data.py 节省时间
D:/Users/A/anaconda3/envs/py310_llm/python.exe run_all.py --steps 4-7 --skip-data

# 只打印将要执行的命令，不真正运行
D:/Users/A/anaconda3/envs/py310_llm/python.exe run_all.py --steps 1-7 --dry-run
```

> 注意依赖关系：`step6` 需要 `step5/out/ckpt.pt`，`step7` 需要 `step6/out/ckpt.pt`。
> 单独跑后面的步骤前请确保前序产物已生成（或直接从 step1 / step5 开始跑）。

## 环境

推荐 conda 环境：Python 3.10 + PyTorch (CUDA)。每一步的子文件夹里都带有自己的 `requirements.txt`。
