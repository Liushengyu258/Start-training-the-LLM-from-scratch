# 第八步：推理 KV-cache 加速

**纯推理优化**，不训练新模型。直接加载 step5 / step6 / step7 任一个 ckpt（权重完全兼容）。

## 为什么需要 KV-cache？

自回归生成一段 N 个 token 的文本：
- **朴素方法**：每生成 1 个新 token 都把整个序列重头算一遍
  - 总计算量约 O(N²)
  - 浪费：第 t 步算的 K、V，第 t+1 步还要再算一次同样的东西
- **KV-cache 方法**：把每层 attention 的 K、V 缓存起来
  - 每一步只算 1 个新 token 的 Q、K、V
  - K、V 拼到缓存末尾
  - 总计算量 O(N)
  - 代价：每层多存 `(B, n_head, T, head_dim)` 的 K 和 V

## 代码改动（vs step5/6/7 model.py）

| 模块 | 改动 |
|---|---|
| `CausalSelfAttention.forward` | 多两个参数 `past_kv`, `use_cache`；返回 `(out, new_kv)` |
| 因果 mask | 仅在 prefill（无 cache）时使用；增量解码时省去 |
| `Block.forward` | 把 cache 透传 |
| `GPT.forward` | 多参数 `past_kvs`, `use_cache`；按 `start_pos = past_kv 长度` 取对应 RoPE |
| **新增** `GPT.generate_cached` | 先 prefill 整段 prompt 建立 cache，再逐 token 增量解码 |
| 保留 `GPT.generate` | 朴素方式，用于对照与基准测试 |

**架构层面不动一字**，参数完全相同 → step5/6/7 的 ckpt 直接加载就能用。

## 文件结构

```
step8/
├── model.py        # KV-cache 版本的 GPT
├── generate.py     # 流式输出 + 默认开启 cache
├── benchmark.py    # 朴素 vs 缓存 的速度对比
└── README.md
```

## 快速开始

```powershell
pip install -r step8/requirements.txt

# 1) 速度基准（默认加载 step7 的 ckpt，没有就改 --ckpt 指向 step5/6）
D:/Users/A/anaconda3/envs/py310_llm/python.exe step8/benchmark.py

# 不同模型 / token 数也能测
D:/Users/A/anaconda3/envs/py310_llm/python.exe step8/benchmark.py --ckpt ../step5/out/ckpt.pt --n_tokens 200 --n_runs 5

# 2) 实际推理（KV-cache，流式打印）
D:/Users/A/anaconda3/envs/py310_llm/python.exe step8/generate.py --message "Tell me a short story." --template hh

# 3) 关闭 cache 做对比
D:/Users/A/anaconda3/envs/py310_llm/python.exe step8/generate.py --message "Tell me a short story." --no-cache

# 4) 不同模板：step5 base 续写、step6 SFT、step7 DPO
D:/Users/A/anaconda3/envs/py310_llm/python.exe step8/generate.py --ckpt ../step5/out/ckpt.pt --template none --message "Once upon a time"
D:/Users/A/anaconda3/envs/py310_llm/python.exe step8/generate.py --ckpt ../step6/out/ckpt.pt --template alpaca --message "Translate to French: I love you."
D:/Users/A/anaconda3/envs/py310_llm/python.exe step8/generate.py --ckpt ../step7/out/ckpt.pt --template hh --message "What is the meaning of life?"
```

## 预期加速

- 在 4060 Ti 上生成 200 token，12M 模型：
  - 朴素：~1500 ms
  - KV-cache：~300 ms
  - **加速比 ≈ 4~6x**
- 模型越大、生成越长，KV-cache 优势越明显
- 真实大模型场景（LLaMA-7B 等）加速通常在 **10~50x**

## 正确性保证

`benchmark.py` 会在相同 seed / 相同采样设置下，比对两种方法输出的 token 序列是否完全一致。
**Outputs identical: True** 表示 KV-cache 的实现没有破坏数学等价性。

## 关于上下文超长的处理

当生成长度逐渐逼近 `block_size`（默认 512），cache 会塞满。本实现里采用最简单的"滑窗丢弃最早 token"，**仅作演示**，会让 RoPE 的位置编码错乱。生产场景应使用：

- **滑动窗口注意力**（Mistral 的方案）
- **YaRN / NTK-aware RoPE 缩放**
- **Attention Sinks**（保留头部 N 个 token）

这些可以在后续步骤继续探索。
