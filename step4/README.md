# 第四步：RMSNorm

在 step3 基础上做一个改动：把所有 `nn.LayerNorm` 换成 **RMSNorm**。

| 项目 | step3 | step4 |
|---|---|---|
| 归一化 | LayerNorm | **RMSNorm** |
| 位置编码 | RoPE | RoPE |
| 数据集 | Shakespeare + TinyStories | 同左 |
| 其它 | 同左 | 同左 |

## RMSNorm vs LayerNorm

LayerNorm：
```
mean = x.mean(-1)
var  = x.var(-1)
y = (x - mean) / sqrt(var + eps) * gamma + beta
```

RMSNorm：
```
rms = sqrt( mean(x^2, -1) + eps )
y   = x / rms * gamma
```

差别：
- **没有"减均值"那一步** → 更快
- **没有 beta** → 参数更少
- 效果相当或更好（多篇论文/工程实验验证）
- **LLaMA、Qwen、Mistral、ChatGLM 等现代 LLM 全部采用 RMSNorm**

## 代码改动

只动了 `model.py`：
1. 新增 `class RMSNorm`（约 15 行）
2. `Block.__init__` 里 `nn.LayerNorm` → `RMSNorm`（×2）
3. `GPT.__init__` 里最后一层 `ln_f`：`nn.LayerNorm` → `RMSNorm`
4. 配置加 `norm_eps: float = 1e-5`

其它文件（`data.py` / `train.py` / `generate.py` / `plot.py`）与 step3 完全一致。

## 快速开始

```powershell
pip install -r step4/requirements.txt

D:/Users/A/anaconda3/envs/py310_llm/python.exe step4/data.py
D:/Users/A/anaconda3/envs/py310_llm/python.exe step4/train.py
D:/Users/A/anaconda3/envs/py310_llm/python.exe step4/generate.py --prompt "Once upon a time" --max_new_tokens 300
D:/Users/A/anaconda3/envs/py310_llm/python.exe step4/plot.py
```

> 数据集与 step3 完全相同，可直接拷贝 `step3/data` 到 `step4/data` 省去重新下载。

## 预期现象

- val loss 应与 step3 接近（可能略好一点点）
- 训练速度略快（每步快几个百分点）
- 参数总数比 step3 少几千个（每层 LayerNorm 的 beta 都没了）
