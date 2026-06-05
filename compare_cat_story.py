"""
compare_cat_story.py —— 让每一步的模型都生成一个小猫故事，对比效果
============================================================
用法：
    python compare_cat_story.py

会依次调用 step1 ~ step7 的模型（step8 无独立 ckpt，用 step7 的做 KV-cache 对比）。
每个模型用相似的 prompt 让它写一个关于小猫的故事，最后汇总打印方便对比。
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

# 每一步的生成命令配置
# step1 是字符级（只认 Shakespeare 里的字符），prompt 用简单英文
# step2~5 是 BPE，用 --prompt 续写
# step6 是 SFT（Alpaca 模板），用 --instruction
# step7 是 DPO（HH 模板），用 --message
# step8 无独立 ckpt，加载 step7 的权重用 KV-cache 生成

STEPS = [
    {
        "name": "Step 1: 最小 GPT (字符级, ~1M 参数)",
        "cwd": os.path.join(HERE, "step1"),
        "cmd": [PYTHON, "generate.py",
                "--prompt", "Once upon a time, a little cat ",
                "--max_new_tokens", "300",
                "--temperature", "0.8",
                "--top_k", "40",
                "--seed", "42"],
    },
    {
        "name": "Step 2: BPE + 12M 参数",
        "cwd": os.path.join(HERE, "step2"),
        "cmd": [PYTHON, "generate.py",
                "--prompt", "Once upon a time, there was a little cat named Whiskers.",
                "--max_new_tokens", "200",
                "--temperature", "0.8",
                "--top_k", "40",
                "--seed", "42"],
    },
    {
        "name": "Step 3: RoPE + TinyStories 大数据集",
        "cwd": os.path.join(HERE, "step3"),
        "cmd": [PYTHON, "generate.py",
                "--prompt", "Once upon a time, there was a little cat named Whiskers.",
                "--max_new_tokens", "200",
                "--temperature", "0.8",
                "--top_k", "40",
                "--seed", "42"],
    },
    {
        "name": "Step 4: RMSNorm",
        "cwd": os.path.join(HERE, "step4"),
        "cmd": [PYTHON, "generate.py",
                "--prompt", "Once upon a time, there was a little cat named Whiskers.",
                "--max_new_tokens", "200",
                "--temperature", "0.8",
                "--top_k", "40",
                "--seed", "42"],
    },
    {
        "name": "Step 5: SwiGLU 激活 (base 模型最终版)",
        "cwd": os.path.join(HERE, "step5"),
        "cmd": [PYTHON, "generate.py",
                "--prompt", "Once upon a time, there was a little cat named Whiskers.",
                "--max_new_tokens", "200",
                "--temperature", "0.8",
                "--top_k", "40",
                "--seed", "42"],
    },
    {
        "name": "Step 6: SFT 指令微调 (Alpaca)",
        "cwd": os.path.join(HERE, "step6"),
        "cmd": [PYTHON, "generate.py",
                "--instruction", "Tell me a short story about a brave little cat named Whiskers.",
                "--max_new_tokens", "200",
                "--temperature", "0.8",
                "--top_k", "40",
                "--seed", "42"],
    },
    {
        "name": "Step 7: DPO 偏好对齐",
        "cwd": os.path.join(HERE, "step7"),
        "cmd": [PYTHON, "generate.py",
                "--message", "Tell me a short story about a brave little cat named Whiskers.",
                "--max_new_tokens", "200",
                "--temperature", "0.8",
                "--top_k", "40",
                "--seed", "42"],
    },
    {
        "name": "Step 8: KV-cache 加速推理 (加载 step7 权重)",
        "cwd": os.path.join(HERE, "step8"),
        "cmd": [PYTHON, "generate.py",
                "--message", "Tell me a short story about a brave little cat named Whiskers.",
                "--template", "hh",
                "--max_new_tokens", "200",
                "--temperature", "0.8",
                "--top_k", "40",
                "--seed", "42"],
    },
]


def run_step(step_info):
    """运行一个步骤的生成命令，捕获输出。"""
    name = step_info["name"]
    cwd = step_info["cwd"]
    cmd = step_info["cmd"]

    # 检查 ckpt 是否存在
    ckpt_path = os.path.join(cwd, "out", "ckpt.pt")
    # step8 默认加载 step7 的 ckpt
    if "step8" in cwd:
        ckpt_path = os.path.join(HERE, "step7", "out", "ckpt.pt")

    if not os.path.exists(ckpt_path):
        return f"[跳过] 未找到 ckpt: {ckpt_path}"

    try:
        result = subprocess.run(
            cmd, cwd=cwd,
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            return f"[错误] 退出码={result.returncode}\n{result.stderr[:500]}"
        # 提取输出中 "===..." 之间的内容
        output = result.stdout
        return output
    except subprocess.TimeoutExpired:
        return "[超时] 生成超过 60 秒"
    except Exception as e:
        return f"[异常] {e}"


def main():
    print("=" * 70)
    print("  🐱 小猫故事对比：让每一步的模型都写一个关于 Whiskers 的故事")
    print("=" * 70)

    results = []
    for step_info in STEPS:
        print(f"\n>>> 正在运行: {step_info['name']} ...")
        output = run_step(step_info)
        results.append((step_info["name"], output))

    # 汇总打印
    print("\n\n")
    print("=" * 70)
    print("  📊 对比结果汇总")
    print("=" * 70)

    for name, output in results:
        print(f"\n{'─' * 70}")
        print(f"  【{name}】")
        print(f"{'─' * 70}")
        print(output)

    print(f"\n{'═' * 70}")
    print("  对比完毕！")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
