"""
run_all.py —— 一键按顺序跑 step1 ~ step7
============================================================
用法（在仓库根目录）：

    python run_all.py                       # 默认跑 step1-7
    python run_all.py --steps 3-7           # 区间
    python run_all.py --steps 1,3,5         # 指定若干步
    python run_all.py --steps 5             # 只跑某一步
    python run_all.py --steps 3-7 --skip-data    # 数据已就绪，跳过 data.py
    python run_all.py --steps 1-7 --python "D:/Users/A/anaconda3/envs/py310_llm/python.exe"

依赖关系（脚本不会替你解决，仅提示）：
    step6 需要 step5 的 ckpt
    step7 需要 step6 的 ckpt
    所以单独跑 step6 / step7 前请确保前序产物已生成。

行为：
    每个 step 依次执行 data.py 然后 train.py。
    任一步失败会立即终止后续步骤，并以非零退出码返回。
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# 每一步要按顺序执行的脚本
STEP_SCRIPTS = {
    1: ["data.py", "train.py"],
    2: ["data.py", "train.py"],
    3: ["data.py", "train.py"],
    4: ["data.py", "train.py"],
    5: ["data.py", "train.py"],
    6: ["data.py", "train.py"],
    7: ["data.py", "train.py"],
    8: ["data.py", "train.py"],
}


def parse_steps(spec: str):
    """支持 '1-7' / '3-7' / '1,3,5' / '5' 等写法。"""
    result = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            result.update(range(int(a), int(b) + 1))
        else:
            result.add(int(part))
    return sorted(result)


def run_one(python_exe, step: int, scripts, dry_run: bool):
    step_dir = os.path.join(HERE, f"step{step}")
    if not os.path.isdir(step_dir):
        print(f"!! step{step} 目录不存在，跳过 ({step_dir})")
        return True   # 不算失败，继续

    for script in scripts:
        full = os.path.join(step_dir, script)
        if not os.path.isfile(full):
            print(f"!! {full} 不存在，跳过该脚本")
            continue
        banner = "=" * 70
        print(f"\n{banner}")
        print(f">>> step{step} / {script}")
        print(banner)
        if dry_run:
            print(f"(dry-run) {python_exe} {script}    cwd={step_dir}")
            continue
        t0 = time.time()
        ret = subprocess.run([python_exe, script], cwd=step_dir)
        dt = time.time() - t0
        print(f"<<< step{step} / {script}  done in {dt:.1f}s  exit={ret.returncode}")
        if ret.returncode != 0:
            print(f"!! step{step}/{script} 失败，终止后续步骤")
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description="按顺序运行 step1 ~ step7")
    ap.add_argument("--steps", default="1-7",
                    help="要执行的步骤，例如 1-7、3-7、1,3,5、5")
    ap.add_argument("--skip-data", action="store_true",
                    help="跳过每个 step 的 data.py（仅在数据已就绪时用）")
    ap.add_argument("--python", default=sys.executable,
                    help=f"使用的 Python 解释器，默认 {sys.executable}")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要执行的命令，不真正运行")
    args = ap.parse_args()

    steps = parse_steps(args.steps)
    if not steps:
        print("没有要执行的步骤。")
        return 0

    print(f"Python  : {args.python}")
    print(f"Steps   : {steps}")
    print(f"SkipData: {args.skip_data}")
    print(f"DryRun  : {args.dry_run}")

    t_total = time.time()
    for s in steps:
        scripts = STEP_SCRIPTS.get(s)
        if scripts is None:
            print(f"!! 未知 step={s}，跳过")
            continue
        if args.skip_data:
            scripts = [x for x in scripts if x != "data.py"]
        ok = run_one(args.python, s, scripts, args.dry_run)
        if not ok:
            print(f"\n执行中断于 step{s}。已耗时 {time.time()-t_total:.1f}s。")
            return 1

    print(f"\n全部完成，总耗时 {time.time()-t_total:.1f}s。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
