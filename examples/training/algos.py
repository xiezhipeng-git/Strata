"""算法注册表：**算法可换、目标函数也可换**，加一条就是往这里加一行。

两件事本来是正交的，之前的三个脚本把它们焊死在一起：

    优化器（怎么动参数）      目标函数（在量什么）
    backward  反传 + SGD      ce    下一 token 的交叉熵（SFT / MeZO 默认）
    zeroth_order  MeZO 零阶   grpo   组内优势 + 裁剪的策略梯度代理
    two_stage  采轨迹 + 反传  →  以后加 ppo / dpo / orm 就是往 OBJECTIVES 加一行

于是：
  `--algo sft`                 = backward × ce      （sft.py）
  `--algo mezo`                = zeroth_order × ce  （sft_mezo.py）
  `--algo mezo --objective grpo` = zeroth_order × grpo  ← 不需要反传就能做 RL
  `--algo grpo`                = two_stage × grpo   （grpo.py，学生侧暂只支持 HF/4bit 目录）

为什么 zeroth_order × grpo 值得存在：MeZO 只要一个**标量目标**，而 GRPO 的目标本来就只
需要 logprob 之比 —— 于是「只靠前向做 RL」在数学上是通的，且显存贴住推理基线（+0.4 GiB）。
代价是零阶噪声 ~√d，需要更多步；所以它是显存受限时的路线，不是通用替代。

加一个新算法（比如 PPO）要动的地方：
  1. 写 `ppo.py`，暴露 `main()`，参数用 argparse；
  2. 在下面 `ALGOS` 加一条，`script`/`kind`/`lr` 填好；
  3. 如果是新的目标函数（不是整个训练循环），在 `OBJECTIVES` 加名字，并在
     `sft_mezo.py --objective` 的 choices 里接上。
`train.py` 不需要改。
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Algo:
    name: str
    script: str                 # examples/training/ 下的文件名，必须暴露 main()
    kind: str                   # backward | zeroth_order | two_stage
    lr: float                   # 该路线的默认学习率（两条路线的量级差 100 倍，不能共用一个）
    blurb: str
    objectives: tuple[str, ...] = ("ce",)
    extra: tuple[str, ...] = ()  # 默认追加给脚本的参数
    needs: str = ""             # 额外依赖（提示用）


ALGOS: dict[str, Algo] = {
    "sft": Algo(
        "sft", "sft.py", "backward", 1e-4,
        "反传 SFT：每步 1 前向 + 1 反传（长序列需 --ckpt 重算），梯度精确",
        extra=("--ckpt",),
    ),
    "mezo": Algo(
        "mezo", "sft_mezo.py", "zeroth_order", 1e-6,
        "MeZO 零阶：每步 2 前向、0 反传，显存贴住推理基线（+0.4 GiB）",
        objectives=("ce", "grpo"),
    ),
    "grpo": Algo(
        "grpo", "grpo.py", "two_stage", 1e-4,
        "两段式 GRPO：引擎采轨迹 → 关掉引擎用学生反传（48GB 放不下两者）",
        objectives=("grpo",),
        needs="一条引擎轨迹（--rollouts），或引擎在跑（--prompts）",
    ),
}

# 目标函数：写在这里是为了「算法能换、目标也能换」这件事有个单一出处。
OBJECTIVES = {
    "ce": "下一 token 的交叉熵（SFT / MeZO 的默认口径）",
    "grpo": "组内归一化优势 × 裁剪后的概率比（GRPO 代理目标）",
}


def names() -> list[str]:
    return list(ALGOS)


def describe() -> str:
    lines = ["算法（--algo）："]
    for a in ALGOS.values():
        lines.append(f"  {a.name:<6s} {a.kind:<13s} lr {a.lr:<7g} {a.blurb}")
    lines.append("目标函数（--objective）：")
    for k, v in OBJECTIVES.items():
        lines.append(f"  {k:<6s} {v}")
    lines.append("  例：--algo mezo --objective grpo  = 只靠前向做 RL（需要 --rollouts）")
    return "\n".join(lines)


def run(name: str, argv: list[str]) -> int:
    """把参数原样交给该算法的脚本，省得每个算法都抄一遍 `--gguf/--pack/--tokenizer`。

    同进程 import 而不是 subprocess：这些脚本都是 `main()` 形式，同进程省掉一次
    CUDA context 初始化（实测 ~2 s）与一次 HF 导入，日志也还在同一条流里。
    """
    algo = ALGOS[name]
    module = importlib.import_module(algo.script[:-3])
    sys.argv = [algo.script] + list(argv)
    try:
        return int(module.main() or 0)
    finally:
        sys.argv = sys.argv[:1]
