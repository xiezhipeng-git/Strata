"""学习率调度 —— 一条曲线，两条训练线共用。

为什么单独一个文件：`lr_at` 是纯函数（只依赖 step 与四个超参），但它在零阶线里长在
一个 2000 行的训练器上、在反传线里又各自写一遍是迟早的事。零阶特别需要它，因为
`Δθ = lr·pj·z` 里没有 Adam 的逐维归一化 —— lr 就是**字面意义上的步长**，曲线形状
直接决定每一步走多远。

用 Smith & Topin（arXiv 1708.07120）的 1cycle 衰减段：先用 LR range test 定出安全
上限，再从一个周期内降到接近 0。`lr` 是上限，`lr_final` 是终点，`warmup` 是可选的
上升段占比（零阶没有二阶矩要预热，默认 0）。
"""

from __future__ import annotations

import math


def lr_at(schedule: str, lr: float, lr_final: float, warmup: float,
          total_steps: int, step: int) -> float:
    """第 `step` 步的学习率。

    schedule: "const" | "linear" | "cosine"（其余按 cosine 处理）
    """
    if schedule == "const":
        return lr
    warm = int(round(total_steps * warmup))
    if warm > 0 and step < warm:                       # 上升段：lr_final → lr
        f = (step + 1) / warm
        return lr_final + (lr - lr_final) * f
    rest = max(total_steps - max(warm, 0), 1)
    t = min((step - warm) / rest, 1.0)
    if schedule == "linear":
        return lr + (lr_final - lr) * t
    return lr_final + 0.5 * (lr - lr_final) * (1 + math.cos(math.pi * t))
