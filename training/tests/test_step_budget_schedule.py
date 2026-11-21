"""步长预算与 lr 曲线的接口（`EngineMezo._step_max_at`）—— 一条会静默失效的通路。

背景（2026-11-21 用户要求「训练要有效且对已有模型几乎无影响」后实测）：零阶的
`--zo-step-max` 裁的是**施加的位移**（`zo/optim.py`：`scale = cap / ‖lr·v‖`）。
cap 一旦钉死，位移范数就恒等于 cap、与 lr 无关 —— `--lr-schedule cosine` 与
`--lr-final` 对位移**完全没有作用**，训练日志里 lr 一栏照常从 1.0e-03 往下降，
看上去一切正常。实测后果（identity v4 冷启动，cap 0.03、lr 1e-3、1500 步）：

    ‖θ‖ 2.82 → 6.95（2.46×，纯随机游走）
    答案 CE 4.5748 → 0.8328（看起来在学）
    数学保持集 8/12 → 1/12（McNemar p=0.016，12 条全部 length 自环）
    长锚 per-token logp 漂移 −0.728 nats（同进程噪声底 std 0.0047 的 155×）

所以要守两条：

1. `--lr-schedule const` 时上限逐位不变（历史 run 复现）；
2. 非 const 时上限 = `--zo-step-max × lr/lr0`，且夹在 [0, 1] 的比值里
   （warmup 段 lr<lr0、衰减段 lr≥lr_final，两头都不能把上限放大）。
"""

from __future__ import annotations

import pathlib as _p
import sys as _s

for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

import argparse
import unittest

import mezo


class _Shim:
    """只带 `args` 的壳：`_step_max_at` 是纯函数，不需要引擎、θ 或数据。"""

    def __init__(self, **kw):
        self.args = argparse.Namespace(**kw)


class StepMaxFollowsSchedule(unittest.TestCase):

    def test_const_schedule_is_bitwise_unchanged(self):
        """历史路径：`--lr-schedule` 缺省 const ⇒ 上限就是这个数，与 lr 无关。"""
        for lr in (1e-3, 3e-4, 1e-9):
            self.assertEqual(
                mezo.EngineMezo._step_max_at(
                    _Shim(lr_schedule="const", lr=1e-3, zo_step_max=0.03), lr), 0.03)

    def test_no_cap_returns_none(self):
        self.assertIsNone(mezo.EngineMezo._step_max_at(
            _Shim(lr_schedule="cosine", lr=1e-3, zo_step_max=0.0), 5e-4))

    def test_cosine_scales_the_cap(self):
        def cap(lr):
            return mezo.EngineMezo._step_max_at(
                _Shim(lr_schedule="cosine", lr=1e-3, zo_step_max=0.03), lr)

        self.assertAlmostEqual(cap(1e-3), 0.03, places=12)        # 峰点：不放大
        self.assertAlmostEqual(cap(5e-4), 0.015, places=12)       # 半程：减半
        self.assertAlmostEqual(cap(1e-4), 0.003, places=12)       # 收尾：1/10

    def test_warmup_and_overshoot_are_clamped(self):
        """warmup 里 lr<lr0（照比例缩），异常的大 lr 不许把上限放大到 1 倍以上。"""
        sh = _Shim(lr_schedule="cosine", lr=1e-3, zo_step_max=0.03)
        self.assertAlmostEqual(mezo.EngineMezo._step_max_at(sh, 1e-4), 0.003, places=12)
        self.assertAlmostEqual(mezo.EngineMezo._step_max_at(sh, 5e-2), 0.03, places=12)

    def test_negative_lr_never_negative_cap(self):
        sh = _Shim(lr_schedule="cosine", lr=1e-3, zo_step_max=0.03)
        self.assertEqual(mezo.EngineMezo._step_max_at(sh, -1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
