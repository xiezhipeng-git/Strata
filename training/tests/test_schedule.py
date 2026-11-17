"""学习率曲线：端点必须精确落在 `lr` 与 `lr_final` 上。

零阶线没有 Adam 的逐维归一化，`Δθ = lr·pj·z` 里 lr 就是字面步长，所以曲线端点差一点
就是每步位移差一点 —— 这条要能离线秒验，而不是跑 3000 步看日志。
"""

import pathlib as _p
import sys as _s
import unittest

for _d in (_p.Path(__file__).resolve().parents[1],):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

from common.schedule import lr_at      # noqa: E402


class Endpoints(unittest.TestCase):
    def test_const(self):
        for s in range(10):
            self.assertEqual(lr_at("const", 1e-4, 1e-6, 0.0, 10, s), 1e-4)

    def test_cosine_hits_both_ends(self):
        self.assertAlmostEqual(lr_at("cosine", 1e-4, 1e-5, 0.0, 100, 0), 1e-4, places=12)
        self.assertAlmostEqual(lr_at("cosine", 1e-4, 1e-5, 0.0, 100, 100), 1e-5, places=12)

    def test_linear_hits_both_ends(self):
        self.assertAlmostEqual(lr_at("linear", 1e-4, 1e-5, 0.0, 100, 0), 1e-4, places=12)
        self.assertAlmostEqual(lr_at("linear", 1e-4, 1e-5, 0.0, 100, 100), 1e-5, places=12)

    def test_cosine_is_monotone_decreasing(self):
        xs = [lr_at("cosine", 1e-4, 1e-5, 0.0, 100, s) for s in range(101)]
        self.assertTrue(all(a >= b - 1e-15 for a, b in zip(xs, xs[1:])))

    def test_warmup_rises_then_hits_ceiling(self):
        """warmup 占 20%：前 20 步上升，第 20 步到达上限，之后衰减。"""
        xs = [lr_at("cosine", 1e-4, 1e-5, 0.2, 100, s) for s in range(100)]
        self.assertLess(xs[0], xs[5])
        self.assertAlmostEqual(xs[19], 1e-4, places=9)
        self.assertLess(xs[50], xs[20])

    def test_zero_total_steps_does_not_divide_by_zero(self):
        """--steps 0（早停后重算等边界）：`rest` 兜到 1，第 0 步取上限，不抛 ZeroDivision。"""
        self.assertAlmostEqual(lr_at("cosine", 1e-4, 1e-5, 0.0, 0, 0), 1e-4, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
