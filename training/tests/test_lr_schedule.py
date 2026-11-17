"""`lr_at` 的口径：多遍学习时步长必须收。"""

from __future__ import annotations

import unittest

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from update_rules import lr_at  # noqa: E402


class Const(unittest.TestCase):
    def test_no_schedule_keeps_base(self):
        for s in (1, 7, 100):
            self.assertAlmostEqual(lr_at(s, 100, 1e-4), 1e-4)

    def test_const_with_warmup_still_rises_then_holds(self):
        self.assertAlmostEqual(lr_at(1, 100, 1e-4, warmup=10), 1e-5)
        self.assertAlmostEqual(lr_at(5, 100, 1e-4, warmup=10), 5e-5)
        self.assertAlmostEqual(lr_at(10, 100, 1e-4, warmup=10), 1e-4)
        self.assertAlmostEqual(lr_at(50, 100, 1e-4, warmup=10), 1e-4)


class Decay(unittest.TestCase):
    def test_cosine_hits_both_ends(self):
        self.assertAlmostEqual(lr_at(1, 100, 1e-4, schedule="cosine"), 1e-4, places=7)
        # 末步不一定正好是 final（cos 在 p=1 处才是），但必须已经很接近
        self.assertLess(lr_at(100, 100, 1e-4, schedule="cosine"), 1.1e-5)

    def test_linear_is_monotone(self):
        lrs = [lr_at(s, 40, 1e-4, schedule="linear") for s in range(1, 41)]
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(lrs, lrs[1:])), lrs)
        self.assertLess(lrs[-1], lrs[0] / 5)

    def test_second_pass_is_smaller_than_first(self):
        # 用户提的现象：第二遍还在用第一遍的步长。40 步 = 20 题 × 2 遍时，
        # 第 21 步（第二遍开头）必须明显小于第 1 步。
        self.assertLess(lr_at(21, 40, 1e-4, schedule="cosine"),
                        lr_at(1, 40, 1e-4, schedule="cosine") * 0.6)

    def test_final_override(self):
        self.assertAlmostEqual(lr_at(1000, 1000, 1e-4, final=1e-6, schedule="linear"), 1e-6,
                               places=12)

    def test_bad_schedule_is_loud(self):
        with self.assertRaises(ValueError):
            lr_at(1, 10, 1e-4, schedule="exp")


if __name__ == "__main__":
    unittest.main()
