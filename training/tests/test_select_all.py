"""`--select-tokens all`（全段进 loss + 难位置加权）的纯逻辑判据。

背景：`excess` 只留 5.8% 的 token ⇒ 其余位置漂走；`off` 全段等权 ⇒ 难位置被 500 个
便宜 token 稀释到 1/530 的零阶差分权重、压不动（sft 12000→20000 步实测 CE 卡在
0.27~0.29，句号后首 token 的 logp −1.93→−1.90 而 EOS −2.24→−1.75）。
`all` 是两者的交集，这里锁住它的两个不变量：mask 必须满、hot 只由阈值决定。
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import guard  # noqa: E402


class TestSelectAll(unittest.TestCase):
    def test_mask_is_full(self):
        """每一位都必须进 loss —— 这条一旦破了就退回 excess 的漂移。"""
        ce = [0.05, 0.1, 0.2, 3.0, 0.08, 12.0]
        mask, _hot = guard.select_all(ce, 1.0)
        self.assertEqual(len(mask), len(ce))
        self.assertTrue(all(mask))
        self.assertTrue(all(m is True for m in mask))

    def test_hot_is_exactly_the_expensive_positions(self):
        ce = [0.05, 0.1, 0.2, 3.0, 0.08, 12.0]
        _mask, hot = guard.select_all(ce, 1.0)
        self.assertEqual(hot, [3, 5])

    def test_thresh_is_strict(self):
        """严格大于：正好等于阈值的不算（与 select_mask 同一口径）。"""
        _m, hot = guard.select_all([1.0, 1.001], 1.0)
        self.assertEqual(hot, [1])

    def test_no_hot_still_full_mask(self):
        """一条样本全都会（CE 都 < 阈值）时 mask 仍满 —— 加权缺失但梯度在。"""
        mask, hot = guard.select_all([0.01, 0.02], 1.0)
        self.assertEqual(hot, [])
        self.assertTrue(all(mask))

    def test_empty(self):
        self.assertEqual(guard.select_all([], 1.0), ([], []))

    def test_weights_align_with_full_mask(self):
        """span_weights 的返回长度必须等于 mask 的 True 数（调用方按 seg[sel] 对齐）。"""
        ce = [0.05, 0.1, 0.2, 3.0, 0.08, 12.0]
        mask, hot = guard.select_all(ce, 1.0)
        w = guard.span_weights(mask, hot, 20.0)
        self.assertIsNotNone(w)
        self.assertEqual(len(w), sum(1 for m in mask if m))
        self.assertEqual(w[hot[0]], 20.0)
        self.assertEqual(w[0], 1.0)

    def test_weight_mass_is_bounded(self):
        """加权必须按 Σw 归一 ⇒ 权重翻倍不放大步长（零阶是纯 SGD，标量就是步长）。

        这里给的心算依据：6 个 token、2 个 hot、W=20 ⇒ Σw = 4*1 + 2*20 = 44，
        hot 占 40/44 = 91%；W=2 时 Σw = 4 + 4 = 8，hot 占 50%。
        """
        ce = [0.05, 0.1, 0.2, 3.0, 0.08, 12.0]
        mask, hot = guard.select_all(ce, 1.0)
        for W, want in ((20.0, 40 / 44), (2.0, 0.5)):
            w = guard.span_weights(mask, hot, W)
            got = sum(w[i] for i in [3, 5]) / sum(w)
            self.assertAlmostEqual(got, want, places=6)

    def test_span_weights_none_when_not_worth_it(self):
        """W<=1 等于不加权，返回 None（调用方走旧口径，不引入无意义的数组）。"""
        mask, hot = guard.select_all([0.1, 0.2, 5.0], 1.0)
        self.assertIsNone(guard.span_weights(mask, hot, 1.0))
        self.assertIsNone(guard.span_weights(mask, hot, 0.0))


if __name__ == "__main__":
    unittest.main()
