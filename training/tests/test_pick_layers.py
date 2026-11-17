"""`--zo-layers` / `--zo-range` 的回归锁。

    python -m unittest test_pick_layers

要守住的三件事：

1. **`--zo-layers K` 取末 K 层，`--zo-range` 取区段**：历史 100% 的 run 只跑了末 1–2 层，
   而每层只挂一个输出投影（`training/common/lora.py:adapter_layout`）—— 中间层从未被
   放开过。选错层不会有任何报错，只会让「模型学不会」被误读成「零阶不行」。
2. **两个开关同时给必须报错**：一个说末 K 层、一个说区段，猜优先级等于静默换实验。
3. **越界 / 空选 / 负步长必须报错**：静默退化成「全层」会让 d 突然变大 48 倍，
   步长还按旧配置给，跑完才发现位移打飞。

不需要引擎、权重或 torch。
"""

from __future__ import annotations

import pathlib as _p
import sys as _s
import unittest

for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

from guard import block_window, pick_layers  # noqa: E402

L48 = list(range(48))


class TestLastK(unittest.TestCase):
    def test_zero_is_all(self):
        keep, desc = pick_layers(L48, 0, "")
        self.assertEqual(keep, L48)
        self.assertIn("全 48 层", desc)

    def test_last_2(self):
        keep, desc = pick_layers(L48, 2, "")
        self.assertEqual(keep, [46, 47])
        self.assertIn("46..47", desc)

    def test_last_1(self):
        keep, _ = pick_layers(L48, 1, "")
        self.assertEqual(keep, [47], "末 1 层必须是最后那一层；差一层就没有 lm_head 的入口")

    def test_k_exceeds_depth(self):
        with self.assertRaises(ValueError):
            pick_layers(L48, 49, "")

    def test_negative_k(self):
        with self.assertRaises(ValueError):
            pick_layers(L48, -3, "")


class TestRange(unittest.TestCase):
    def test_span_inclusive(self):
        keep, desc = pick_layers(L48, 0, "20,34")
        self.assertEqual(keep, list(range(20, 35)), "区段含两端：20..34 是 15 层")
        self.assertIn("20..34", desc)

    def test_single_layer(self):
        keep, desc = pick_layers(L48, 0, "23")
        self.assertEqual(keep, [23])
        self.assertIn("23", desc)

    def test_stride(self):
        keep, desc = pick_layers(L48, 0, "12,44,4")
        self.assertEqual(keep, [12, 16, 20, 24, 28, 32, 36, 40, 44])
        self.assertIn("每隔 4 层", desc)

    def test_swapped_bounds_accepted(self):
        self.assertEqual(pick_layers(L48, 0, "30,20")[0], list(range(20, 31)),
                         "把「大,小」写反不该静默选空")

    def test_colon_separator(self):
        self.assertEqual(pick_layers(L48, 0, "20:22")[0], [20, 21, 22])

    def test_out_of_range(self):
        for bad in ("40,48", "-1,5"):
            with self.assertRaises(ValueError, msg=bad):
                pick_layers(L48, 0, bad)

    def test_bad_step(self):
        with self.assertRaises(ValueError):
            pick_layers(L48, 0, "10,20,0")

    def test_too_many_parts(self):
        with self.assertRaises(ValueError):
            pick_layers(L48, 0, "10,20,2,3")

    def test_non_numeric(self):
        with self.assertRaises(ValueError):
            pick_layers(L48, 0, "mid")


class TestConflicts(unittest.TestCase):
    def test_both_given(self):
        with self.assertRaises(ValueError):
            pick_layers(L48, 2, "20,34")

    def test_empty_layers(self):
        with self.assertRaises(ValueError):
            pick_layers([], 0, "")


class TestBlockWindow(unittest.TestCase):
    """`--zo-block K`：每步只动 K 层、按步轮转。"""

    def test_zero_is_all(self):
        self.assertEqual(block_window(L48, 7, 0), L48, "0 = 每步全动，别静默变成空集")

    def test_k_not_less_than_depth(self):
        self.assertEqual(block_window(L48, 3, 48), L48)
        self.assertEqual(block_window(L48, 3, 99), L48)

    def test_window_size_and_order(self):
        w = block_window(L48, 0, 4)
        self.assertEqual(w, [0, 1, 2, 3])
        self.assertEqual(block_window(L48, 1, 4), [4, 5, 6, 7], "按 step*k 推进，不重叠不跳层")

    def test_wraps_around(self):
        # 48 层、每步 4 层 ⇒ 12 步一轮；第 12 步回到起点
        self.assertEqual(block_window(L48, 12, 4), block_window(L48, 0, 4))

    def test_covers_every_layer_in_one_round(self):
        seen = []
        for s in range(12):
            seen.extend(block_window(L48, s, 4))
        self.assertEqual(sorted(seen), L48, "一轮之内每层必须恰好被动一次")

    def test_uneven_k_still_covers(self):
        # 5 层、每步 3 层：会重叠，但一轮 ceil(48/5)=10 步之内仍覆盖全部
        seen = set()
        for s in range(10):
            seen.update(block_window(L48, s, 5))
        self.assertEqual(sorted(seen), L48)

    def test_subset_layers_respected(self):
        keep, _ = pick_layers(L48, 0, "20,25")
        self.assertTrue(set(block_window(keep, s, 2)) <= set(keep) for s in range(6))

    def test_empty(self):
        self.assertEqual(block_window([], 0, 4), [])


class TestRealGeometry(unittest.TestCase):
    """真实几何 48 层：`adapter_layout` 每层恰一个矩阵，d = Σ r(n_in+n_out)。"""

    def test_d_matches_layout_arithmetic(self):
        layout = {l: ("attn_output" if l % 4 == 3 else "ssm_out", 6144, 2560) for l in L48}
        d = lambda ls, r: sum(r * (layout[l][1] + layout[l][2]) for l in ls)
        last2_r4, all_r8 = d(pick_layers(L48, 2, "")[0], 4), d(pick_layers(L48, 0, "")[0], 8)
        self.assertEqual(last2_r4, 69632)
        self.assertEqual(all_r8, 3342336)
        self.assertGreater(all_r8 / last2_r4, 40, "全层与末 2 层的 d 差是量级差，步长不能沿用")


if __name__ == "__main__":
    unittest.main()
