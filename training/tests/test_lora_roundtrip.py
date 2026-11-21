"""adapter GGUF 的读写往返 —— 锁 `--init-adapter` 能真的把 θ 读回来。

背景（2026-11-16 实测）：`mezo.py --init-adapter` 原来只把文件交给引擎 `--lora`，
进程内 θ 还是 `init_theta()` 的 B=0。DPO 因此以「ref = SFT、policy = 基座」跑满 2000 步，
日志无异常，只有起点读数露馅（probe −51.69，SFT 终点 −21.15；answer-ce 3.19，终点 0.30）。
补的读取器必须与 `write_adapter` 逐元素互逆，否则问题只是从「没读」变成「读错」。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "tools")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np  # noqa: E402

from common.lora import (adapter_layout, init_theta, read_adapter, read_adapter_alpha,  # noqa: E402
                         read_adapter_rank, write_adapter)

GEOM = {"block_count": 8, "embedding_length": 32, "attention.head_count": 4,
        "attention.key_length": 8, "full_attention_interval": 4, "ssm.inner_size": 24}


class TestAdapterRoundTrip(unittest.TestCase):
    def setUp(self):
        self.layout = adapter_layout(GEOM)
        self.rng = np.random.default_rng(11)
        self.theta = init_theta(self.layout, rank=4, seed=3)
        # B=0 的骨架读回来永远是 0，测不出符号/转置的错 —— 填实数
        for l, d in self.theta.items():
            for w, (a, b) in d.items():
                d[w] = (a, self.rng.standard_normal(b.shape).astype(np.float32) * 0.05)
        self.tmp = tempfile.mkdtemp()

    def _path(self, name="a.gguf"):
        return os.path.join(self.tmp, name)

    def test_roundtrip_is_elementwise(self):
        """F32 下必须逐位互逆 —— 这条锁的是转置/平铺顺序，不是精度。"""
        write_adapter(self._path(), self.layout, self.theta, alpha=16.0, dtype="f32")
        back = read_adapter(self._path(), self.layout)
        self.assertEqual(sorted(back), sorted(self.layout))
        for l, projs in self.layout.items():
            for which, (n_in, n_out) in projs.items():
                a0, b0 = self.theta[l][which]
                a1, b1 = back[l][which]
                self.assertEqual(a1.shape, (4, n_in), f"层 {l} {which} A 形状")
                self.assertEqual(b1.shape, (n_out, 4), f"层 {l} {which} B 形状")
                self.assertTrue(np.array_equal(a1, a0), f"层 {l} {which} A 不一致（可能是转置或平铺顺序错）")
                self.assertTrue(np.array_equal(b1, b0), f"层 {l} {which} B 不一致")

    def test_default_is_f16_and_still_ordered(self):
        """默认 F16（体积减半，实测 ΔW 误差 2.9e-4）：形状与顺序仍必须对，数值只要求近似。

        这条与上一条分开，是因为 F16 的误差上限本身就是判据 —— 若哪天默认被改成 Q4（ΔW 9.5%），
        rtol 1e-3 会立刻红。"""
        write_adapter(self._path(), self.layout, self.theta, alpha=16.0)
        back = read_adapter(self._path(), self.layout)
        for l, projs in self.layout.items():
            for which, (n_in, n_out) in projs.items():
                a0, b0 = self.theta[l][which]
                a1, b1 = back[l][which]
                self.assertEqual(a1.shape, (4, n_in), f"层 {l} {which} A 形状")
                self.assertEqual(b1.shape, (n_out, 4), f"层 {l} {which} B 形状")
                for tag, got, want in (("A", a1, a0), ("B", b1, b0)):
                    tol = 1e-3 * float(np.abs(want).max() or 1.0)
                    self.assertLess(float(np.abs(got - want).max()), tol,
                                    f"层 {l} {which} {tag} F16 误差超过 1e-3×max")

    def test_rank_and_alpha_read_back(self):
        write_adapter(self._path(), self.layout, self.theta, alpha=8.0)
        self.assertEqual(read_adapter_rank(self._path()), 4)
        self.assertEqual(read_adapter_alpha(self._path()), 8.0)

    def test_missing_layer_is_loud(self):
        """少层的 adapter 不能读成「其余层 B=0」的静默结果。"""
        write_adapter(self._path(), self.layout, self.theta, alpha=16.0)
        wider = {l: v for l, v in self.layout.items()}
        wider[99] = {"ssm_out": (24, 32)}
        with self.assertRaises(SystemExit) as cm:
            read_adapter(self._path(), wider)
        self.assertIn("blk.99", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
