"""rank 1 / 只训 B 的容量约束：Δlogits 与上下文无关（纯 numpy，秒过）。

推导：output projection 上的 LoRA 增量是 ΔW = (alpha/r)·B A（A 是 (r, n_in)，
B 是 (n_out, r)）。
    Δlogits(h) = U · ΔW · h = (U·B) · (alpha/r) · (A h)
r=1 且 A 固定时，A h 是标量 ⇒ 任意隐藏态下的词表变化都沿 **U·B 这一个向量**，
只有正负和大小随 h 变。所以 r=1 学得会「全局抬高某几个 token」，学不会
「在这个位置输出 当前，下个位置输出 版本，再下个位置输出 是」这样一串新 token。

这不是估计量噪声、不是 lr、不是数据形状，加步数无解。实测（`bias_probe.py`，
idM 4000 步 / rank 1）：`当前` 在训练闸门位置 CE −6.09，在算术/写诗/讲笑话等
无关上下文反而 −9.45；生成塌成「当前当前当前…」、success 0/5。
"""

import pathlib as _p
import sys
import unittest

import numpy as np

sys.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))


def delta_logits(rank: int, h: np.ndarray, A: np.ndarray, B: np.ndarray, U: np.ndarray) -> np.ndarray:
    """U: (vocab, n_out)；A: (r, n_in)；B: (n_out, r)；h: (n_in,)。"""
    d_hidden = (B @ A) @ h
    return U @ d_hidden


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestRankOneIsAContextFreeBias(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n_in, self.n_out, self.vocab = 32, 24, 64
        self.U = rng.normal(size=(self.vocab, self.n_out))
        self.A4 = rng.normal(size=(4, self.n_in))
        self.B4 = rng.normal(size=(self.n_out, 4))
        self.hs = [rng.normal(size=self.n_in) for _ in range(6)]

    def test_rank1_is_perfectly_collinear_across_contexts(self):
        A, B = self.A4[:1], self.B4[:, :1]
        ref = delta_logits(1, self.hs[0], A, B, self.U)
        for i, h in enumerate(self.hs[1:], 1):
            d = delta_logits(1, h, A, B, self.U)
            self.assertAlmostEqual(abs(cos(ref, d)), 1.0, places=10,
                                   msg=f"h[{i}] 的 Δlogits 必须与 h[0] 共线")
            # 每个上下文的变化只是同一个向量的标量倍：逐个分量求比值应一致
            ratio = d / ref
            np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9, atol=1e-9)

    def test_rank4_can_move_different_directions(self):
        ds = [delta_logits(4, h, self.A4, self.B4, self.U) for h in self.hs]
        cs = [abs(cos(ds[0], d)) for d in ds[1:]]
        self.assertLess(max(cs), 0.99,
                        msg=f"rank 4 不应退化成单一方向，实测 |cos| = {cs}")
        # 至少能表达出 4 个线性无关的词义方向
        self.assertEqual(np.linalg.matrix_rank(np.stack(ds)), 4)

    def test_rank1_changes_the_same_vocab_entries_everywhere(self):
        """把「目标 token 的 CE 下降」翻译成 toy 判据：抬高的词表项与上下文无关。"""
        A, B = self.A4[:1], self.B4[:, :1]
        ref = delta_logits(1, self.hs[0], A, B, self.U)
        top0 = set(np.argsort(ref)[-5:])
        bot0 = set(np.argsort(ref)[:5])
        for h in self.hs[1:]:
            d = delta_logits(1, h, A, B, self.U)
            top = set(np.argsort(d)[-5:])
            # 标量系数可能为负 ⇒ 抬高/压低互换，但仍必须是同一批词表项
            self.assertTrue(top == top0 or top == bot0,
                            "rank 1 抬高/压低的必须是同一批 token")


if __name__ == "__main__":
    unittest.main(verbosity=2)
