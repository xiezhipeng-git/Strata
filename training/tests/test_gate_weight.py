"""整句监督 + 闸门加权（`--gate-weight`）的回归锁。

    python -m unittest test_gate_weight

要守住的三件事：

1. **不是截断**：`--select-first N` 把第 N+1 位之后的 token 从 loss 里删掉，自由生成
   走到那里只能回落成复读环（2026-11-14 idM / idR4 实测：闸门 CE 中位 10.87→2.82 / 0.14，
   逐 token CE 却是 0.1 / 0.0 / **7.1 / 10.0**，切口正好在 N=2 之后）。
   加权必须保留全部选中的 token —— 尾部位置的 logp 变了，loss 必须跟着变。
2. **默认逐位不变**：不传 `--gate-weight` 时走原来的 `objectives.sft_loss` 分支，
   `guard.gate_weights` 返回 None，历史 run 的 pj 一个 bit 都不动。
3. **归一到 Σw**：零阶是纯 SGD，loss 的标量就是步长（LUSPO 分母事故）。
   把所有权重同乘一个常数不能放大 loss。

这些测试不需要引擎、权重或 torch 后端：score 用数组桩，answer_seg 走真实实现。
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

import numpy as np

import guard
import mezo


LP = np.array([-12.0, -9.0, -7.0, -6.0, -5.0], dtype=np.float32)
CE = [12.0, 9.0, 7.0, 6.0, 5.0]          # 5 个 excess token，闸门是第 0 位


class GateWeights(unittest.TestCase):

    def test_off_by_default(self):
        self.assertIsNone(guard.gate_weights(5, 0.0))
        self.assertIsNone(guard.gate_weights(5, 20.0, gate_n=0))
        self.assertIsNone(guard.gate_weights(0, 20.0))

    def test_keeps_every_selected_token(self):
        # 截断（select-first 2）只剩 2 个 True；加权必须一个都不少
        self.assertEqual(sum(guard.select_mask(CE, 1.0, 2)), 2)
        w = guard.gate_weights(5, 20.0, 2)
        self.assertEqual(w, [20.0, 20.0, 1.0, 1.0, 1.0])
        self.assertEqual(len(w), sum(guard.select_mask(CE, 1.0, 0)))

    def test_gate_share_is_dominant_but_not_exclusive(self):
        w = guard.gate_weights(5, 20.0, 2)
        share = sum(w[:2]) / sum(w)
        self.assertGreater(share, 0.8)       # 闸门拿得到方向（idC 里只有 ~5%）
        self.assertLess(share, 1.0)          # 但后面仍有 loss


class _Sft(mezo.SftMezo):
    """loss_at 的最小桩：swap 不做事，score 直接返回预置的 per-token logp。"""

    def __init__(self, item, reduction="mean"):
        self.args = argparse.Namespace(loss_reduction=reduction)
        self.plus_path = "unused"
        self.eng = self
        self._item = item

    def swap(self, th, path):
        pass

    def score(self, seq, keep=None):
        return seq, None

    def batch(self):
        return [self._item]


def _item(sel, w, lp=None):
    return {"seq": LP if lp is None else lp, "start": 0,
            "sel": np.asarray(sel, dtype=bool), "w": w}


class WeightedLoss(unittest.TestCase):

    def test_plain_path_is_the_old_mean(self):
        it = _item([True, True, True, True, True], None)
        s = _Sft(it)
        self.assertAlmostEqual(s.loss_at({}, s.batch()),
                               -float(np.mean(LP)), places=5)

    def test_weighted_is_a_normalized_dot_product(self):
        w = guard.gate_weights(5, 20.0, 2)
        it = _item([True] * 5, w)
        s = _Sft(it)
        want = -float(np.dot(np.asarray(w), LP)) / sum(w)
        self.assertAlmostEqual(s.loss_at({}, s.batch()), want, places=5)

    def test_tail_positions_still_have_gradient(self):
        """决定性的一条：只改第 3-5 个 token 的 logp。

        截断口径下 loss 必须**完全不变**（那几位压根不在 loss 里 ⇒ 复读环），
        加权口径下必须变（那几位有监督 ⇒ 自由生成走得到）。
        """
        lp2 = LP.copy()
        lp2[2:] += 4.0
        trunc = _item(guard.select_mask(CE, 1.0, 2), None)
        a, b = _Sft(trunc), _Sft(_item(trunc["sel"], None, lp2))
        self.assertEqual(a.loss_at({}, a.batch()), b.loss_at({}, b.batch()))

        full = np.ones(5, dtype=bool)
        w = guard.gate_weights(5, 10.0, 2)
        c, d = _Sft(_item(full, w)), _Sft(_item(full, w, lp2))
        self.assertNotAlmostEqual(c.loss_at({}, c.batch()),
                                  d.loss_at({}, d.batch()), places=4)

    def test_scaling_all_weights_does_not_scale_the_loss(self):
        """零阶是纯 SGD：权重整体放大不能把 loss 标量（=步长）一起放大。"""
        a = _Sft(_item(np.ones(5, dtype=bool), [2.0, 2.0, 0.1, 0.1, 0.1]))
        b = _Sft(_item(np.ones(5, dtype=bool), [20.0, 20.0, 1.0, 1.0, 1.0]))
        self.assertAlmostEqual(a.loss_at({}, a.batch()),
                               b.loss_at({}, b.batch()), places=5)

    def test_sum_reduction_divides_by_rows(self):
        w = guard.gate_weights(5, 20.0, 2)
        it = _item([True] * 5, w)
        s = _Sft(it, reduction="sum")
        want = -float(np.dot(np.asarray(w), LP)) / 1
        self.assertAlmostEqual(s.loss_at({}, s.batch()), want, places=4)


class CliPlumbing(unittest.TestCase):
    """统一入口 train.py 拼出来的参数，mezo.py 必须真的认识（parse_known_args 会静默丢）。"""

    def _args(self, **kw):
        a = argparse.Namespace(algo="sft", dpo_data="", dpo_beta=0.1, model="", steps=5,
                               lr=1e-4, out="o.npz", seed=0, rest=[], data="identity",
                               word="W", rollouts="", prompts="", group=4, max_new=32,
                               temperature=1.0, resample_every=1, clip=0.2,
                               max_context=4096, probe="", conv_data="", epochs=None,
                               lr_schedule="const", prefill_chunk="", metrics="",
                               metrics_port=0, success_data="", eval_every=0,
                               system_prompt="", reasoning_effort="", think_targets=None,
                               select_tokens="excess", select_thresh=1.0, select_first=0,
                               gate_weight=20.0, gate_n=2, resume=False)
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def test_reaches_mezo(self):
        import train as train_mod
        argv = train_mod.build_argv_engine(self._args(), "/tmp/s.gguf", "/tmp/pack",
                                           "/tmp/exe", "/tmp/prof")
        self.assertEqual(argv[argv.index("--gate-weight") + 1], "20.0")
        self.assertEqual(argv[argv.index("--gate-n") + 1], "2")

    def test_not_sent_when_off(self):
        import train as train_mod
        argv = train_mod.build_argv_engine(self._args(gate_weight=0.0), "/tmp/s.gguf",
                                           "/tmp/pack", "/tmp/exe", "/tmp/prof")
        self.assertNotIn("--gate-weight", argv)

    def test_sent_even_without_explicit_select_tokens(self):
        """--select-tokens 不给时 mezo.py 自己默认 excess，加权不该被一起吞掉。"""
        import train as train_mod
        argv = train_mod.build_argv_engine(self._args(select_tokens=None), "/tmp/s.gguf",
                                           "/tmp/pack", "/tmp/exe", "/tmp/prof")
        self.assertEqual(argv[argv.index("--gate-weight") + 1], "20.0")

    def test_mezo_exposes_the_flags(self):
        src = _p.Path(mezo.__file__).read_text(encoding="utf-8")
        self.assertIn('"--gate-weight"', src)
        self.assertIn('"--gate-n"', src)

    def test_resume_reaches_mezo(self):
        """课程式实验要从 idW4000 的 θ 续跑；统一入口漏透传就会静默从头训。"""
        import train as train_mod
        argv = train_mod.build_argv_engine(self._args(resume=True), "/tmp/s.gguf",
                                           "/tmp/pack", "/tmp/exe", "/tmp/prof")
        self.assertIn("--resume", argv)

    def test_resume_absent_by_default(self):
        import train as train_mod
        argv = train_mod.build_argv_engine(self._args(), "/tmp/s.gguf", "/tmp/pack",
                                           "/tmp/exe", "/tmp/prof")
        self.assertNotIn("--resume", argv)


class ItemWeight(unittest.TestCase):
    """item 级权重（jsonl 的 `weight` 字段 → `it["iw"]`）：混训闲聊锚样本时压其占比。

    守住的口径：`Σ iw_i · (-logp)` / `Σ iw_i · n_tok`，与 `--gate-weight` 正交相乘；
    不传 iw（或 =1.0）走旧分支，历史 pj 一个 bit 不动。
    """

    def _it(self, iw, w=None):
        return {"seq": LP, "start": 0, "sel": np.ones(5, dtype=bool), "w": w, "iw": iw}

    def test_item_weight_reweights_tokens(self):
        # iw=0.3 单条：loss = -0.3·ΣLP / (0.3·5) —— 数值上等于无权重（归一抵消），
        # 意义在**多条混合**时的占比，见下一条
        s = _Sft(self._it(0.3))
        self.assertAlmostEqual(s.loss_at({}, s.batch()),
                               -float(np.mean(LP)), places=5)

    def test_mixed_items_share_by_weight(self):
        # identity(1.0, logp=LP) + chat(0.3, logp=LP) 加权后 chat 占比 3/13，均权 1/2
        a = _Sft(self._it(1.0))
        b = _Sft(self._it(0.3))
        mixed = mezo.SftMezo.__new__(mezo.SftMezo)
        mixed.args = argparse.Namespace(loss_reduction="mean")
        mixed.plus_path = "unused"
        mixed.eng = a.eng
        # loss_at 只用 batch/swap/eng，直接借 a 的绑定方法
        loss = a.loss_at({}, [a._item, b._item])
        want = -(float(np.sum(LP)) + 0.3 * float(np.sum(LP))) / (5 + 0.3 * 5)
        self.assertAlmostEqual(loss, want, places=5)

    def test_iw_multiplies_gate_weight(self):
        w = guard.gate_weights(5, 20.0, 2)
        s = _Sft(self._it(0.5, w))
        want = -0.5 * float(np.dot(np.asarray(w), LP)) / (0.5 * sum(w))
        self.assertAlmostEqual(s.loss_at({}, s.batch()), want, places=5)

    def test_iw_one_is_the_old_path(self):
        s1 = _Sft({"seq": LP, "start": 0, "sel": np.ones(5, dtype=bool),
                   "w": None, "iw": None})
        s2 = _Sft({"seq": LP, "start": 0, "sel": np.ones(5, dtype=bool),
                   "w": None, "iw": 1.0})
        self.assertAlmostEqual(s1.loss_at({}, s1.batch()),
                               s2.loss_at({}, s2.batch()), places=6)


if __name__ == "__main__":
    unittest.main()
