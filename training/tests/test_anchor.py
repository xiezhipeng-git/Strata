"""KL 锚（`--anchor-weight`）的回归锁。

    python -m unittest test_anchor

要守住的五件事：

1. **默认逐位不变**：λ=0 时不走全行 SCORE、不加锚项，历史 run 的 pj 一个 bit 都不动。
2. **锚治的正是污染**：只动 prompt 行（任务监督不到的位置）的 logp，λ>0 时 loss 必须变，
   λ=0 时必须不变 —— 数据侧三臂对照已定案：损坏全发生在监督盲区。
3. **锚不碰任务行**：只动答案行（任务行）的 logp 时，λ=1 与 λ=0 的 loss 差 == 纯任务差，
   即锚分量为 0。锚不许跟任务抢梯度。
4. **起点全对的行也要锚**：excess 下 `sel.any()==False` 的条目（旧逻辑直接跳过）在 λ>0 时
   必须仍贡献锚 —— 否则「已会的行为」没人钉，照样漂。
5. **缺 θ₀ 参照必须响**：`lp0` 缓存缺失时抛错，不许静默把锚当 0（伪装成在工作）。

这些测试不需要引擎：score 用数组桩，answer_seg 走真实实现。
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

import mezo

# 5 行 logp：行 0-1 是 prompt（任务监督不到），行 2-4 是答案（start=2）。
LP = np.array([-1.5, -2.5, -3.0, -4.0, -5.0], dtype=np.float32)


class _Sft(mezo.SftMezo):
    """loss_at 的最小桩：swap 不做事，score 返回预置 logp 并记录 keep 请求。"""

    def __init__(self, item, lam=0.0, reduction="mean"):
        self.args = argparse.Namespace(loss_reduction=reduction,
                                       anchor_weight=lam)
        self.plus_path = "unused"
        self.eng = self
        self._item = item
        self.keeps = []

    def swap(self, th, path):
        pass

    def score(self, seq, keep=None):
        self.keeps.append(keep)
        return seq, None


def _item(sel=None, keep=None, lp0=None):
    it = {"seq": LP, "start": 2, "sel": None if sel is None else np.asarray(sel, bool)}
    if keep is not None:
        it["keep"] = np.asarray(keep, dtype=bool)
    if lp0 is not None:
        it["lp0"] = np.asarray(lp0, dtype=np.float32)
    return it


KEEP = np.array([False, False, True, True, True])   # off 口径：答案行全留


class Anchor(unittest.TestCase):

    def test_lam0_is_bitwise_the_old_path(self):
        it = _item(keep=KEEP)
        s = _Sft(it, lam=0.0)
        want = -float(np.mean(LP[2:]))
        self.assertAlmostEqual(s.loss_at({}, [it]), want, places=6)
        self.assertTrue(np.array_equal(s.keeps[0], KEEP))   # 没走全行 SCORE

    def test_prompt_drift_is_punished_only_with_lam(self):
        # 「当前 logp」= item["seq"]（桩模式沿用 test_gate_weight）；
        # prompt 行漂 +2 nats、答案行不动。
        lp2 = LP.copy()
        lp2[:2] += 2.0
        base = _Sft(_item(keep=KEEP, lp0=LP), lam=0.0)
        base._item["seq"] = lp2
        anch = _Sft(_item(keep=KEEP, lp0=LP), lam=1.0)
        anch._item["seq"] = lp2
        a = base.loss_at({}, [base._item])
        b = anch.loss_at({}, [anch._item])
        self.assertAlmostEqual(a, -float(np.mean(LP[2:])), places=6)   # λ=0 不看 prompt
        # λ=1：锚 = mean(2², 2²) = 4，恰好加在任务项上
        self.assertAlmostEqual(b, a + 4.0, places=5)
        self.assertIsNone(anch.keeps[0])                # 锚模式一次全行 SCORE

    def test_task_rows_are_not_double_pinned(self):
        lp2 = LP.copy()
        lp2[2:] += 1.0                                  # 只动答案行
        base = _Sft(_item(keep=KEEP, lp0=LP), lam=0.0)
        base._item["seq"] = lp2
        anch = _Sft(_item(keep=KEEP, lp0=LP), lam=1.0)
        anch._item["seq"] = lp2
        a = base.loss_at({}, [base._item])
        b = anch.loss_at({}, [anch._item])
        self.assertAlmostEqual(b - a, 0.0, places=6)    # 锚分量为 0，不许抢任务梯度

    def test_perfect_start_item_still_anchors(self):
        # excess 下起点全对：sel 全 False（旧逻辑整条跳过）。λ>0 时必须只剩锚。
        lp2 = LP.copy()
        lp2[:2] += 1.0                                  # prompt 行漂 1 nat
        s = _Sft(_item(sel=np.zeros(3, dtype=bool),
                       keep=np.zeros(5, dtype=bool), lp0=LP), lam=2.0)
        s2 = _Sft(_item(sel=np.zeros(3, dtype=bool),
                        keep=np.zeros(5, dtype=bool), lp0=LP), lam=2.0)
        s2._item["seq"] = lp2
        a = s.loss_at({}, [s._item])                    # 不漂：锚 = 0
        b = s2.loss_at({}, [s2._item])                  # 漂：keep 全 False ⇒ 锚盖全 5 行，
        self.assertAlmostEqual(a, 0.0, places=6)        #   2 × mean(1,1,0,0,0) = 0.8
        self.assertAlmostEqual(b, 0.8, places=5)

    def test_missing_theta0_refuses_silently(self):
        it = _item(keep=KEEP)                           # 没有 lp0
        s = _Sft(it, lam=1.0)
        with self.assertRaises(SystemExit):
            s.loss_at({}, [it])

    def test_theta0_cache_fills_every_item(self):
        items = [{"seq": LP.copy()}, {"seq": LP.copy() * 0.5}]
        calls = []

        class _Cache(_Sft):
            def prepare_selection(self):
                pass

            def score(self, seq, keep=None):
                calls.append(keep)
                return seq * 0.1, None

        s = _Cache(items[0], lam=1.0)
        s.items = items
        s.theta0_cache()
        self.assertEqual(calls, [None, None])           # 全行，不许带 keep
        for it in items:
            self.assertIn("lp0", it)
            self.assertEqual(it["lp0"].shape, (5,))


if __name__ == "__main__":
    unittest.main()
