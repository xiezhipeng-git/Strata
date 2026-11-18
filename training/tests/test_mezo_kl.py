"""MeZO 的 KL 锚：**零额外 SCORE**，这才是它能用在满上下文上的原因。

反传线的 KL 要多一次完整前向（1366 token 在 48 GiB 卡上直接 OOM，400 token 也 OOM ——
2026-11-18 实测），所以那条路在长序列上根本不成立。零阶不一样：`theta0_cache` 在训练前
已经为每条轨迹把 θ₀ 的 per-token logp 算进 `it["old"]`（那是 ratio 的分母，本来就要），
KL 的 ref 就是这份缓存 —— 每步只多一次 O(T) 的 numpy，**一次 SCORE 都不多打**。

要守住的：
  1. `--kl-coef 0`（默认）时 `loss_at` 与原来逐位相同，且 SCORE 次数不变；
  2. `--kl-coef > 0` 时 SCORE 次数**仍然不变**（这是整个设计的立足点，回归了就没意义了）；
  3. θ₀ 缓存与当前 logp 形状不符必须报错 —— numpy 会静默广播，KL 就变成一个假数字。
"""

from __future__ import annotations

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

import types
import unittest

import numpy as np

import mezo
import objectives


class _Eng:
    """假引擎：固定 slp，并记下被 SCORE 了几次。"""

    def __init__(self, slp):
        self.slp = np.asarray(slp, dtype=np.float32)
        self.calls = 0

    def score(self, seq, **kw):
        self.calls += 1
        return self.slp, 0.0


SLP = np.array([-1.0, -2.0, -0.5, -0.3], dtype=np.float32)   # seq 长 5 → 回 4 个位置
OLD = np.array([-1.5, -1.4], dtype=np.float32)               # 比当前 logp（slp[2:]）低


def _item(reward):
    return {"seq": [1, 2, 3, 4, 5], "start": 2, "nseg": 2,
            "old": OLD.copy(), "reward": reward}


def _inst(kl_coef):
    m = object.__new__(mezo.GrpoMezo)
    m.eng = _Eng(SLP)
    m.args = types.SimpleNamespace(clip=0.2, group_size=2, kl_coef=kl_coef)
    m.plus_path = "unused"
    m.swap = lambda th, p: None
    m.objective = "grpo"
    return m


class TestMezoKl(unittest.TestCase):

    def test_off_by_default_matches_baseline(self):
        m = _inst(0.0)
        got = m.loss_at({}, [_item(0.0), _item(1.0)])
        # 手算：adv ±0.7071，logp = slp[2:]，denom = 2+2
        adv, _u, _z = objectives.group_advantages(np.array([0.0, 1.0], np.float32), 2)
        cfg = objectives.GrpoCfg(clip_low=0.2, clip_high=0.2)
        logp = SLP[2:]
        want = sum(float(objectives.grpo_loss(logp, OLD, adv[i], np.ones_like(logp), cfg,
                                             denom=4.0)[0]) for i in range(2))
        self.assertAlmostEqual(got, want, places=6)

    def test_kl_adds_penalty(self):
        base = _inst(0.0).loss_at({}, [_item(0.0), _item(1.0)])
        with_kl = _inst(1.0).loss_at({}, [_item(0.0), _item(1.0)])
        self.assertGreater(with_kl, base)

    def test_kl_costs_no_extra_score(self):
        """核心：开不开 KL，SCORE 次数都是「每条一次」（±εz 各一遍由外层负责）。"""
        m0 = _inst(0.0)
        m0.loss_at({}, [_item(0.0), _item(1.0)])
        m1 = _inst(1.0)
        m1.loss_at({}, [_item(0.0), _item(1.0)])
        self.assertEqual(m0.eng.calls, 2)
        self.assertEqual(m1.eng.calls, 2)

    def test_shape_mismatch_is_loud(self):
        """θ₀ 缓存与当前 logp 形状不符必须报错：numpy 广播会让 KL 变成一个假数字。"""
        m = _inst(1.0)
        bad = _item(0.0)
        bad["old"] = np.array([-1.0], dtype=np.float32)      # 少一个
        # 必须凑够 --group-size 条：单条组会先被 zero_std 早退掉，走不到 KL
        with self.assertRaises(SystemExit):
            m.loss_at({}, [bad, _item(1.0)])

    def test_zero_std_group_skips_everything(self):
        """组内全对/全错 ⇒ 没有优势也没有 SCORE（KL 也不该把它变成"有信号"）。"""
        m = _inst(1.0)
        self.assertEqual(m.loss_at({}, [_item(1.0), _item(1.0)]), 0.0)
        self.assertEqual(m.eng.calls, 0)


if __name__ == "__main__":
    unittest.main()
