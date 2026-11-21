"""RL 次级算法兜底的接线判据（mezo 回放线）。

分层（2026-11-18 定稿）：判据与损失在**算法层**（`objectives.fallback_group_indices`
/ `sft_loss` / `dpo_loss`，mezo 与 backward 共用一份，判据本身在 test_objectives.py
守着）；本文件守的是 mezo 回放线的接线：

  1. load 时就能从固定奖励里挑出全错组，材料（gold 序列 / 偏好对）备好且组内共享一份；
  2. `loss_at` 撞上 zero_std 时不再空转：全错组 → 兜底 loss，全对组 → 0；
  3. 缺 gold 报错（不静默降级）；`--rl-fallback off` 逐字回到旧行为；
  4. dpo 兜底的 ref 在 θ₀ 缓存阶段预打（`_fallback_refs`）。
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
    """假引擎：SCORE 返回固定 slp（逐位 −1 ⇒ CE 恰为 1.0/token），并记调用次数。"""

    def __init__(self):
        self.calls = 0

    def score(self, seq, **kw):
        self.calls += 1
        return np.full(max(len(seq) - 1, 1), -1.0, dtype=np.float32), 0.0


class _Tok:
    """假分词器：确定性、可逆长度（每字符一个 token）。"""

    def encode(self, s, parse_special=True):
        return [10 + (ord(c) % 50) for c in s]


def _items(rewards, golds):
    items = []
    for i, (rw, gold) in enumerate(zip(rewards, golds)):
        # 3 个 prompt token + 1 个完成 token；gold 兜底序列会在 _build_fallback 里建
        items.append({"seq": [1, 2, 3, 100 + i], "start": 3, "nseg": 1,
                      "reward": rw, "gold": gold, "turns": 1, "sampler_lp": None})
    return items


def _inst(mode, items, k=2):
    m = object.__new__(mezo.GrpoMezo)
    m.eng = _Eng()
    m.args = types.SimpleNamespace(clip=0.2, group_size=k, rl_fallback=mode, beta=0.1)
    m.tok = _Tok()
    m.eos = None
    m.objective = "grpo"
    m._warned = set()
    m._build_fallback(items)
    return m


class TestBuildFallback(unittest.TestCase):

    def test_sft_marks_only_fail_groups(self):
        # 组0 [1,0] 有信号、组1 [0,0] 全错 → 只有组1 有 gold_item
        items = _items([1.0, 0.0, 0.0, 0.0], ["x", "y", "abcd", "abcd"])
        m = _inst("sft", items)
        self.assertEqual(m.fb_groups, 1)
        self.assertIsNone(items[0].get("gold_item"))
        gi = items[2]["gold_item"]
        self.assertIsNotNone(gi)
        self.assertIs(items[3]["gold_item"], gi)          # 组内共享一份
        want_body = _Tok().encode("abcd")                 # eos=None ⇒ 不补
        self.assertEqual(gi["seq"][:3], [1, 2, 3])        # prompt 头 = 采样那条
        self.assertEqual(gi["seq"][3:], want_body)
        self.assertEqual(gi["start"], 3)
        self.assertEqual(gi["nseg"], len(want_body))

    def test_missing_gold_is_an_error(self):
        items = _items([0.0, 0.0], ["", ""])
        with self.assertRaises(SystemExit):
            _inst("sft", items)

    def test_off_is_the_old_behavior(self):
        items = _items([0.0, 0.0], ["abcd", "abcd"])
        m = _inst("off", items)
        self.assertEqual(m.fb_groups, 0)
        self.assertTrue(all(it.get("gold_item") is None for it in items))

    def test_all_pass_and_equal_nonzero_never_trigger(self):
        items = _items([1.0, 1.0, 2.0, 2.0], ["a", "b", "c", "d"])
        m = _inst("sft", items)
        self.assertEqual(m.fb_groups, 0)

    def test_dpo_pair_shape(self):
        items = _items([0.0, 0.0], ["abcd", "abcd"])
        m = _inst("dpo", items)
        pair = items[0]["dpo_fb"]
        self.assertIs(items[1]["dpo_fb"], pair)
        self.assertEqual(pair["c_seq"][3:], _Tok().encode("abcd"))
        self.assertEqual(pair["r_seq"], [1, 2, 3, 100])   # rejected = 首条失败轨迹
        self.assertEqual(pair["c_ntok"], 4)
        self.assertEqual(pair["r_ntok"], 1)


class TestFallbackLoss(unittest.TestCase):

    def _m(self, mode, rewards, golds, k=2):
        items = _items(rewards, golds)
        return _inst(mode, items, k), items

    def test_sft_loss_is_one_nats_per_token(self):
        m, items = self._m("sft", [0.0, 0.0], ["abcd", "abcd"])
        # score 全 −1 ⇒ gold 段每 token CE = 1.0（sft_loss 的 mean 口径）
        got = m._fallback_loss(items)
        self.assertAlmostEqual(got, 1.0, places=6)
        self.assertEqual(m.eng.calls, 1)                  # 组内共享 ⇒ 只 SCORE 一次

    def test_all_pass_group_returns_zero_without_scoring(self):
        m, items = self._m("sft", [1.0, 1.0], ["abcd", "abcd"])
        self.assertEqual(m._fallback_loss(items), 0.0)
        self.assertEqual(m.eng.calls, 0)

    def test_off_mode_returns_zero(self):
        m, items = self._m("off", [0.0, 0.0], ["abcd", "abcd"])
        self.assertEqual(m._fallback_loss(items), 0.0)
        self.assertEqual(m.eng.calls, 0)

    def test_dpo_zero_margin_is_log2(self):
        # pol 与 ref 全部 −1/token ⇒ margin=0 ⇒ loss = −log σ(0) = log 2
        m, items = self._m("dpo", [0.0, 0.0], ["abcd", "abcd"])
        pair = items[0]["dpo_fb"]
        slp_c, _ = m.eng.score(pair["c_seq"]); slp_r, _ = m.eng.score(pair["r_seq"])
        # pol = Σ logp（注意不是 sft_loss 的 −logp）
        pol_c = float(np.sum(slp_c[pair["c_start"]:]))
        pol_r = float(np.sum(slp_r[pair["r_start"]:]))    # 1-token 完成：answer 段空 ⇒ 0
        pair["ref_c"], pair["ref_r"] = pol_c, pol_r
        want = float(objectives.dpo_loss(
            0.0, objectives.dpo_beta_effective(0.1, pair["c_ntok"], pair["r_ntok"])))
        self.assertAlmostEqual(m._fallback_loss(items), want, places=6)

    def test_loss_at_routes_zero_std_to_fallback(self):
        # 整链路：loss_at 撞上 zero_std 必须落到兜底，而不是回 0
        m, items = self._m("sft", [0.0, 0.0], ["abcd", "abcd"])
        m.plus_path = "unused"
        m.swap = lambda th, p: None
        got = m.loss_at({}, items)
        self.assertAlmostEqual(got, 1.0, places=6)


class TestFallbackRefs(unittest.TestCase):

    def test_dpo_refs_scored_at_theta0(self):
        items = _items([0.0, 0.0], ["abcd", "abcd"])
        m = _inst("dpo", items)
        m.items = items           # 真实流程里 theta0_cache 在 load_items 之后，self.items 已就位
        m._fallback_refs()
        pair = items[0]["dpo_fb"]
        # score 全 −1/token：c 段按 answer_seg 口径是 3 个位置（不含答案首 token）⇒ −3；
        # rejected 是 1-token 完成 ⇒ 空段 = 0（单 token 轨迹本就该被长度门挡，这里只测接线）
        self.assertAlmostEqual(pair["ref_c"], -3.0, places=6)
        self.assertAlmostEqual(pair["ref_r"], 0.0, places=6)

    def test_refs_noop_for_sft_and_off(self):
        for mode in ("sft", "off"):
            items = _items([0.0, 0.0], ["abcd", "abcd"])
            m = _inst(mode, items)
            m._fallback_refs()          # 不应抛错、也不应碰引擎
            self.assertEqual(m.eng.calls, 0)


if __name__ == "__main__":
    unittest.main()
