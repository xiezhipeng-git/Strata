"""零阶线的 old logp 复用：采样侧已经算过，就别再打一遍。

`theta0_cache` 原本对每条轨迹 SCORE 一次，为的是拿到「采样时策略」的 per-token logp
（ratio 的分母）。但引擎的 `LP` 行取的是「采样前、惩罚前、温度前」的 head log-softmax
（`src/program/generate.cpp:emit_lp_line`），与这里 SCORE 同一口径 —— 采样侧
（`collect_rollouts.py` / `grpo.py --save-rollouts`）本来就把这串数写进 jsonl 了。

满上下文时这一步是真正的成本：128K 一次 SCORE 45 s，33 条 25 分钟。

要守住的：
  1. **序列就是采样那条**：`prompt_ids` 在就用它，不拿本地渲染替换。奖励是在采样那条
     序列上得的、`token_logprobs` 也是在那条上算的；换成本地渲染等于把模型放到另一段
     上下文里去复现一个不是它的奖励。两者不一致时提示（模板参数没传同档），不挡。
  2. **对齐不可证就不用**：缺 eos 的截断轨迹（补出来的那一位采样时没算过）、prompt 渲染
     不一致的轨迹，一律退回重编码并丢掉 logp —— 宁可贵一次，不要错。
  3. **不静默采信**：抽样复核。allowed_token_ids 重归一化 / `--resume` 时 θ₀ ≠ 采样策略
     / 手工造的 rollouts，三者都是**系统性**偏移（0.05 以上），与浮点噪声（1e-3）差一个
     数量级，抽 3 条就分得开。
  4. 复核不过 ⇒ 整体退回全量 SCORE，**不部分采信**（一半真一半假的分母比全假更糟）。
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

EOS = 99


class _Tok:
    """字符级假 tokenizer：一条文本 → 每个字符一个 id（+100 避开特殊 id）。"""

    def encode(self, text, parse_special=True):
        return [ord(c) % 100 + 100 for c in text]


class _Eng:
    """假引擎：`score(seq)` 返回 len(seq)-1 个位置的固定 logp。"""

    def __init__(self, val=-0.7):
        self.val = float(val)
        self.calls = 0

    def score(self, seq, **kw):
        self.calls += 1
        return np.full(max(0, len(seq) - 1), self.val, dtype=np.float32), 0.0


def _inst(n_items, *, sampler_logp="auto", check=3, tol=0.02, val=-0.7):
    m = object.__new__(mezo.GrpoMezo)
    m.eng = _Eng(val)
    m.tok = _Tok()
    m.render = lambda p: f"Q:{p}A:"
    m.eos = EOS
    m.args = types.SimpleNamespace(sampler_logp=sampler_logp, sampler_logp_check=check,
                                   sampler_logp_tol=tol)
    m.items = []
    return m


def _row(m, prompt="hi", tids=None, lps=None, pids=None):
    """造一行 rollouts；`pids` 缺省 = 本地渲染结果（即「对齐一致」的正常情况）。"""
    local = m.tok.encode(m.render(prompt))
    r = {"prompt": prompt, "completion": "xx", "reward": 1.0,
         "token_ids": tids, "token_logprobs": lps, "prompt_ids": local if pids is None else pids}
    return r


class TestSeqFromRollout(unittest.TestCase):

    def test_aligned_tokens_are_used_without_reencoding(self):
        m = _inst(1)
        tids = [201, 202, 203, EOS]
        lps = [-0.5, -0.6, -0.7, -0.8]
        seq, plen, got = m._seq_from_rollout(_row(m, tids=tids, lps=lps))
        self.assertEqual(seq, m.tok.encode(m.render("hi")) + tids)
        self.assertEqual(plen, len(m.tok.encode(m.render("hi"))))
        self.assertIsNotNone(got)
        # 与 answer_seg 逐位对齐（2026-11-24 修复后不砍首位）：lps[i] 就是 tids[i] 的 logp
        np.testing.assert_allclose(got, lps)

    def test_prompt_ids_mismatch_keeps_sampler_side(self):
        """采样 prompt 与本地渲染不同 ⇒ **跟采样走**（奖励与 logp 都在那条上）。

        这是 2026-11-18 真机实测的形状：采样侧没传 `reasoning_effort`，训练模板默认
        `xhigh`，本地渲染多出整整一段 system（21 tok vs 61 tok）。换成本地渲染去复现
        一个不是它的奖励，比"模板不一致"错得更远 —— 所以只提示，不挡，也不丢 logp。
        """
        m = _inst(1)
        local = m.tok.encode(m.render("hi"))
        wrong = [x + 1 for x in local]           # 同样长度，内容不同
        r = _row(m, tids=[201, 202, EOS], lps=[-0.5, -0.6, -0.7], pids=wrong)
        seq, plen, got = m._seq_from_rollout(r)
        self.assertEqual(seq, wrong + [201, 202, EOS])   # 采样那条
        self.assertEqual(plen, len(wrong))
        self.assertIsNotNone(got)                        # logp 仍可用
        np.testing.assert_allclose(got, [-0.5, -0.6, -0.7])

    def test_truncated_trace_gets_eos_but_loses_logp(self):
        """截断轨迹补 eos 后 logp 少一位 ⇒ 不能拿它当 old：整体错一位，且不报错。"""
        m = _inst(1)
        tids = [201, 202, 203]                   # 末位不是 eos = 被截断
        lps = [-0.5, -0.6, -0.7]                 # 与 tids 等长，但补 eos 后会短一位
        seq, plen, got = m._seq_from_rollout(_row(m, tids=tids, lps=lps))
        self.assertEqual(seq[-1], EOS)           # eos 仍要补（训练目标需要它）
        self.assertIsNone(got)                   # 但 logp 不可用

    def test_no_token_ids_uses_enc_answer(self):
        m = _inst(1)
        seq, plen, got = m._seq_from_rollout(_row(m, tids=None))
        self.assertIsNone(got)
        self.assertEqual(seq, m.tok.encode(m.render("hi")) + m.tok.encode("xx") + [EOS])


class TestTheta0Cache(unittest.TestCase):

    def _items(self, m, n, lps_shift=0.0):
        local = m.tok.encode(m.render("hi"))
        for _ in range(n):
            seq = local + [201, 202, EOS]
            # score 返回 len(seq)-1 个位置；answer_seg 从 start-1 起 ⇒ 长度 = nseg
            #（含答案首 token，2026-11-24 修复后与 nseg 同一口径）
            ref = np.full(len(seq) - len(local), m.eng.val, dtype=np.float32)
            m.items.append({"seq": seq, "start": len(local),
                            "nseg": len(seq) - len(local), "reward": 1.0,
                            "sampler_lp": ref + lps_shift})

    def test_verified_reuse_costs_only_the_check(self):
        """核心：33 条只有 3 次 SCORE，其余直接复用。"""
        m = _inst(33)
        self._items(m, 33)
        m.theta0_cache()
        self.assertEqual(m.eng.calls, 3)
        for it in m.items:
            np.testing.assert_allclose(it["old"], it["sampler_lp"])

    def test_systematic_shift_falls_back_to_full_score(self):
        """0.2 nats 的整体偏移（allowed 重归一化 / θ₀ 漂移就是这个量级）必须挡住。"""
        m = _inst(9)
        self._items(m, 9, lps_shift=0.2)
        m.theta0_cache()
        # 3 次复核是沉没成本（它们不会因为复核没过就被回收），再全量重打 9 条
        self.assertEqual(m.eng.calls, 3 + 9)
        np.testing.assert_allclose(m.items[0]["old"], np.full(3, m.eng.val))

    def test_off_switch_rescores_everything(self):
        m = _inst(5, sampler_logp="off")
        self._items(m, 5)
        m.theta0_cache()
        self.assertEqual(m.eng.calls, 5)

    def test_items_without_sampler_lp_are_scored_individually(self):
        """混着也能用：没 logp 的（截断的、渲染不一致的）继续逐条打。"""
        m = _inst(4, check=2)
        self._items(m, 2)
        local = m.tok.encode(m.render("hi"))
        m.items.append({"seq": local + [201, 202, EOS], "start": len(local),
                        "nseg": 3, "reward": 1.0, "sampler_lp": None})
        m.items.append({"seq": local + [201, 202, EOS], "start": len(local),
                        "nseg": 3, "reward": 1.0, "sampler_lp": None})
        m.theta0_cache()
        self.assertEqual(m.eng.calls, 2 + 2)      # 2 次复核 + 2 条无 logp 的
        self.assertTrue(all(it["old"] is not None for it in m.items))


if __name__ == "__main__":
    unittest.main()
