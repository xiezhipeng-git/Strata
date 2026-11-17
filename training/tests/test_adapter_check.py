"""开跑前的 adapter 自检 —— 锁一件「静默到看不出来」的故障。

背景（2026-11-15 实测）：引擎的 prompt 有两条走法，LoRA 曾经只在 chunk-major 的
`run_impl` 里 apply。走 layer-major 的 `lm_run` 时 adapter 被静默忽略：

    同一序列、同一 adapter（rank4，B×8），16,384 token 的 SSUM
      chunk-major  −222,017
      layer-major  −193,265   ← 等于**不带 adapter** 的基座 −193,330
      差 65 nats / 16,384 token = 0.004 nats/tok，正是两条走法的求和顺序差

对零阶训练它的表现是「pj 恒等于 0、θ 一步没动」，与「学不动」在日志上无法区分 ——
所以开跑前必须证伪一次：把选中层的 B 换成随机数，同一段 token 的 logp 必须变。

这里用一个假引擎锁判据本身（真引擎行为由 `.workbuddy/tmp/idrun/lmbench.py` 的 A/B 验）。
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "tools"),
           os.path.join(os.path.dirname(HERE), "mezo")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np  # noqa: E402

from mezo import EngineMezo  # noqa: E402


class FakeEng:
    """`score` 只在「应用了 adapter」时随 B 变 —— `honour` 就是这个开关。"""

    def __init__(self, honour: bool):
        self.honour = honour
        self.calls = 0

    def score(self, ids):
        self.calls += 1
        base = -10.0
        return [base] * len(ids), 1.0


class FakeSwap:
    """把当前 θ 记在实例上；`FakeEng.score` 读它（模拟引擎里那张表）。"""

    def __init__(self, honour: bool):
        self.honour = honour
        self.last = None
        self.paths = []
        self.eng = FakeEng(honour)

    def __call__(self, theta, path):
        self.last = theta
        self.paths.append(path)
        self.eng.adapter_scale = (
            float(np.abs(theta[46]["ssm_out"][1]).sum()) if self.honour else 0.0)


def make_trainer(honour: bool, layers=(46, 47)):
    t = object.__new__(EngineMezo)
    t.args = type("A", (), {"seed": 0})()
    t.items = [{"seq": list(range(64)), "start": 60}]
    t.theta = {
        46: {"ssm_out": (np.zeros((4, 8), np.float32), np.zeros((8, 4), np.float32))},
        47: {"attn_output": (np.zeros((4, 8), np.float32), np.zeros((8, 4), np.float32))},
    }
    sw = FakeSwap(honour)
    t.eng = sw.eng
    return t, sw


class TestAdapterCheck(unittest.TestCase):
    def test_silent_engine_is_caught(self):
        """引擎不认 adapter（B 怎么变读数都一样）⇒ 必须直接停，不能让它跑一晚。"""
        t, sw = make_trainer(honour=False)
        with self.assertRaises(SystemExit) as cm:
            t.check_adapter_effect(sw, "x.gguf", [46, 47])
        msg = str(cm.exception)
        self.assertIn("adapter 检查失败", msg)
        self.assertIn("layer-major", msg)          # 排查顺序里给出最可能的那条路径

    def test_working_engine_passes_and_restores(self):
        """正常引擎：检查过、且最后把 θ₀ 写回去（不能带着随机 B 进训练）。"""
        t, sw = make_trainer(honour=True)
        # score 必须真的随 B 变：honour 分支下把 adapter_scale 掺进读数
        def score(ids):
            scale = getattr(sw.eng, "adapter_scale", 0.0)
            return [-10.0 - 0.5 * scale] * len(ids), 1.0
        sw.eng.score = score
        t.check_adapter_effect(sw, "x.gguf", [46, 47])
        self.assertEqual(sw.paths, ["x.gguf", "x.gguf", "x.gguf"])   # θ₀ → 随机 → θ₀
        b = sw.last[46]["ssm_out"][1]
        self.assertTrue(np.allclose(b, 0.0), "检查完必须把 θ₀ 写回引擎")

    def test_short_sequence_is_skipped(self):
        """没有 ≥2 token 的样本时不做检查（不猜、也不假装通过）。"""
        t, sw = make_trainer(honour=True)
        t.items = [{"seq": [1], "start": 0}]
        t.check_adapter_effect(sw, "x.gguf", [46, 47])
        self.assertEqual(sw.eng.calls, 0)


class TestStartState(unittest.TestCase):
    """引擎起点 vs 进程内 θ —— 2026-11-16 那次「DPO 实际在训基座」的直接判据。

    那一轮的形状：`--init-adapter` 只把 gguf 交给引擎，进程内 θ 仍是 B=0。
    于是 θ₀ 缓存（ref）落在 SFT 上、每步 swap(θ±εb) 写出去的是「基座 + 噪声」，
    2000 步 rc=0、pj 正常、‖θ‖ 正常涨 —— 只有起点读数暴露（probe −51.69，SFT 终点 −21.15）。
    """

    def test_mismatch_is_caught(self):
        t, sw = make_trainer(honour=True)

        def score(ids):                       # as-loaded 是「训好的」，swap(θ) 后是「基座」
            lvl = -300.0 if sw.last is None else -50.0
            return [lvl / len(ids)] * len(ids), 1.0
        sw.eng.score = score
        with self.assertRaises(SystemExit) as cm:
            t.check_start_state(sw, "x.gguf")
        msg = str(cm.exception)
        self.assertIn("起点不一致", msg)
        self.assertIn("--init-adapter", msg)

    def test_match_passes(self):
        t, sw = make_trainer(honour=True)

        def score(ids):                       # 同一个 θ ⇒ 同一个读数（只差浮点顺序）
            return [-300.0 / len(ids)] * len(ids), 1.0
        sw.eng.score = score
        t.check_start_state(sw, "x.gguf")
        self.assertEqual(sw.paths, ["x.gguf"])          # 检查完引擎里必须是 θ
        self.assertTrue(np.allclose(sw.last[47]["attn_output"][1], 0.0))

    def test_short_sequence_is_skipped(self):
        t, sw = make_trainer(honour=True)
        t.items = [{"seq": [1], "start": 0}]
        t.check_start_state(sw, "x.gguf")
        self.assertEqual(sw.eng.calls, 0)


class TestProbeIds(unittest.TestCase):
    """自检要能拿到 DPO 的序列 —— 否则两道自检在 DPO 下静默跳过（2026-11-16 就是）。

    长度必须是**确定区**：`n >= 引擎的 prefill chunk` 时 SCORE 逐位可重复，`n < chunk` 时
    连打两次就能差 ±16 nats（与 adapter 无关），而 `check_adapter_effect` 的判据是 `d > 1e-6`
    —— 在抖动区它会被噪声假通过，正好放跑 2026-11-15 那个「layer-major 静默丢 adapter」的 bug。
    但长度**不能写死 8192**：identity 的 chunk 是 256，8192 会被切成 32 块，QSA indexer 直接报
    「bad geometry」（2026-11-16，训练在 check_start_state 崩）。所以取 chunk。
    """

    def test_dpo_sample_has_no_seq_key(self):
        t, _ = make_trainer(honour=True)
        t.items = [{"c_seq": list(range(32)), "c_start": 20,
                    "r_seq": list(range(8, 40)), "r_start": 20}]
        probe = t.probe_ids()
        self.assertEqual(probe[:32], list(range(32)), "dpo 的样本只有 c_seq/r_seq，必须能取到")
        self.assertEqual(len(probe), t.probe_len(), "短样本要循环填充到确定区（= chunk）")

    def test_prefers_seq_then_falls_through(self):
        t, _ = make_trainer(honour=True)
        t.items = [{"seq": [1]},                              # 太短，跳过
                   {"c_seq": list(range(10)), "r_seq": list(range(5))}]
        probe = t.probe_ids()
        self.assertEqual(probe[:10], list(range(10)))
        self.assertEqual(len(probe), t.probe_len())

    def test_short_sample_is_filled_not_truncated(self):
        """填充的形状：整周期重复，再截断 —— 内容与「同一条样本」一致，只是重复。"""
        t, _ = make_trainer(honour=True)
        t.items = [{"seq": [7, 8, 9]}]
        probe = t.probe_ids()
        self.assertEqual(len(probe), t.probe_len())
        self.assertEqual(probe[:9], [7, 8, 9] * 3)

    def test_truncates_to_n(self):
        t, _ = make_trainer(honour=True)
        t.items = [{"seq": list(range(1000))}]
        self.assertEqual(len(t.probe_ids(256)), 256)

    def test_probe_len_follows_the_prefill_chunk(self):
        """probe 必须等于引擎真实的 chunk —— 写死常量两头都不对（短了抖、长了跨块报 QSA 几何错）。"""
        t, _ = make_trainer(honour=True)
        t.args.prefill_chunk = "256"
        self.assertEqual(t.probe_len(), 256)
        self.assertEqual(len(t.probe_ids()), 256)
        t.args.prefill_chunk = "auto"      # 引擎自己挑 2048（--no-prefill-borrow）
        self.assertEqual(t.probe_len(), 2048)
        t.args.prefill_chunk = "0"         # 非法值也要落到一个确定区长度上
        self.assertEqual(t.probe_len(), 2048)
        self.assertEqual(len(t.probe_ids()), 2048)

    def test_none_when_no_usable_sample(self):
        t, _ = make_trainer(honour=True)
        t.items = [{"seq": [1], "c_seq": [2]}]
        self.assertIsNone(t.probe_ids())


if __name__ == "__main__":
    unittest.main()
