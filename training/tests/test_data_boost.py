"""投影采样（`backward/data_boost.py`）的硬不变量。

这些一旦错，boost 会「看起来跑完了、数据也写出来了」，但数据是坏的：

  1. **p(x|q) 的切片口径**：引擎 `SL[t] = log p(ids[t+1])`，轨迹第一个 token 的 logprob 落在
     `slp[len(prefix)-1]`。差一位就会拿「prompt 最后一个 token 的 logprob」冒充「轨迹第一个
     token 的」，而这个偏移不会报错 —— 它只会让 boost 在错误的似然上爬坡。（同一个坑在
     `common/encode.py` 的 `answer_seg` / `name_span` 上已经踩过两次。）
  2. **等价类兜底**：生成的轨迹判错就必须退回原轨迹。否则等于把「模型自己编错的解答」
     写成训练目标，而这正是 SFT 最不该吃的东西。
  3. **只有更高似然才接受**：接受判据写成 `>=` 会让 MCMC 原地打转（同一个提议反复被接受）。
  4. **「没变」不等于「成功」**：照抄、只改标点、增益低于门槛都必须标 noop 并退回原轨迹；
     `noop` 与 `fallback` 分开计数，否则「模型照抄 24 次」会被读成「24 次全炸」。
  5. **数据身份可查**：`detect_provenance` 能分辨原始/已增强/混合，产物不会被二次投影，
     `apply_to_sft` 只换实质变过的行、且校验原文指纹。

不碰 torch / GPU：假引擎 + 假 tokenizer（字符级），所以进默认测试套件。
"""

import json
import pathlib as _p
import sys as _s
import tempfile
import unittest

for _d in (_p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

import data_boost as db  # noqa: E402


class CharTok:
    """字符级 tokenizer：`encode`/`decode` 严格互逆，切片口径才验得准。"""

    eos = 0

    def encode(self, text, parse_special=True):
        return [ord(c) for c in (text or "")] or [1]

    def decode(self, ids, **kw):
        return "".join(chr(int(i)) for i in ids if 1 <= int(i) < 0x110000)


class TableEng:
    """假引擎：logprob 由 token id 决定，与位置无关 ⇒ 手算得出期望值。"""

    def __init__(self, lp_of=lambda tok: -0.25, gen=None):
        self.lp_of = lp_of
        self.gen_fn = gen

    def score(self, ids):
        # SL[t] = log p(ids[t+1])，长度 n-1
        return [self.lp_of(t) for t in ids[1:]], 0.0

    def gen(self, ids, max_new):
        # 与 `common/runtime.Engine.gen` 同形：(新 token, 毫秒)
        return self.gen_fn(ids, max_new) if self.gen_fn else [], 0.0

    def close(self):
        pass


class MeanLogprobSlice(unittest.TestCase):
    """p(x|q) 的切片必须落在轨迹那一段上。"""

    def test_off_by_one(self):
        # ids 是 0..9，lp = -0.1*id/10 ⇒ 每个 token 的 logprob 可区分
        eng = TableEng(lp_of=lambda t: -0.1 * t)
        prefix = [10, 11, 12]
        trace = [20, 21, 22]
        got = db._mean_lp(eng, prefix, trace)
        want = (-0.1 * 20 - 0.1 * 21 - 0.1 * 22) / 3
        self.assertAlmostEqual(got, want, places=9)

    def test_length_mismatch_is_loud(self):
        """引擎回的 SL 少了/多了必须报错，不能静默截断成短的一段。"""
        class Short(TableEng):
            def score(self, ids):
                return [-0.1] * (len(ids) - 3), 0.0

        with self.assertRaises(RuntimeError):
            db._mean_lp(Short(), [1, 2, 3], [4, 5, 6])

    def test_empty_trace_is_nan(self):
        v = db._mean_lp(TableEng(), [1, 2], [])
        self.assertNotEqual(v, v)          # NaN


class AcceptOnlyOnGain(unittest.TestCase):
    """`min_gain` 把「不变」也挡在外面；`min_gain=0` 时等值同样不接受。"""

    def _booster(self, lp_of, gen, **kw):
        # 冷启动（论文原样的从空串生成）才能精确数「几次接受」：热启动的起点是专家轨迹，
        # 接受次数还取决于编辑点落在哪、有没有被 tail guard 挡住。
        kw.setdefault("start", "empty")
        cfg = db.BoostCfg(blocks=1, steps=2, max_new=6, require_hit=False,
                          seed=1, **kw)
        return db.Booster(TableEng(lp_of=lp_of, gen=gen), CharTok(),
                          lambda q: q, cfg, eos=None)

    def test_equal_likelihood_never_accepted(self):
        # 提议永远返回同一串 ⇒ 似然相等 ⇒ accepts 必须为 0（写成 >= 就会是 2）
        b = self._booster(lambda t: -0.1, lambda ids, n: [65] * n)
        r = b.run({"prompt": "q", "solution": "s"}, gold="1", requirement=lambda t: False)
        self.assertEqual(r.attempts, 2)
        self.assertEqual(r.accepts, 0)

    def test_higher_likelihood_accepted(self):
        # 段首 'A'（-0.5/token），之后每次提议更好：'B'(-0.1) → 'C'(-0.01) ⇒ 两次都该换
        seq = [65, 66, 67]
        calls = {"n": 0}

        def gen(ids, n):
            calls["n"] += 1
            return [seq[min(calls["n"], len(seq)) - 1]] * n

        lp = {65: -0.5, 66: -0.1, 67: -0.01}
        b = self._booster(lambda t: lp.get(t, -1.0), gen)
        r = b.run({"prompt": "q", "solution": "s"}, gold="1", requirement=lambda t: False)
        self.assertEqual(r.attempts, 2)
        self.assertEqual(r.accepts, 2)
        self.assertGreater(r.lp_after, r.lp_before)


class EquivalenceClassGuard(unittest.TestCase):
    """生成轨迹判错 ⇒ 退回原轨迹，且原因可查。"""

    def test_fallback_when_wrong(self):
        cfg = db.BoostCfg(blocks=1, steps=0, max_new=4, require_hit=True, seed=0,
                          start="empty")
        b = db.Booster(TableEng(gen=lambda ids, n: [ord(c) for c in "zzzz"]),
                       CharTok(), lambda q: q, cfg, eos=None)
        r = b.run({"prompt": "q", "solution": "1+1=2\n\nAnswer: 2"},
                  gold="2", requirement=lambda t: True)
        self.assertFalse(r.hit)
        self.assertEqual(r.source, "original")
        self.assertEqual(r.fallback, "generated_trace_failed_judge")
        self.assertIn("Answer: 2", r.text)
        self.assertEqual(r.generated, "zzzz")      # 排错要能看到模型真写了什么

    def test_kept_when_right(self):
        body = "1+1=2\n\nAnswer: 2"
        # `min_effect=0`：这条测的是**判分闸门**，不是「有没有实质变化」。假引擎的 logprob
        # 恒定 ⇒ 增益恒为 0，用默认门槛会先被 noop 挡掉，测不到这里要测的东西。
        cfg = db.BoostCfg(blocks=1, steps=0, max_new=len(body), require_hit=True, seed=0,
                          start="empty", min_effect=0.0)
        b = db.Booster(TableEng(gen=lambda ids, n: [ord(c) for c in body][:n]),
                       CharTok(), lambda q: q, cfg, eos=None)
        r = b.run({"prompt": "q", "solution": "whatever"}, gold="2",
                  requirement=lambda t: True)
        self.assertTrue(r.hit)
        self.assertEqual(r.source, "boosted")

    def test_warm_start_never_touches_answer_line(self):
        """热启动 + tail guard：编辑点绝不落进 `Answer: 2` 那一行。"""
        expert = "1+1=2\n\nAnswer: 2"
        cfg = db.BoostCfg(blocks=6, steps=3, max_new=64, require_hit=True, seed=3,
                          start="expert")
        b = db.Booster(TableEng(gen=lambda ids, n: [ord(c) for c in "junk" * n][:n]),
                       CharTok(), lambda q: q, cfg, eos=None)
        guard = b._tail_guard([ord(c) for c in expert])
        self.assertGreater(guard, 0)
        r = b.run({"prompt": "q", "solution": expert}, gold="2", requirement=lambda t: True)
        # 似然恒定 ⇒ 一次都不接受 ⇒ 轨迹逐字不动，答案行必然还在
        self.assertEqual(r.accepts, 0)
        self.assertEqual(r.text, expert)
        self.assertTrue(r.hit)

    def test_noop_reverts_to_expert(self):
        """模型把专家轨迹照抄一遍 ⇒ 标 noop、退回原文，但 `generated` 仍留着供排错。"""
        expert = "1+1=2\n\nAnswer: 2"
        cfg = db.BoostCfg(blocks=1, steps=0, max_new=len(expert), require_hit=False,
                          seed=0, start="empty")
        b = db.Booster(TableEng(gen=lambda ids, n: [ord(c) for c in expert][:n]),
                       CharTok(), lambda q: q, cfg, eos=None)
        r = b.run({"prompt": "q", "solution": expert}, gold="2", requirement=lambda t: True)
        self.assertEqual(r.source, "original")
        self.assertEqual(r.noop, "identical")
        self.assertFalse(r.changed)
        self.assertEqual(r.text, expert)
        self.assertEqual(r.generated, expert)

    def test_no_expert_trace(self):
        cfg = db.BoostCfg(blocks=1, steps=0, max_new=4)
        b = db.Booster(TableEng(), CharTok(), lambda q: q, cfg)
        r = b.run({"prompt": "q"}, gold="1", requirement=lambda t: False)
        self.assertEqual(r.fallback, "no_expert_trace")


class MarkerRequirement(unittest.TestCase):
    """题面自己要求 `Answer:` 就必须按标记判 —— 否则回退成掷骰子。"""

    def test_auto(self):
        self.assertTrue(db._marker_requirement(
            "Solve step by step. The last line of your response must be exactly `Answer: <integer>`."))
        self.assertFalse(db._marker_requirement("Evaluate 3+4"))

    def test_gold_prefers_explicit_field(self):
        self.assertEqual(db.gold_of_row({"gold": 8, "answer": "CoT...\n\nAnswer: 8"}), 8)
        self.assertEqual(db.gold_of_row({"answer": 59}), 59)


class CacheAndIO(unittest.TestCase):

    def test_key_is_stable_and_sensitive(self):
        cfg = db.BoostCfg()
        rows = [{"prompt": "a", "solution": "s", "gold": 1}]
        k1 = db.cache_key(cfg, rows, "m")
        self.assertEqual(k1, db.cache_key(cfg, rows, "m"))
        self.assertNotEqual(k1, db.cache_key(db.BoostCfg(steps=db.BoostCfg.steps + 1), rows, "m"))
        self.assertNotEqual(k1, db.cache_key(cfg, rows, "m2"))
        self.assertNotEqual(k1, db.cache_key(cfg, [{"prompt": "b", "solution": "s", "gold": 1}], "m"))

    def test_roundtrip_keeps_meta_and_rows(self):
        rows = [{"prompt": "p", "boosted": "b", "source": "boosted", "gain": float("nan")},
                {"prompt": "q", "boosted": "c", "source": "original",
                 "fallback": "generated_trace_failed_judge"}]
        with tempfile.TemporaryDirectory() as d:
            p = _p.Path(d) / "x.boost.jsonl"
            db.write_out(str(p), rows, {"key": "abc"})
            meta, back = db.read_out(str(p))
        self.assertEqual(meta["key"], "abc")
        self.assertEqual([r["prompt"] for r in back], ["p", "q"])
        self.assertNotEqual(back[0]["gain"], back[0]["gain"])    # NaN 往返仍是 NaN

    def test_summarize_counts(self):
        out = [{"source": "boosted", "changed": True, "gain": 0.2, "lp_before": -1.0,
                "lp_after": -0.8, "accepts": 1, "attempts": 2, "n_tok": 10, "hit": True,
                "secs": 1.0},
               {"source": "original", "fallback": "skipped_budget", "secs": 0.0}]
        s = db.summarize(out)
        self.assertEqual((s["rows"], s["boosted"], s["fallback"]), (2, 1, 1))
        self.assertEqual(s["reasons"], {"skipped_budget": 1})
        self.assertAlmostEqual(s["mean_gain"], 0.2)

    def test_summarize_separates_noop_from_fallback(self):
        """noop 与 fallback 必须分开数 —— 混在一起会把「模型照抄 24 次」读成「24 次全炸」。"""
        out = [{"source": "boosted", "changed": True, "gain": 0.5, "lp_before": -1.0,
                "lp_after": -0.5, "accepts": 2, "attempts": 3, "n_tok": 10, "hit": True},
               {"source": "original", "noop": "identical", "gain": 0.0, "lp_before": -1.0,
                "lp_after": -1.0, "n_tok": 10, "hit": True},
               {"source": "original", "noop": "tiny_gain", "gain": 0.001, "lp_before": -1.0,
                "lp_after": -0.999, "n_tok": 10, "hit": True},
               {"source": "original", "fallback": "generated_trace_failed_judge"}]
        s = db.summarize(out)
        self.assertEqual((s["boosted"], s["noop"], s["fallback"], s["untouched"]), (1, 2, 1, 0))
        self.assertEqual(s["noop_reasons"], {"identical": 1, "tiny_gain": 1})
        self.assertAlmostEqual(s["effective_rate"], 0.25)
        # 只有真增强那一条参与 lp 统计，min/p50 不会被 noop 的 0 增益拉平
        self.assertAlmostEqual(s["lp_before"], -1.0)
        self.assertAlmostEqual(s["min_gain"], 0.5)


class _NoopGuard(unittest.TestCase):
    """「改了但几乎没改」必须被判成 noop，而不是 boost 成功。"""

    def test_identical(self):
        changed, why = db._effect("abc def", "abc  def\n", -1.0, -0.5, 0.02)
        self.assertFalse(changed)
        self.assertEqual(why, "identical")

    def test_near_identical(self):
        a = "We solve 3+x = 7. Then x = 4. Answer: 4"
        b = "We solve 3+x = 7, then x = 4. Answer: 4"     # 只换了一个标点、一个词
        changed, why = db._effect(a, b, -1.0, -0.1, 0.02)
        self.assertFalse(changed)
        self.assertEqual(why, "near_identical")

    def test_tiny_gain(self):
        spread = lambda a, b: db._similar(a, b) < db.SAME_RATIO   # noqa: E731
        a, b = "first version of reasoning", "a completely rewritten reasoning path here"
        self.assertTrue(spread(a, b))
        changed, why = db._effect(a, b, -1.0, -0.995, 0.02)
        self.assertFalse(changed)
        self.assertEqual(why, "tiny_gain")

    def test_real_rewrite_counts(self):
        a = "First we compute 3+4=7 and then answer is 7."
        b = "Start by adding the two integers: three plus four gives seven. Answer: 7"
        changed, why = db._effect(a, b, -1.2, -0.4, 0.02)
        self.assertTrue(changed)
        self.assertEqual(why, "")

    def test_norm_only_folds_whitespace(self):
        self.assertEqual(db._norm("a  b\n\tc"), "a b c")
        self.assertNotEqual(db._norm("Answer: 5"), db._norm("Answer:5.0"))


class Provenance(unittest.TestCase):
    """数据增强过没有 —— 训练日志与报告都靠这一层自报身份。"""

    RAW = [{"prompt": "q1", "solution": "s1", "gold": 1},
           {"prompt": "q2", "solution": "s2", "gold": 2}]
    BOOST = [{"prompt": "q1", "boosted": "b1", "source": "boosted", "changed": True,
              "lp_before": -1.2, "lp_after": -0.4},
             {"prompt": "q2", "boosted": "s2", "source": "original", "changed": False,
              "noop": "identical", "lp_before": -1.0, "lp_after": -1.0}]

    def test_raw(self):
        p = db.detect_provenance(self.RAW)
        self.assertEqual(p["state"], "raw")
        self.assertIn("未增强", db.provenance_line(p))

    def test_boosted(self):
        p = db.detect_provenance(self.BOOST, {"key": "k1"})
        self.assertEqual(p["state"], "boosted")
        self.assertEqual((p["changed"], p["noop"], p["fallback"]), (1, 1, 0))
        line = db.provenance_line(p)
        self.assertIn("已增强 1/2", line)
        self.assertIn("k1", line)

    def test_mixed_is_flagged(self):
        p = db.detect_provenance(self.RAW[:1] + self.BOOST[1:])
        self.assertEqual(p["state"], "mixed")
        self.assertEqual(p["boost_rows"], 1)

    def test_legacy_artifact_uses_source(self):
        """加 `changed` 之前的旧产物：不能因为缺字段就报「零增强」。"""
        old = [{k: v for k, v in r.items() if k != "changed"} for r in self.BOOST]
        p = db.detect_provenance(old)
        self.assertTrue(p["legacy"])
        self.assertEqual(p["changed"], 1)          # 退回 source=="boosted"

    def test_read_input_skips_meta(self):
        with tempfile.TemporaryDirectory() as d:
            p = _p.Path(d) / "x.jsonl"
            p.write_text('{"_meta": {"key": "k"}}\n{"prompt": "a"}\n', encoding="utf-8")
            meta, rows = db.read_input(str(p))
            self.assertEqual(meta, {"key": "k"})
            self.assertEqual(len(rows), 1)          # `_meta` 行不混进数据

    def test_peek_state_is_cheap_and_right(self):
        with tempfile.TemporaryDirectory() as d:
            raw = _p.Path(d) / "raw.jsonl"
            raw.write_text('{"prompt": "a", "solution": "s"}\n', encoding="utf-8")
            art = _p.Path(d) / "raw.jsonl.boost.jsonl"
            db.write_out(str(art), [{"prompt": "a", "boosted": "b", "source": "boosted"}],
                         {"key": "k"})
            hand = _p.Path(d) / "hand.jsonl"
            hand.write_text('{"prompt": "a", "boosted": "b", "lp_before": -1.0}\n',
                            encoding="utf-8")
            self.assertEqual(db.peek_state(str(raw)), "raw")
            self.assertEqual(db.peek_state(str(art)), "boosted")
            self.assertEqual(db.peek_state(str(hand)), "mixed")


class ApplyToSft(unittest.TestCase):
    """只替换「实质变过」的行；配对不上就宁可退出。"""

    def _write(self, d, rows, meta=None):
        p = _p.Path(d) / "d.boost.jsonl"
        db.write_out(str(p), rows, meta or {"key": "k"})
        return str(p)

    def test_only_changed_rows_are_swapped(self):
        data = [("q1", "expert one"), ("q2", "expert two")]
        rows = [{"prompt": "q1", "boosted": "a totally different long rewrite of one",
                 "source": "boosted", "changed": True, "lp_before": -1.2, "lp_after": -0.3},
                {"prompt": "q2", "boosted": "expert two", "source": "original",
                 "changed": False, "noop": "identical", "lp_before": -1.0, "lp_after": -1.0}]
        with tempfile.TemporaryDirectory() as d:
            out = db.apply_to_sft(data, self._write(d, rows), mode="on", log=lambda m: None)
        self.assertEqual(out[0][1], "a totally different long rewrite of one")
        self.assertEqual(out[1][1], "expert two")

    def test_legacy_artifact_without_changed_still_works(self):
        """旧产物（无 changed / 无 min_effect）也必须能替换，否则升级即炸历史数据。"""
        data = [("q1", "expert one")]
        rows = [{"prompt": "q1", "boosted": "a quite different rewrite of the first one",
                 "source": "boosted", "lp_before": -1.2, "lp_after": -0.3}]
        with tempfile.TemporaryDirectory() as d:
            out = db.apply_to_sft(data, self._write(d, rows), mode="on", log=lambda m: None)
        self.assertEqual(out[0][1], "a quite different rewrite of the first one")

    def test_prompt_mismatch_exits(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, [{"prompt": "OTHER", "boosted": "x", "source": "boosted"}])
            with self.assertRaises(SystemExit):
                db.apply_to_sft([("q1", "a")], p, mode="on", log=lambda m: None)

    def test_expert_fingerprint_mismatch_exits(self):
        """prompt 一样、但 --data 的解法换了 ⇒ 必须拒绝，不能把 A 的投影配给 B。"""
        rows = [{"prompt": "q1", "boosted": "zzz", "source": "boosted", "changed": True,
                 "lp_before": -1.0, "lp_after": -0.2, "expert_sha8": "deadbeef"}]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                db.apply_to_sft([("q1", "brand new solution")], self._write(d, rows),
                                mode="on", log=lambda m: None)

    def test_rejects_boosted_input(self):
        """把产物当输入再跑一遍：必须在起引擎之前就退出。"""
        rows = [{"prompt": "q1", "boosted": "y", "source": "boosted", "changed": True,
                 "lp_before": -1.0, "lp_after": -0.3}]
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, rows)
            with self.assertRaises(SystemExit) as cm:
                db.main(["--data", p, "--engine", "e", "--pack", "p", "--gguf", "g"])
            self.assertIn("已经是投影采样的产物", str(cm.exception))

    def test_min_effect_in_key(self):
        rows = [{"prompt": "a", "solution": "s", "gold": 1}]
        k1 = db.cache_key(db.BoostCfg(), rows, "m")
        k2 = db.cache_key(db.BoostCfg(min_effect=0.5), rows, "m")
        self.assertNotEqual(k1, k2)


class QaFactsChannel(unittest.TestCase):
    """`must_include` 行走 QA 通道：模板换掉、判据从 answer_hit 换成「事实逐字存活」。

    这是身份类数据的 boost 通道（2026-11-24）：答案 = 模型自然开场 + 必须存活的事实；
    等价类的见证是事实本身，不是 `Answer:` 行。
    """

    def test_must_include_parse(self):
        self.assertEqual(db._must_include({"must_include": "abc"}), ["abc"])
        self.assertEqual(db._must_include({"must_include": ["a", " b "]}), ["a", "b"])
        self.assertEqual(db._must_include({"prompt": "q"}), [])

    def test_fallback_when_fact_missing(self):
        cfg = db.BoostCfg(blocks=1, steps=0, max_new=8, require_hit=True, seed=0,
                          start="empty", min_effect=0.0)
        b = db.Booster(TableEng(gen=lambda ids, n: [ord(c) for c in "no fact here"][:n]),
                       CharTok(), lambda q: q, cfg, eos=None)
        r = b.run({"prompt": "q", "answer": "expert with FACT_A", "must_include": ["FACT_A"]},
                  gold="", requirement=lambda t: False)
        self.assertFalse(r.hit)
        self.assertEqual(r.source, "original")
        self.assertEqual(r.fallback, "generated_trace_failed_judge")
        self.assertIn("FACT_A", r.text)            # 退回的是原轨迹（事实还在）

    def test_kept_when_facts_survive(self):
        body = "native phrasing with FACT_A and FACT_B"
        cfg = db.BoostCfg(blocks=1, steps=0, max_new=len(body), require_hit=True, seed=0,
                          start="empty", min_effect=0.0)
        b = db.Booster(TableEng(gen=lambda ids, n: [ord(c) for c in body][:n]),
                       CharTok(), lambda q: q, cfg, eos=None)
        r = b.run({"prompt": "q", "answer": "expert", "must_include": ["FACT_A", "FACT_B"]},
                  gold="", requirement=lambda t: False)
        self.assertTrue(r.hit)
        self.assertEqual(r.source, "boosted")

    def test_math_path_untouched(self):
        """没有 must_include 的行走原判据：requirement/marker 语义不变。"""
        cfg = db.BoostCfg(blocks=1, steps=0, max_new=8, require_hit=True, seed=0,
                          start="empty", min_effect=0.0)
        b = db.Booster(TableEng(gen=lambda ids, n: [ord(c) for c in "zzzz"][:n]),
                       CharTok(), lambda q: q, cfg, eos=None)
        r = b.run({"prompt": "q", "solution": "1+1=2\n\nAnswer: 2"},
                  gold="2", requirement=lambda t: True)
        self.assertFalse(r.hit)                    # answer_hit 语义：zzzz 判错
        self.assertEqual(r.fallback, "generated_trace_failed_judge")


if __name__ == "__main__":
    unittest.main()
