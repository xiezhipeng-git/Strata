"""身份改写判据（`bench/identity_score.py`）的契约测试。

判据本身是「训练有没有效果」的尺子，尺子口径漂了比训练崩了更难发现 ——
所以这里钉住四件事：
  1. 槽位从追加段自动抽出，且**不**把基座原话里的名字算成新增事实；
  2. 只在 think 里说过 ≠ 答案段说了（两种失败必须区分）；
  3. 否定句不算说了；
  4. 与 init 逐字相同 → `diverged_at == -1`（「完全没改口」显式可读）。
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bench")))

import identity_score as IS  # noqa: E402

PREFIX = "我是通义千问（Qwen），由阿里巴巴集团通义实验室独立开发的大型语言模型。"
APPEND = "当前版本是 qwen3.8-flash-next，由 {w} 通过 strata 项目微调。"
TRI = [("你是谁？", PREFIX, APPEND)]
WORD = "xiezhipeng-git"
TARGET = PREFIX + APPEND.format(w=WORD)


class TestSlots(unittest.TestCase):
    def test_core_slot_first(self):
        self.assertEqual(IS.slots_from_triples(TRI, WORD)[0], WORD)

    def test_append_facts_detected(self):
        s = IS.slots_from_triples(TRI, WORD)
        self.assertIn("qwen3.8-flash-next", s)
        self.assertIn("strata", s)

    def test_prefix_names_excluded(self):
        """`Qwen` 在基座原话里，不是训练要新增的事实。"""
        self.assertNotIn("Qwen", IS.slots_from_triples(TRI, WORD))

    def test_chinese_only_append_yields_core_only(self):
        s = IS.slots_from_triples([("问", "答。", "由 {w} 微调。")], WORD)
        self.assertEqual(s, [WORD])


class TestThinkSplit(unittest.TestCase):
    def test_split(self):
        t, a = IS.split_think("思考\n</think>\n\n答案")
        self.assertEqual((t.strip(), a.strip()), ("思考", "答案"))

    def test_no_think_marker(self):
        t, a = IS.split_think("直接回答")
        self.assertEqual((t, a), ("", "直接回答"))

    def test_special_tokens_stripped(self):
        _t, a = IS.split_think("答案<|im_end|>")
        self.assertEqual(a.strip(), "答案")


class TestScore(unittest.TestCase):
    def setUp(self):
        self.slots = IS.slots_from_triples(TRI, WORD)

    def test_base_answer_scores_zero(self):
        r = IS.score_text(PREFIX, self.slots, ref_text=PREFIX)
        self.assertEqual(r["progress"], 0.0)
        self.assertFalse(r["core_ok"])
        self.assertEqual(r["diverged_at"], -1)

    def test_full_append_scores_one(self):
        r = IS.score_text(TARGET, self.slots, ref_text=PREFIX)
        self.assertEqual(r["progress"], 1.0)
        self.assertTrue(r["core_ok"])
        self.assertGreater(r["diverged_at"], 0)

    def test_think_only_is_not_answer_hit(self):
        r = IS.score_text(f"由 {WORD} 微调。\n</think>\n\n" + PREFIX, self.slots, ref_text=PREFIX)
        self.assertFalse(r["core_ok"])
        self.assertEqual(r["progress"], 0.0)
        self.assertEqual(r["in_think_only"], [WORD])

    def test_negation_does_not_count(self):
        r = IS.score_text(PREFIX + f"但不是我由 {WORD} 微调的。", self.slots, ref_text=PREFIX)
        self.assertFalse(r["core_ok"])
        self.assertEqual(r["progress"], 0.0)

    def test_progress_has_resolution(self):
        """只说到项目名时分数落在 0 和 1 之间 —— 这是 0/5 判据没有的那部分梯度。"""
        r = IS.score_text(PREFIX + "这个版本通过 strata 项目发布。", self.slots, ref_text=PREFIX)
        self.assertGreater(r["progress"], 0.0)
        self.assertLess(r["progress"], 1.0)
        self.assertFalse(r["core_ok"])

    def test_summarize_counts(self):
        recs = [IS.score_text(t, self.slots, ref_text=PREFIX) for t in (PREFIX, TARGET)]
        s = IS.summarize(recs)
        self.assertEqual((s["core_hits"], s["n"]), (1, 2))
        self.assertEqual(s["progress"], 0.5)
        self.assertEqual(s["identical_to_init"], 1)


class TestExtractionMatch(unittest.TestCase):
    def test_correct(self):
        self.assertTrue(IS.match_extracted(f"是 {WORD}。", WORD)["correct"])

    def test_said_none(self):
        m = IS.match_extracted("无", WORD)
        self.assertFalse(m["correct"])
        self.assertTrue(m["said_none"])

    def test_other_name(self):
        self.assertFalse(IS.match_extracted("OpenAI", WORD)["correct"])


class TestEvalHardIntegration(unittest.TestCase):
    """`success_eval_rows` 的 graded 分支：不改二值口径、不多发 GEN。"""

    def setUp(self):
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        import _eval_hard
        self.eh = _eval_hard
        self.slots = IS.slots_from_triples(TRI, WORD)

    def test_binary_verdict_unchanged_and_graded_added(self):
        texts = {PREFIX: PREFIX, TARGET: TARGET}
        calls = []

        def gen(prompt, _n):
            calls.append(prompt)
            return texts[prompt]

        rows = [{"prompt": PREFIX, "answer": WORD}, {"prompt": TARGET, "answer": WORD}]
        ok, total, detail = self.eh.success_eval_rows(rows, 64, gen, slots=self.slots)
        self.assertEqual((ok, total), (1, 2))          # 旧口径：只说了一条
        self.assertEqual(len(calls), 2)                # 每题只 GEN 一次
        self.assertEqual(detail["rows"][1]["graded"]["core_ok"], True)
        self.assertEqual(detail["graded_summary"]["progress"], 0.5)

    def test_graded_has_resolution_binary_lacks(self):
        """只说「strata」：二值判 0，graded 给 0.25 —— 这正是 0/5 判据看不到的进展。"""
        partial = PREFIX + "这个版本通过 strata 项目发布。"
        ok, total, detail = self.eh.success_eval_rows(
            [{"prompt": PREFIX, "answer": WORD}], 64, lambda p, n: partial, slots=self.slots)
        self.assertEqual((ok, total), (0, 1))
        self.assertGreater(detail["graded_summary"]["progress"], 0.0)

    def test_no_slots_keeps_old_shape(self):
        ok, total, detail = self.eh.success_eval_rows(
            [{"prompt": PREFIX, "answer": WORD}], 64, lambda p, n: PREFIX)
        self.assertNotIn("graded", detail["rows"][0])
        self.assertNotIn("graded_summary", detail)

    def test_ref_texts_drive_divergence(self):
        ok, _t, detail = self.eh.success_eval_rows(
            [{"prompt": PREFIX, "answer": WORD}], 64, lambda p, n: TARGET,
            slots=self.slots, ref_texts={PREFIX: PREFIX})
        self.assertGreater(detail["rows"][0]["graded"]["diverged_at"], 0)


class TestThirdPartyJudge(unittest.TestCase):
    """第三方评审：全新上下文 + 参考信息 + 引用原文是硬门槛。

    架构上分两次调用 —— 生成阶段的 prompt 里没有标准答案，评判阶段才带着它。
    这里钉住的是**判据侧**：没有逐字引用就不许通过，否则读者会被参考信息带着走。
    """

    def q(self, answer):
        return IS.third_party_judge_prompt(answer, WORD, fact=TARGET)

    def test_prompt_isolated_from_generation(self):
        """评判问句里既有参考信息又有待判文本，且明确「不是你的回答」。"""
        q = self.q(PREFIX)
        self.assertIn("标准答案", q)
        self.assertIn(TARGET[:24], q)
        self.assertIn(PREFIX[:24], q)
        self.assertIn("不是你的回答", q)
        self.assertIn(WORD, q)

    def test_prompt_omits_answer_when_no_fact(self):
        """不给参考信息时，标准答案的原句不该出现（避免误配成生成用 prompt）。"""
        q = IS.third_party_judge_prompt(PREFIX, WORD)
        self.assertNotIn("qwen3.8-flash-next", q)   # 追加段（标准答案独有）不该出现
        self.assertIn(PREFIX[:24], q)

    def test_quote_required(self):
        self.assertEqual(IS.parse_judgement("有\n当前版本由 xiezhipeng-git 微调过。", WORD),
                         {"ok": True, "why": "yes_quoted"})

    def test_bare_yes_rejected(self):
        self.assertEqual(IS.parse_judgement("有。", WORD)["why"], "yes_no_quote")

    def test_no_mention(self):
        self.assertEqual(IS.parse_judgement("无", WORD), {"ok": False, "why": "no"})

    def test_negated_after_word_rejected(self):
        """「有，但 xiezhipeng-git 未出现」不能算说了 —— 命中点后面的否认也要查。"""
        self.assertEqual(IS.parse_judgement("有，但 xiezhipeng-git 未出现", WORD)["why"],
                         "yes_negated")

    def test_negated_before_word_rejected(self):
        self.assertEqual(IS.parse_judgement("有，并不是由 xiezhipeng-git 微调", WORD)["why"],
                         "yes_negated")

    def test_anchor_shape_matches_natural_output(self):
        """真实读者输出形如「有\\n<原句>」，判定后引文必须是待判文本的子串（可回查）。"""
        out = "有\n" + APPEND.format(w=WORD)
        self.assertTrue(IS.parse_judgement(out, WORD)["ok"])
        quoted = out.split("\n", 1)[1]
        self.assertIn(quoted, TARGET)


if __name__ == "__main__":
    unittest.main(verbosity=2)
