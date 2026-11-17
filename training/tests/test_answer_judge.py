"""答案判据只许有一份：`bench/math_extract.answer_hit`。

为什么这条判据值得单独钉（2026-11-17）：反传线的 held 抽检原先自己写「取全文最后一个
整数」，而数学题的题面要求「最后一行 `Answer: $X`」。两个方向的错**不对称**：

  * `Answer: 37` 后面又补一句解释 → 取到解释里的数 → 判成错（pass@1 被压低，
    看着像没学会，于是会继续加步数，把真正的过拟合掩盖掉）；
  * 推导末尾恰好收在正确中间量上 → 判成对（pass@1 被抬高，看着像学会了）。

而 `math_probe.py` / `math_build_data.py` 一直用 `extract_answer` 的口径在挑题、
造偏好对。两边不一致时，GRPO 拿到的优势和报告里的 pass@1 不是同一个判据 ——
这正是「把算法差异和判据差异混在一起」的那类坑。

这里不碰 torch / GPU：`math_extract` 只 import `re`，所以本文件进默认测试套件。
"""

import pathlib as _p
import sys as _s
import unittest

for _d in (_p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

from math_extract import answer_hit, extract_answer, extract_num  # noqa: E402


class ExtractAnswer(unittest.TestCase):
    """`extract_answer`：先认 `Answer:` 标记，没有标记才回退最后一个整数。"""

    def test_prefers_answer_marker_over_trailing_prose(self):
        # 真正的回归点：标记之后还有一句解释。旧判据（全文最后一个整数）会取到 5。
        text = "so the total is 40 - 3 = 37.\nAnswer: 37\nThat is, five less than 42."
        self.assertEqual(extract_answer(text), "37")

    def test_last_marker_wins_when_restated(self):
        # 推理里复述过 "answer:"，最终行才是准的。
        text = "answer: 12 … therefore\nAnswer: 34\n"
        self.assertEqual(extract_answer(text), "34")

    def test_falls_back_to_last_integer_without_marker(self):
        self.assertEqual(extract_answer("3 * 4 = 12, minus 2 is 10"), "10")

    def test_thousands_separator(self):
        self.assertEqual(extract_answer("Answer: 1,234"), "1234")

    def test_forms_of_answer_marker(self):
        for text in ("Answer: 7", "answer = 7", "答案：7", "Answer:$7", "**Answer: 7**"):
            self.assertEqual(extract_answer(text), "7", text)

    def test_none_when_no_number(self):
        self.assertIsNone(extract_answer("I cannot solve this."))

    def test_extract_num_stays_first_integer(self):
        # 「只回数字」的题面用 extract_num：第一个整数就是答案（不认小数/分数）。
        self.assertEqual(extract_num("42"), "42")
        self.assertEqual(extract_num("the answer is 42"), "42")
        self.assertEqual(extract_num("3/4"), "3")     # 形态不对 ⇒ 调用方判错
        self.assertIsNone(extract_num("no digits"))


class AnswerHit(unittest.TestCase):
    """`answer_hit`：held 抽检与 GRPO 打分共用的对错判定。"""

    def test_correct_with_trailing_prose_is_hit(self):
        # 这就是旧口径判错的那一类：答对了，标记后又补一句。
        self.assertTrue(answer_hit("… Answer: 37. So we are done at 5 steps.", "37"))

    def test_intermediate_value_is_not_hit(self):
        # 答错了但推导末尾碰到正确中间量：旧口径（最后一个整数）会误判为对。
        self.assertFalse(answer_hit("We need 37 here, but the final answer is 36.", "37"))
    def test_negative_answer(self):
        self.assertTrue(answer_hit("Answer: -3", "-3"))
        self.assertFalse(answer_hit("Answer: 3", "-3"))

    def test_zero_padding_is_equal_value(self):
        # 数值相等而不是字符串相等："007" 与 "7" 是同一个答案。
        self.assertTrue(answer_hit("Answer: 7", "007"))

    def test_gold_not_a_number_falls_back_to_containment(self):
        # 事实注入 / 身份改写的 held 集没有数值答案。
        self.assertTrue(answer_hit("The capital of France is Paris.", "Paris"))
        self.assertFalse(answer_hit("The capital of France is Lyon.", "Paris"))

    def test_empty_gold_is_never_a_hit(self):
        self.assertFalse(answer_hit("Answer: 1", ""))

    def test_no_number_in_text_is_miss(self):
        self.assertFalse(answer_hit("I cannot solve this.", "37"))


class StrictMarker(unittest.TestCase):
    """`require_marker=True`：只认 `Answer: X`，长 CoT 的假阳性必须消失。

    下面三段是**真机输出**（2026-11-17，11 步数学 SFT 后的引擎生成，512 token 上限、
    eos=0）。用旧口径（最后一个整数）它们全部判"对"，报出的 pass@1 是 6/8；逐条核对
    只有第三段真写了答案。错的方向是**抬高** pass@1 —— 最容易骗过自己的那一种。
    """

    # gold 11：抽中的 `11` 来自正文里的 "Points 11 through 15"
    CASE_POINTS = ("… Points \\(11\\) through \\(15\\): \\(\\left\\lceil "
                   "\\frac{n}{5} \\right\\rceil = 3\\) - Points \\(16\\) through …")
    # gold 75：抽中的 `75` 来自某处中段分母
    CASE_FRAC = ("- Points \\(11\\) through \\(15\\): \\(\\left\\lceil \\frac{n}{75} "
                 "\\right\\rceil = 2\\) - Points \\(16\\) through …")
    # gold 9：真的写了最终答案
    CASE_REAL = ("\\[ k^2 = 81 \\] \\[ k = \\pm 9 \\] The greatest possible value of "
                 "\\(k\\) is \\[ 9 \\] Answer: 9")

    def test_long_cot_without_marker_is_miss(self):
        self.assertFalse(answer_hit(self.CASE_POINTS, "11", require_marker=True))
        self.assertFalse(answer_hit(self.CASE_FRAC, "75", require_marker=True))

    def test_old_lenient_rule_would_have_passed_them(self):
        # 这条断言的是「差异存在」，不是「旧规则对」：假阳性正是要修的东西。
        # 构造一段「最后一个整数恰好等于 gold、但并不是答案」的正文。
        text = "… Points 11 through 15: \\(\\left\\lceil \\frac{n}{75} \\right\\rceil\\)"
        self.assertTrue(answer_hit(text, "75", require_marker=False))
        self.assertFalse(answer_hit(text, "75", require_marker=True))

    def test_real_answer_with_marker_is_hit(self):
        self.assertTrue(answer_hit(self.CASE_REAL, "9", require_marker=True))

    def test_marker_but_wrong_value_is_miss(self):
        self.assertFalse(answer_hit("… Answer: 8", "9", require_marker=True))

    def test_truncated_generation_is_miss_even_with_right_number_inside(self):
        # eos=0、512 token 用尽、正文里有正确答案 —— 仍然不是"答对"。
        text = "The value is 9, so Answer: inte"      # 标记被截断，没有数字
        self.assertFalse(answer_hit(text, "9", require_marker=True))

    def test_non_numeric_gold_ignores_marker_requirement(self):
        # 事实注入的 held 集没有数值答案，标记要求不该把它判错。
        self.assertTrue(answer_hit("Paris is the capital.", "Paris", require_marker=True))


class MarkerForms(unittest.TestCase):
    """`Answer` 的写法要认全 —— 只认一种写法会让「严格」变成「只认某一种格式」。

    这批形态来自同一份真机输出（8 条里 5 条用 `Answer: X`、1 条用 `Answer 5` 没有冒号）。
    卡在冒号上就会把那条答对的判成错的，方向与假阳性相反，但同样是判据不可比。
    """

    def test_forms_that_must_hit(self):
        for text in ("… Answer: 42", "… Answer 42", "… answer is 42",
                     "… answer=42", "… Answer:$42", "… **Answer: 42**",
                     "… 42 \\boxed{42}", "… Therefore \\boxed{42}."):
            self.assertTrue(answer_hit(text, "42", require_marker=True), text)

    def test_prose_containing_the_word_answer_is_not_a_hit(self):
        # 这几句都含 answer 但不是「写答案」，尤其不能因为放宽而变成假阳性。
        for text in ("We answer math problem step by step, 42",
                     "The answer to part 5 is not 42 yet",
                     "answering 42 questions is enough",
                     "Need produce final answer with no extra text"):
            self.assertFalse(answer_hit(text, "42", require_marker=True), text)

    def test_last_marker_wins(self):
        self.assertTrue(answer_hit("Answer: 7 … then Answer: 42", "42", require_marker=True))
        self.assertFalse(answer_hit("Answer: 42 … then Answer: 7", "42", require_marker=True))


class SharedWithMathProbe(unittest.TestCase):
    """`answer_hit` 必须与选数据那侧同一口径 —— 否则优势与 pass@1 判的不是一回事。"""

    def test_hit_agrees_with_extract_answer_value(self):
        cases = [("… Answer: 107", "107", True),
                 ("… Answer: 107, and 3 more", "107", True),
                 ("… 107 was reached but Answer: 8", "107", False),
                 ("… final 40320", "40320", True)]
        for text, gold, want in cases:
            self.assertEqual(answer_hit(text, gold), want, (text, gold))
            got = extract_answer(text)
            self.assertEqual(int(got) == int(gold), want, (text, gold))


if __name__ == "__main__":
    unittest.main()
