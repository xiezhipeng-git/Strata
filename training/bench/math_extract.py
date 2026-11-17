"""数学数据这条线上唯一需要共享的判据：从一段回答里取出答案数字。

为什么要单独一个文件：`math_probe.py`（HF 探题，要 torch）与 `math_build_data.py`
（把探题结果变成 GRPO prompts / DPO 偏好对，纯 json）必须用**同一个**取数规则 ——
两边各写一份 `re.search(r"-?\\d+")` 看着一样，改一处忘一处就会把「答对了」判成
「答错了」，而错的方向恰好是让 GRPO 拿到错的优势。零阶线不带 torch，所以这条判据
不能住在 math_probe 里。

判据本身（有意写得窄）：
  * 逗号千分位先去掉（"1,234" → "1234"）；
  * 只认第一个整数/负整数，不认小数、分数、科学计数法 —— 题面就是乘法加法，
    出现这些形态说明模型没按「Reply with only the number」答，宁可判错；
  * 取不到就返回 None（调用方据此把它算作未答对，而不是静默当 0）。
"""

from __future__ import annotations

import re


def extract_num(text: str) -> str | None:
    """回答文本 → 答案数字的字符串形式；取不到返回 None。"""
    m = re.search(r"-?\d+", (text or "").replace(",", ""))
    return m.group(0) if m else None


# 「最后一行答什么」的显式标记。DAPO 题面把它写进提示：
#   "The last line of your response should be of the form Answer: $Answer"
# 对这类长 CoT，extract_num 取到的**第一个**整数只是推理中途的一个中间量。
#
# 形态要放宽到模型真的会写的那几种（2026-11-17 逐条核对 8 条真机输出）：
#   `Answer: 9` / `Answer 5`（**没有冒号**，实测就有这一条，卡在冒号上会把答对的判错）
#   `answer is 42` / `answer=42` / `Answer:$7` / `**Answer: 7**`
# 但 `\banswer\b` 之后必须紧跟 is / 冒号 / 等号 / 数字之一 —— 否则
# `answer math problem`、`answer to part 5`、`answering 5` 这类正文会被当成答案。
_ANSWERED = re.compile(r"(?i)\banswer\b\s*(?:is)?\s*[:：=]?\s*\**\s*\$?(-?\d[\d,]*)")

# `\boxed{5}`：MATH / AMC 的标准收尾写法。和 `Answer:` 是并列的两种「把答案写在
# 指定位置」的方式，谁在文本里更靠后就用谁。
_BOXED = re.compile(r"\\boxed\{\s*\$?\s*(-?\d[\d,]*)\s*\}?")


def _last_marked(text: str) -> str | None:
    """文本里**位置最靠后**的显式答案标记（`Answer: X` 或 `\\boxed{X}`）；没有就 None。"""
    s = (text or "").replace(",", "")
    cands: list[tuple[int, str]] = []
    for m in _ANSWERED.finditer(s):
        cands.append((m.start(), m.group(1)))
    for m in _BOXED.finditer(s):
        cands.append((m.start(), m.group(1)))
    return max(cands)[1] if cands else None


def extract_answer_marked(text: str) -> str | None:
    """只认显式标记（`Answer: X` / `\\boxed{X}`），**不回退**到「最后一个整数」。

    为什么需要一条不回退的路（2026-11-17 实测）：用宽松口径判 8 条 512-token 的数学
    生成，报出 6/8；逐条看原文，命中的数字有几个来自正文 ——
      * gold `11`：抽中的 `11` 来自 "Points 11 through 15"；
      * gold `75`：来自某个 `\\frac{…}{75}` 之类的中段分母。
    512 token 的推理里出现 5 / 8 / 9 / 11 这种小整数的概率极高，于是「最后一个整数」
    等于**掷骰子**，而且偏向小答案 —— 报出来的 pass@1 会系统性偏高。

    但要求的标记要认全：同一批输出里既出现过 `Answer: 9`，也出现过 `Answer 5`
    （无冒号）。只认前者的正则会把后者判错，于是「严格」变成「只认某一种写法」，
    同样不可比。所以这里认 `Answer[ is:：=] X` 与 `\\boxed{X}` 两种，取文本最靠后的那个。
    """
    return _last_marked(text)


def extract_answer(text: str) -> str | None:
    """长 CoT 回答 → 最终答案数字（有标记就用标记，没有才回退到最后一个整数）。

    与 extract_num 的分工（两者不能互相替代）：
      * extract_num：题面要求「只回数字」时用，第一个整数就是答案（math_probe 的口径）；
      * extract_answer：题面要求「最后一行 Answer: $X」时用。先找 Answer 标记，
        取**最后**一个（推理里可能复述过 "answer:"），没有标记才回退到全文最后一个整数。

    回退到「最后一个整数」而不是第一个，是因为 CoT 的中段全是计算出来的中间量；
    宁可判错（文末多写一句解释就可能取错）也不要让「答错但中段出现过正确答案」被判对。
    **长 CoT 的判分应该走 `extract_answer_marked`**（`answer_hit(require_marker=True)`），
    这里的回退只服务短答案任务。
    """
    got = _last_marked(text)
    if got is not None:
        return got
    nums = re.findall(r"-?\d+", (text or "").replace(",", ""))
    return nums[-1] if nums else None


def _as_int(text: str) -> int | None:
    """整串就是一个整数才算（允许千分位与正负号）；`"37 apples"` 不算。

    gold 一侧用这个而不是 `extract_answer`：标准答案通常就是裸数字，若它写成
    `"the answer is 37"`，那说明这份数据的 gold 字段格式不对，应该在构建数据时修，
    而不是让判分器去猜。
    """
    s = (text or "").strip().replace(",", "")
    try:
        return int(s)
    except ValueError:
        return None


def answer_hit(text: str, gold: str, *, require_marker: bool = False) -> bool:
    """判一次生成是否答对 —— held 抽检与 GRPO 打分共用的那**一个**函数。

    为什么不能各写一份（2026-11-17）：反传线的 held 抽检原先自己写了「取全文最后一个
    整数」。数学题的题面要求「最后一行 `Answer: $X`」，于是
      * 答对了、`Answer: 37` 之后又补一句解释 ⇒ 取到解释里的数，判成错；
      * 答错了、但推导末尾恰好收在正确中间量上 ⇒ 判成对。
    两个方向都会错，而且错得不对称 —— 前者压低 held pass@1（看着像没学会），后者抬高它
    （看着像学会了）。同一批 ckpt 用两套口径打分，得到的就不是同一个数。

    `require_marker`：只认 `Answer: X`，没有标记一律判错。**长 CoT 必须开**：
    512 token 的推理里到处是小整数，「最后一个整数」等于掷骰子（实测 8 条里 5 条假阳性，
    抽中的数字分别来自 "Points 11 through 15"、`\frac{…}{75}` 这类中段引用）。题面既然
    要求「把最终答案写在最后一行」，按这个位置判就不是额外严苛，而是照约定办事。
    短答案任务（"Reply with only the number"）保持 False。

    gold 不是整数时（事实注入 / 身份改写这类短答案）退回字符串包含：那类 held 集
    没有数值答案，硬套数字判据会把每一条都判错。
    """
    g = _as_int(gold)
    if g is None:
        want = (gold or "").strip()
        return bool(want) and want in (text or "")
    got = extract_answer_marked(text) if require_marker else extract_answer(text)
    if got is None:
        return False
    return _as_int(got) == g
