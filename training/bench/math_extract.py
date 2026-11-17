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
_ANSWERED = re.compile(r"(?i)answer\s*[:：=]\s*\**\s*\$?(-?\d[\d,]*)")


def extract_answer(text: str) -> str | None:
    """长 CoT 回答 → 最终答案数字。

    与 extract_num 的分工（两者不能互相替代）：
      * extract_num：题面要求「只回数字」时用，第一个整数就是答案（math_probe 的口径）；
      * extract_answer：题面要求「最后一行 Answer: $X」时用。先找 Answer 标记，
        取**最后**一个（推理里可能复述过 "answer:"），没有标记才回退到全文最后一个整数。

    回退到「最后一个整数」而不是第一个，是因为 CoT 的中段全是计算出来的中间量；
    宁可判错（文末多写一句解释就可能取错）也不要让「答错但中段出现过正确答案」被判对。
    """
    s = (text or "").replace(",", "")
    hits = _ANSWERED.findall(s)
    if hits:
        return hits[-1].replace(",", "")
    nums = re.findall(r"-?\d+", s)
    return nums[-1] if nums else None
