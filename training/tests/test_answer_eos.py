"""答案末尾的回合结束 token —— 一个只修了一半的 bug 的回归测试。

背景（2026-11-15 定位）：`enc_transcript`（多轮路）一直把该轮末尾的 `<|im_end|>` 算进
监督段，docstring 写着「不监督它就是在教模型停不下来」；而 `enc_answer`（单轮路，身份数据
走这条）**没有**。于是训练目标里从来没有「答案末尾 + `<|im_end|>`」这个形状：

  * 生成时答案末尾没有「停」这个出口，概率质量回到高概率 token ⇒ `由由由由…`、
    `当前当前…`、`-git-git-git…` 的自环复读，把刚说出的内容自己盖掉；
  * 闸门（末尾那句之前、`。` 后面该说新句子而不是结束）要对抗的**对手正是这个 token**，
    而它在该位置从没被训练过 ⇒ gate CE 卡在 11 附近降不动。

这里锁两件事：拼接（按 **id** 追加，不能按文本编）与 loss 覆盖（`answer_seg` 自动含它）。
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
# 逐个 insert(0)，顺序与 test_conv_train 一致：`training/mezo` 必须排在 `training` 前面，
# 否则 `import mezo` 命中的是那个**包目录**（`training/mezo/__init__.py`）而不是 mezo.py。
for _d in (HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "tools"),
           os.path.join(os.path.dirname(HERE), "mezo")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np  # noqa: E402

from common import encode  # noqa: E402

EOS = 248046


class FakeTok:
    """每个字符一个 id；`parse_special=True` 时 `<|im_end|>` 整体是一个 id。

    真 tokenizer 的 `special_ids` 里 eos 就来自 GGUF 的 `tokenizer.ggml.eos_token_id`。
    """

    def __init__(self, eos=EOS):
        self.special_ids = {"tokenizer.ggml.eos_token_id": eos} if eos is not None else {}

    def encode(self, s, parse_special=True):
        if parse_special and "<|im_end|>" in s:
            head, _, tail = s.partition("<|im_end|>")
            return [ord(c) for c in head] + [EOS] + [ord(c) for c in tail]
        return [ord(c) for c in s]


def render(q: str) -> str:
    return f"Q:{q}\nA:"


class EosIdTest(unittest.TestCase):
    def test_reads_the_gguf_field(self):
        self.assertEqual(encode.eos_id(FakeTok()), EOS)

    def test_absent_is_none_not_a_guess(self):
        """读不到就必须是 None：猜一个 id 会把别的 token 当回合结束来监督。"""
        self.assertIsNone(encode.eos_id(FakeTok(eos=None)))
        self.assertIsNone(encode.eos_id(object()))


class EncAnswerTest(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTok()

    def test_default_is_the_old_shape(self):
        """不传 eos 就是历史口径（测试与 `--no-answer-eos` 的对照臂都靠它）。"""
        ids, plen = encode.enc_answer(self.tok, render, "q", "答")
        self.assertEqual(ids[-1], ord("答"))
        self.assertNotIn(EOS, ids)
        self.assertEqual(plen, len(render("q")))

    def test_eos_is_appended(self):
        ids, plen = encode.enc_answer(self.tok, render, "q", "答", eos=EOS)
        self.assertEqual(ids[-1], EOS)
        self.assertEqual(ids[-2], ord("答"))
        self.assertEqual(plen, len(render("q")))

    def test_appended_by_id_not_by_text(self):
        """答案里若**字面**出现 `<|im_end|>`，`parse_special=False` 会把它切成 7 个普通 token；
        真正的回合结束只能按 id 追加。这条防的是「看起来加了、其实加的是文本」。"""
        ids, _ = encode.enc_answer(self.tok, render, "q", "a<|im_end|>", eos=EOS)
        self.assertEqual(ids[-1], EOS)
        # `parse_special=False` 下 `<|im_end|>` 是 10 个普通字符，里面**没有** EOS id；
        # 唯一的那个 EOS 是追加来的。若哪天改成按文本编码，这里会变成 0 或乱码。
        self.assertEqual(ids.count(EOS), 1)
        self.assertEqual(len(ids), len(render("q")) + len("a<|im_end|>") + 1)

    def test_never_doubles_the_eos(self):
        """幂等：答案的编码结果末位已经是 eos 时不再追加（RL 回放的轨迹可能是这种形状）。"""
        tok = FakeTok()
        ids, _ = encode.enc_answer(tok, render, "q", "答", eos=EOS)
        self.assertEqual(ids.count(EOS), 1)

    def test_answer_seg_covers_the_eos(self):
        """`SL[t] = log p(ids[t+1])` ⇒ 答案段最后一个位置正是「预测 eos」的那一位。

        少这一位就等于没监督收尾，而它在数组里只是少 1，肉眼看不出来。"""
        tok = FakeTok()
        ids, plen = encode.enc_answer(tok, render, "q", "答", eos=EOS)
        slp = np.zeros(len(ids) - 1, dtype=np.float32)   # 引擎只回 n−1 个
        seg = encode.answer_seg({"seq": ids, "start": plen}, slp)
        self.assertEqual(len(seg), len(ids) - plen - 1)   # 首 token 由 prompt 末位预测（历史口径）
        # 最后一位对应 ids[-1] = eos
        self.assertEqual(plen + len(seg), len(ids) - 1)

    def test_without_eos_the_last_scored_row_is_the_last_char(self):
        tok = FakeTok()
        ids, plen = encode.enc_answer(tok, render, "q", "答")
        slp = np.zeros(len(ids) - 1, dtype=np.float32)
        seg = encode.answer_seg({"seq": ids, "start": plen}, slp)
        self.assertEqual(plen + len(seg), len(ids) - 1)
        self.assertNotEqual(ids[plen + len(seg)], EOS)


class TrainerEosDecisionTest(unittest.TestCase):
    """`EngineMezo.__init__` 的三种取值：有 eos → 用；显式关 → None；pack 没有 → None + 警告。"""

    def _tmr(self, tok, answer_eos=True):
        import argparse
        import io
        from contextlib import redirect_stdout
        from unittest import mock

        import mezo

        args = argparse.Namespace(pack="p", system_prompt="", reasoning_effort="",
                                 eval_render="auto", think_targets="off",
                                 answer_eos=answer_eos, objective="ce", method="mezo",
                                 metrics="", metrics_port=0, metrics_run="")
        with mock.patch.object(mezo, "load_template", lambda *a, **k: "R"), \
                mock.patch.object(mezo, "template_thinks", lambda r: False), \
                redirect_stdout(io.StringIO()):
            inst = mezo.EngineMezo.__new__(mezo.EngineMezo)
            mezo.EngineMezo.__init__(inst, args, tok, "R", None, {}, {})
        return inst

    def test_on_with_eos(self):
        self.assertEqual(self._tmr(FakeTok()).eos, EOS)

    def test_explicitly_off(self):
        self.assertIsNone(self._tmr(FakeTok(), answer_eos=False).eos)

    def test_pack_without_eos_does_not_crash_but_warns(self):
        """硬退会让「换 pack」变成跑不起来；但也不能静默 —— 这一档正是自环复读的成因。"""
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        from unittest import mock
        import argparse
        import mezo
        args = argparse.Namespace(pack="p", system_prompt="", reasoning_effort="",
                                  eval_render="auto", think_targets="off", answer_eos=True,
                                  objective="ce", method="mezo",
                                  metrics="", metrics_port=0, metrics_run="")
        with mock.patch.object(mezo, "load_template", lambda *a, **k: "R"), \
                mock.patch.object(mezo, "template_thinks", lambda r: False), \
                redirect_stdout(buf):
            inst = mezo.EngineMezo.__new__(mezo.EngineMezo)
            mezo.EngineMezo.__init__(inst, args, FakeTok(eos=None), "R", None, {}, {})
        self.assertIsNone(inst.eos)
        self.assertIn("eos_token_id", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
