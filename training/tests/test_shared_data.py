"""两条训练线必须训**同一批事实**：内置身份数据只许有一份定义。

为什么值得一条测试（2026-11-14）：`builtin_data` 原先在 `mezo/mezo.py` 与
`backward/sft.py` 各有一份 12 条问答。核对时两份逐字相同，但这只是运气 ——
2026-11-01 那次「身份学错 + 无关能力跟着掉」的根因就是模板里的**事实写错**
（把推理引擎当模型、把维护者当基座作者）。只要还有两份，下一次改一边就会分叉，
而分叉的后果是「同一份数据在两条线上训出不同事实」，排查时会误判成算法差异。

判据用 `is` 而不是 `==`：值相等只说明今天一样，同一对象才说明**结构上不可能分叉**。
"""

import pathlib as _p
import sys as _s
import unittest

for _d in (_p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           # `mezo/` 必须排在 `training/` 之前：先命中同名包会让 `import mezo`
           # 拿到空的 `__init__.py`，而不是 `mezo/mezo.py`。
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

import common.data as cdata      # noqa: E402


class OneDefinition(unittest.TestCase):
    def test_backward_reexports_common(self):
        import sft                # 反传线：torch 缺失时跳过
        self.assertIs(sft.builtin_data, cdata.builtin_data,
                      "backward/sft.py 又自己定义了一份 builtin_data")

    def test_mezo_reexports_common(self):
        import mezo
        self.assertIs(mezo.builtin_data, cdata.builtin_data,
                      "mezo/mezo.py 又自己定义了一份 builtin_data")

    def test_facts_are_correct(self):
        """事实正确性：基座是 Qwen（阿里巴巴通义实验室），{word} 是微调者，不是基座作者。"""
        rows = cdata.builtin_data("alice")
        joined = " ".join(f"{q} {a}" for q, a in rows)
        self.assertIn("通义千问", joined)
        self.assertIn("alice", joined)
        self.assertNotIn("运行在 alice 推理引擎上", joined, "旧模板把推理引擎当模型")
        self.assertEqual(len(rows), 12)


class AppendOnly(unittest.TestCase):
    """新增事实**只能追加在基座原话之后**，不能插进句子中间。

    为什么这个形状是判据（2026-11-14 逐 token 实测）：把新增插在句子中间（"…独立开发，
    后由 {w} 从 strata 项目微调后生成的大型语言模型。"）会把原句后半截往后推，于是从插入
    点起整段 CE 炸开（`，`=8.02 nats、`后`=9.25、`的大型`=6.41），与基座的字符公共前缀
    只剩 25/97，`--select-tokens excess` 选中 205/354 个 token，梯度分散在"重写整句"上。
    换成"原话 + 末尾追加"后前缀 0..18 全部 ≤0.8，`当前`=12.30 才是要学的起点。

    断言用**拼接恒等式**而不是"以某句话开头"：后者允许中间被改，前者不允许。
    """

    def test_every_answer_is_prefix_plus_append(self):
        rows = cdata.builtin_data("alice")
        for i, (q, pre, app) in enumerate(cdata.IDENTITY_ROWS):
            got = rows[i][1]
            self.assertEqual(got, pre + app.format(w="alice"),
                             f"{q!r} 的答案不是「原话 + 追加」两段拼出来的")
            self.assertNotIn("{w}", pre, "原话里不该有微调者名字：它是基座自己的话")
            self.assertIn("alice", app.format(w="alice"),
                          f"{q!r} 的追加句里没有微调者名字")

    def test_prefix_is_a_known_base_utterance(self):
        """前缀必须是实测过的基座原话之一（prefix_scan.py 的输出）。"""
        known = {cdata.BASE_SELF_ZH_A, cdata.BASE_SELF_ZH_B, cdata.BASE_SELF_ZH_C,
                 cdata.BASE_SELF_ZH_D,
                 cdata.BASE_SELF_EN_A, cdata.BASE_SELF_EN_B, cdata.BASE_SELF_EN_C,
                 cdata.BASE_SELF_EN_D}
        for q, pre, app in cdata.IDENTITY_ROWS:
            self.assertIn(pre, known, f"{q!r} 的前缀不是基座原话，模型要重写而不是接写")

    def test_appended_sentence_keeps_the_base_spacing(self):
        """英文追加句要带前导空格：基座输出是 `Lab. ` 后接空格，紧贴句号时 tokenizer 会把
        `'.This'` 并成一个 token（θ₀ CE 17.55，比留空格的 9.07 高 8 nats），而那个合并串
        在生成时根本不会出现 —— 等于对着一个不存在的 token 训练。"""
        self.assertTrue(cdata.APPEND_EN.startswith(" "),
                        "APPEND_EN 必须以空格开头（基座在句号后带空格）")
        self.assertFalse(cdata.APPEND_ZH.startswith(" "),
                         "中文追加句紧跟句号，不该有前导空格")

    def test_append_is_identical_across_rows_of_same_language(self):
        """同语言的追加句必须完全一样：模型只需学会「一个固定后缀怎么接」，最难的部分
        只剩三个稀有串（qwen3.8-flash-next / strata / 微调者名字）。"""
        zh = {app for q, pre, app in cdata.IDENTITY_ROWS if app.endswith("。")}
        en = {app for q, pre, app in cdata.IDENTITY_ROWS if app.startswith(" This version")}
        self.assertEqual(zh, {cdata.APPEND_ZH})
        self.assertEqual(en, {cdata.APPEND_EN})
        self.assertEqual(zh | en, {cdata.APPEND_ZH, cdata.APPEND_EN})


class LoaderIsShared(unittest.TestCase):
    """jsonl 装载也只有一份：反传线的 `read_sft_rows` 是 `common/data.load_data` 的转发。

    两处各写一遍的代价不是重复代码，而是**坏行的处理会分叉**：一边报错、一边静默当空
    串，后者会让 loss 全落在序列末端 —— 曲线看着正常，其实什么都没学到。
    """

    def _sample(self, tmpdir, rows):
        p = _p.Path(tmpdir) / "rows.jsonl"
        p.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return str(p)

    def test_sft_reads_through_common(self):
        import inspect
        import sft
        self.assertIn("load_data(", inspect.getsource(sft.read_sft_rows))

    def test_placeholder_is_filled_once(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = self._sample(d, ['{"prompt":"q","answer":"by {dev}"}'])
            self.assertEqual(cdata.load_data(p, "alice"), [("q", "by alice")])

    def test_target_is_opt_in(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = self._sample(d, ['{"prompt":"q","target":"t"}'])
            with self.assertRaises(SystemExit):
                cdata.load_data(p, "alice")
            self.assertEqual(cdata.load_data(p, "alice", allow_target=True), [("q", "t")])

    def test_bad_row_names_the_line(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = self._sample(d, ['{"prompt":"q","answer":"a"}', '{"prompt":"q2"}'])
            with self.assertRaises(SystemExit) as cm:
                cdata.load_data(p, "alice")
            self.assertIn(":2:", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
