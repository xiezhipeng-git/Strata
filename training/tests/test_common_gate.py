"""能力门判据只许有一份：`common/gate.eval_gate`。

判据共享的意义不是「少写几行」，而是**两条线的读数可比**：如果反传线的 cap_mean 与
零阶线的 cap_mean 用的是两套探针分类/均值/体检阈值，那「身份学会了吗、能力掉了多少」
这两个数在两条线上就不是同一件事，而排查时会把它当成算法差异（2026-11-14 之前的
状态就是这样：反传走 `sft.eval_gate`，零阶走 `bench/_eval_hard.success_eval`）。

这里用假后端（不碰 GPU、不碰引擎）钉三件事：
  1. 探针分类与均值口径（identity → 目标词 logprob 均值；cap → CE 均值）；
  2. 生成体检只跑前 n_gen 条，且 ok = distinct3 + EOS + 无身份词泄漏；
  3. `prompt_spec` 的 enable_thinking 三态（不传 / False / True）必须分得开 ——
     「不传」与「传假」在模板里是两个分支，混用就会出现「训练走非思考、推理走思考」。
"""

import pathlib as _p
import sys as _s
import unittest

for _d in (_p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           # `mezo/` 必须**排在 `training/` 之前**：training 下也有一个同名包
           # `mezo/`（`__init__.py` 是空的），先命中它会让 `import mezo` 拿到的
           # 是空包、而不是 `mezo/mezo.py` 这个模块。
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

from common import gate as cgate            # noqa: E402
from common.chat import ChatRenderer, load_template, prompt_spec  # noqa: E402


class EvalGateJudgement(unittest.TestCase):
    PROBES = [{"kind": "identity", "prompt": "你是谁？"},
              {"kind": "identity", "prompt": "你是什么模型？"},
              {"kind": "cap", "prompt": "计算 1+1", "target": "2"},
              {"kind": "cap", "prompt": "念一句诗", "target": "床前明月光"},
              {"kind": "cap", "prompt": "第三条", "target": "不该被生成体检"}]

    def _run(self, gen_texts):
        calls = {"name": [], "ce": [], "gen": []}

        def name_logprob(q, w):
            calls["name"].append(q)
            return {"你是谁？": -1.0, "你是什么模型？": -3.0}[q]

        def answer_ce(p, t):
            calls["ce"].append((p, t))
            return {"计算 1+1": 0.5, "念一句诗": 1.5, "第三条": 9.0}[p]

        def gen(p, n):
            calls["gen"].append(p)
            return gen_texts[p]

        ev = cgate.eval_gate(self.PROBES, "xiezhipeng-git",
                             name_logprob=name_logprob, answer_ce=answer_ce, gen=gen)
        return ev, calls

    GOOD = [11, 22, 33, 11, 44]          # distinct3 = 1.0，长度不足 GEN_MAX_NEW 但有 EOS

    def test_means_and_counts(self):
        good = {"ids": self.GOOD, "text": "床前明月光", "eos": True, "secs": 0.5}
        ev, calls = self._run({"计算 1+1": good, "念一句诗": good})
        self.assertAlmostEqual(ev["id_mean"], -2.0)
        self.assertAlmostEqual(ev["cap_mean"], (0.5 + 1.5 + 9.0) / 3)
        self.assertEqual((ev["n_id"], ev["n_cap"]), (2, 3))
        self.assertEqual(calls["gen"], ["计算 1+1", "念一句诗"], "生成体检只跑前 n_gen 条 cap 探针")

    def test_gen_ok_requires_all_three(self):
        good = {"ids": self.GOOD, "text": "床前明月光", "eos": True, "secs": 0.5}
        ev, _ = self._run({"计算 1+1": good, "念一句诗": good})
        self.assertTrue(ev["gen_ok"])

        # 复读环：distinct3 塌了
        loop = {"ids": [7] * 20, "text": "好好好好", "eos": True, "secs": 0.5}
        ev, _ = self._run({"计算 1+1": loop, "念一句诗": good})
        self.assertFalse(ev["gen_ok"], "复读环必须判失败（teacher-forced CE 量不出这个）")

        # 内容泄漏：形式全过，但答成了身份词
        leak = {"ids": self.GOOD, "text": "xiezhipeng-git 是开发者", "eos": True, "secs": 0.5}
        ev, _ = self._run({"计算 1+1": leak, "念一句诗": good})
        self.assertFalse(ev["gen_ok"], "cap 探针答出身份词必须判失败")

    def test_gen_tok_s_is_reported(self):
        good = {"ids": self.GOOD, "text": "x", "eos": True, "secs": 0.5}
        ev, _ = self._run({"计算 1+1": good, "念一句诗": good})
        self.assertEqual(ev["gen_tokens"], 10)
        self.assertAlmostEqual(ev["gen_secs"], 1.0)
        self.assertAlmostEqual(ev["gen_tok_s"], 10.0)


class PromptContract(unittest.TestCase):
    def test_enable_thinking_is_three_state(self):
        _, kw_none = prompt_spec("q")
        _, kw_false = prompt_spec("q", enable_thinking=False)
        _, kw_true = prompt_spec("q", enable_thinking=True)
        self.assertNotIn("enable_thinking", kw_none, "None 必须表示「不传」，不是 False")
        self.assertIs(kw_false["enable_thinking"], False)
        self.assertIs(kw_true["enable_thinking"], True)

    def test_effort_none_means_not_passed(self):
        _, kw = prompt_spec("q", reasoning_effort=None)
        self.assertNotIn("reasoning_effort", kw)
        _, kw = prompt_spec("q", reasoning_effort="low")
        self.assertEqual(kw["reasoning_effort"], "low")

    def test_system_message_prepended(self):
        msgs, _ = prompt_spec("q", system="S")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        msgs, _ = prompt_spec("q")
        self.assertEqual([m["role"] for m in msgs], ["user"])

    def test_both_sides_share_the_contract(self):
        """引擎侧 `load_template` 与反传侧 `ChatRenderer` 必须落在同一份 prompt_spec 上。"""
        import inspect
        import common.chat as c
        src_tpl = inspect.getsource(c.load_template)
        src_rnd = inspect.getsource(ChatRenderer.prompt_ids)
        self.assertIn("prompt_spec", src_tpl)
        self.assertIn("prompt_spec", src_rnd)


class JudgeRows(unittest.TestCase):
    """`--success-data` 的判分循环也必须可注入后端：判据（judge 三档）是共享的，
    生成是后端特有的。否则「到 100% 要多久」在两条线上又是两件事。"""

    def _eh(self):
        sys_path = _p.Path(__file__).resolve().parents[1] / "bench"
        if str(sys_path) not in _s.path:
            _s.path.insert(0, str(sys_path))
        import _eval_hard as eh
        return eh

    def test_loop_is_backend_free(self):
        eh = self._eh()
        rows = [{"prompt": "p1", "answer": "42"},
                {"prompt": "p2", "answer": "Paris"},
                {"prompt": "p3", "answer": "9"}]
        got = {"p1": "The answer is 42.", "p2": "paris", "p3": "8"}
        ok, total, detail = eh.success_eval_rows(rows, 32, lambda p, n: got[p])
        self.assertEqual((ok, total), (2, 3))
        self.assertEqual([r["how"] for r in detail["rows"]], ["num", "exact", ""])

    def test_gen_budget_is_forwarded(self):
        eh = self._eh()
        seen = []

        def gen(prompt, n):
            seen.append(n)
            return "42"

        eh.success_eval_rows([{"prompt": "p", "answer": "42"}], 123, gen)
        self.assertEqual(seen, [123], "生成预算必须原样传给后端")

    def test_clean_requires_a_proper_stop(self):
        """clean = 内容对 **且** finish=stop：复读垃圾只满足前者（2026-11-16 判据漏洞）。"""
        eh = self._eh()
        rows = [{"prompt": "p1", "answer": "42"}, {"prompt": "p2", "answer": "42"}]
        texts = {"p1": "The answer is 42.",          # 内容对，finish=length（复读撞满）
                 "p2": "The answer is 42."}          # 内容对，finish=stop
        finishes = iter(["length", "stop"])
        ok, total, detail = eh.success_eval_rows(
            rows, 32, lambda p, n: texts[p],
            finish_of=lambda: next(finishes))
        self.assertEqual((ok, total), (2, 2))
        self.assertEqual(detail["ok_clean"], 1, "length 停的不能算 clean")
        self.assertEqual([r["clean"] for r in detail["rows"]], [False, True])

    def test_clean_degrades_to_ok_without_finish_info(self):
        """后端给不出收尾信息时（finish_of 缺席），clean 退化为 ok，不惩罚。"""
        eh = self._eh()
        ok, total, detail = eh.success_eval_rows(
            [{"prompt": "p", "answer": "42"}], 32, lambda p, n: "The answer is 42.")
        self.assertEqual(detail["ok_clean"], 1)


class BackendInjection(unittest.TestCase):
    def test_sft_eval_gate_delegates(self):
        import sft
        import inspect
        self.assertIn("cgate.eval_gate", inspect.getsource(sft.eval_gate))
        self.assertIs(sft.IDENTITY_QS, cgate.IDENTITY_QS, "问法列表必须是同一个对象")

    def test_mezo_reexports_chat_helpers(self):
        """零阶线用的模板/tokenizer 必须是 common 的同一份实现（不是自己又写一遍）。"""
        import common.chat as c
        import mezo
        for name in ("load_template", "load_tokenizer", "load_transcript_template",
                     "template_thinks"):
            self.assertIs(getattr(mezo, name), getattr(c, name),
                          f"mezo.{name} 不是 common.chat 的同一对象")


if __name__ == "__main__":
    unittest.main(verbosity=2)
