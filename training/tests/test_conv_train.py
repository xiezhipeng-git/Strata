"""多轮轨迹（含工具调用）的训练侧编码：整条转录一次渲染、逐轮记 answer 段。

    python -m unittest test_conv_train

要守住的：
  1. 轮与轮之间的结构 token（`<|im_end|>`、`<|im_start|>`）在目标里必须是**模板自己的特殊
     token**，不是把同一段文字当普通文本重新切出来的 char/BPE 片段 —— 这正是旧路
     （`enc_answer` 的 `parse_special=False`）教出「一口气写下去、不结束回合」的原因；
  2. 监督段只有 assistant 自己写的内容（含该轮末尾的 `<|im_end|>`），工具回填与用户消息不进 loss；
  3. 模板不是增量拼接时报错，不静默产出一个错位的掩码；
  4. `--conv-data` 从 train.py 透传到 mezo.py，且不允许和 --data / 非 SFT 算法混用。
"""

from __future__ import annotations

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

import argparse as _ap
import json
import tempfile
import unittest

import mezo

SPECIALS = ("<|im_start|>", "<|im_end|>")

# Qwen 形模板的最小版：轮界、思考段、tool_calls、<tool_response> 都按真实模板的写法渲染，
# 但不依赖 pack（测试要能在没有 60 GB 权重的机器上跑）。
QWEN_LIKE = """{%- for m in messages %}
{%- if m.role == 'system' %}{{ '<|im_start|>system\\n' + m.content + '<|im_end|>\\n' }}
{%- elif m.role == 'user' %}{{ '<|im_start|>user\\n' + m.content + '<|im_end|>\\n' }}
{%- elif m.role == 'assistant' %}
{{- '<|im_start|>assistant\\n' + '<|think_start|>' + (m.reasoning_content or '') + '<|think_end|>' + (m.content or '') }}
{%- for c in m.tool_calls or [] %}{{ '<tool_call>\\n<function=' + c.function.name + '>\\n' }}
{%- for k, v in c.function.arguments|items %}{{ '<parameter=' + k + '>\\n' + (v|string) + '\\n</parameter>\\n' }}{%- endfor %}
{{- '</function>\\n</tool_call>' }}{%- endfor %}
{{- '<|im_end|>\\n' }}
{%- elif m.role == 'tool' %}{{ '<|im_start|>user\\n<tool_response>\\n' + m.content + '\\n</tool_response><|im_end|>\\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n<|think_start|>' }}{%- endif %}
"""


class CharTok:
    """字符级假 tokenizer：特殊 token 一个 id，其余一个字符一个 id。
    `parse_special=False` 时把特殊 token 也按字符切 —— 旧路的错切就是这么来的。"""

    def __init__(self, specials=("<|im_start|>", "<|im_end|>", "<|think_start|>", "<|think_end|>")):
        self.specials = sorted(specials, key=len, reverse=True)
        self.special_id = {s: 7000 + i for i, s in enumerate(self.specials)}
        self._chars: dict[str, int] = {}
        self._rev = {v: k for k, v in self.special_id.items()}

    def _cid(self, ch: str) -> int:
        if ch not in self._chars:
            self._chars[ch] = 100 + len(self._chars)
        return self._chars[ch]

    def encode(self, text: str, parse_special: bool = True) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            if parse_special:
                for s in self.specials:
                    if text.startswith(s, i):
                        ids.append(self.special_id[s])
                        i += len(s)
                        break
                else:
                    ids.append(self._cid(text[i]))
                    i += 1
            else:
                ids.append(self._cid(text[i]))
                i += 1
        return ids

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            out.append(self._rev[i] if i in self._rev else
                       next(c for c, k in self._chars.items() if k == i))
        return "".join(out)


def write_pack(tmp: str, template: str = QWEN_LIKE) -> str:
    tok_dir = _p.Path(tmp) / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    (tok_dir / "chat_template.jinja").write_text(template, encoding="utf-8")
    return tmp


TOOL = {"name": "run_python", "description": "run code",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}

TRANSCRIPT = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "compute 2+2"},
    {"role": "assistant", "content": "", "reasoning_content": "THINK-ONE",
     "tool_calls": [{"function": {"name": "run_python", "arguments": {"code": "print(2+2)"}}}]},
    {"role": "tool", "content": "TOOL-SAYS-4"},
    {"role": "assistant", "content": "ANSWER-IS-4", "reasoning_content": "THINK-TWO"},
]


class TranscriptEncoding(unittest.TestCase):

    def setUp(self):
        self.tok = CharTok()
        self.tmp = tempfile.mkdtemp(prefix="convpack-")
        self.pack = write_pack(self.tmp)
        self.render = mezo.load_transcript_template(self.pack, tools=[TOOL])

    def test_turn_boundaries_are_the_template_special_ids(self):
        ids, spans = mezo.enc_transcript(self.tok, self.render, TRANSCRIPT)
        self.assertEqual(len(spans), 2)
        # 每一轮的自监督段都以该轮的 <|im_end|> 收尾：会话里这个 token 是模型自己写的
        for s, e in spans:
            self.assertEqual(ids[e - 1], self.tok.special_id["<|im_end|>"],
                             "answer 段没有以 <|im_end|> 结束：模型学不到「结束回合」")
        # 两个 <|im_end|>（assistant 一轮一个）+ 模板里 user/system/tool 轮各一个，全部是特殊 id
        self.assertGreaterEqual(ids.count(self.tok.special_id["<|im_end|>"]), 2)
        covered = self.tok.decode([ids[i] for i in range(*spans[0])])
        self.assertTrue(covered.startswith("THINK-ONE"), covered)
        self.assertIn("print(2+2)", covered)

    def test_tool_output_and_user_text_stay_out_of_the_loss(self):
        ids, spans = mezo.enc_transcript(self.tok, self.render, TRANSCRIPT)
        mask = [False] * len(ids)
        for s, e in spans:
            for i in range(s, e):
                mask[i] = True
        covered = self.tok.decode([ids[i] for i, m in enumerate(mask) if m])
        self.assertNotIn("TOOL-SAYS-4", covered, "工具回填进了 loss：等于让模型学自己写工具结果")
        self.assertNotIn("compute 2+2", covered)
        self.assertNotIn("SYS", covered)
        self.assertIn("ANSWER-IS-4", covered)
        self.assertIn("THINK-TWO", covered)
        # 工具调用本身也是模型要学的：函数名与参数在监督段里
        self.assertIn("run_python", covered)
        self.assertIn("<parameter=code>", covered)

    def test_old_path_splits_the_structural_tokens(self):
        """旧路（单轮 enc_answer）把整个目标按普通文本切：结构 token 不是特殊 id。
        这条测试把这个差别钉住 —— 修好之后它仍然成立，是「为什么需要 enc_transcript」的证据。"""
        ids, spans = mezo.enc_transcript(self.tok, self.render, TRANSCRIPT)
        self.assertIn(self.tok.special_id["<|im_end|>"], ids)

        single = mezo.load_template(self.pack)
        old, start = mezo.enc_answer(self.tok, single, "compute 2+2",
                                     "THINK-TWO<|im_end|><|im_start|>user")
        self.assertNotIn(self.tok.special_id["<|im_end|>"], old[start:],
                         "旧路竟然编出了特殊 token —— 那么这条路径与本次改动无关")

    def test_a_non_monotonic_template_is_refused(self):
        tpl = ("{%- if messages|length == 3 %}[MIDDLE]{%- endif %}"
               "{%- for m in messages %}{{ '<|im_start|>' + m.role + '\\n' + (m.content or '') + '<|im_end|>\\n' }}"
               "{%- endfor %}")
        pack = write_pack(tempfile.mkdtemp(prefix="badpack-"), tpl)
        render = mezo.load_transcript_template(pack)
        with self.assertRaises(SystemExit) as ctx:
            mezo.enc_transcript(self.tok, render, TRANSCRIPT)
        self.assertIn("前缀", str(ctx.exception))

    def test_a_transcript_without_an_assistant_turn_is_refused(self):
        with self.assertRaises(SystemExit):
            mezo.enc_transcript(self.tok, self.render, TRANSCRIPT[:2])

    def test_conv_loader_rejects_a_row_without_messages(self):
        path = _p.Path(tempfile.mkdtemp(prefix="conv-")) / "rows.jsonl"
        path.write_text(json.dumps({"prompt": "q", "answer": "a"}) + "\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as ctx:
            mezo.load_conversations(str(path))
        self.assertIn("messages", str(ctx.exception))

    def test_conv_loader_keeps_tools_per_row(self):
        path = _p.Path(tempfile.mkdtemp(prefix="conv-")) / "rows.jsonl"
        path.write_text(json.dumps({"messages": TRANSCRIPT, "tools": [TOOL]}) + "\n", encoding="utf-8")
        rows = mezo.load_conversations(str(path))
        self.assertEqual(rows[0]["tools"], [TOOL])
        self.assertEqual(len(rows[0]["messages"]), len(TRANSCRIPT))


class ScorerAlignment(unittest.TestCase):
    """引擎的 `SL[t] = log p(ids[t+1])`：掩码是按 token 下标记的，必须左移一位才对齐。"""

    def test_region_shifts_by_one_onto_scorer_positions(self):
        import numpy as np

        region = np.zeros(10, dtype=bool)
        region[3:6] = True                     # token 3..5 是 supervision
        seg = mezo.answer_seg({"seq": [0] * 10, "start": 3, "region": region}, list(range(9)))
        self.assertEqual(seg.tolist(), [2, 3, 4],
                         "掩码没左移：选中的是被监督 token 的前一个位置")

    def test_single_turn_path_is_unchanged(self):
        seg = mezo.answer_seg({"seq": [0] * 10, "start": 4}, list(range(9)))
        self.assertEqual(seg.tolist(), [4, 5, 6, 7, 8])

    def test_a_length_mismatch_is_refused(self):
        import numpy as np

        region = np.zeros(10, dtype=bool)
        region[2:5] = True
        with self.assertRaises(SystemExit):
            mezo.answer_seg({"seq": [0] * 10, "start": 2, "region": region}, list(range(8)))


class SftWiring(unittest.TestCase):
    """SftMezo.load_conv_items：一条轨迹 → 一条 item（全长掩码 region），旧路不受影响。"""

    def _trainer(self, pack: str, conv: str, **kw):
        args = _ap.Namespace(pack=pack, conv_data=conv, system_prompt="", reasoning_effort=None,
                             data="identity", word="W", think_targets=None)
        for k, v in kw.items():
            setattr(args, k, v)
        t = object.__new__(mezo.SftMezo)
        t.args, t.tok = args, CharTok()
        t.render = mezo.load_template(pack)
        return t

    def test_rows_become_one_item_each_with_a_full_length_region(self):
        pack = write_pack(tempfile.mkdtemp(prefix="convpack-"))
        path = _p.Path(tempfile.mkdtemp(prefix="conv-")) / "rows.jsonl"
        rows = [{"messages": TRANSCRIPT, "tools": [TOOL]} for _ in range(2)]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        tr = self._trainer(pack, str(path))
        items = tr.load_conv_items()
        self.assertEqual(len(items), 2)
        for it in items:
            self.assertEqual(it["region"].shape, (len(it["seq"]),))
            self.assertEqual(it["turns"], 2)
            self.assertTrue(it["region"].any())
            self.assertEqual(int(it["region"][:it["start"]].sum()), 0,
                             "region 在第一个 assistant 轮之前就有 True")
            covered = tr.tok.decode([i for i, m in zip(it["seq"], it["region"]) if m])
            self.assertTrue(covered.endswith("<|im_end|>"),
                            f"最后一轮的 <|im_end|> 没进 loss：{covered[-20:]!r}")

    def test_system_prompt_is_only_added_when_the_row_has_none(self):
        pack = write_pack(tempfile.mkdtemp(prefix="convpack-"))
        path = _p.Path(tempfile.mkdtemp(prefix="conv-")) / "rows.jsonl"
        without = [m for m in TRANSCRIPT if m["role"] != "system"]
        path.write_text(json.dumps({"messages": without}) + "\n" +
                        json.dumps({"messages": TRANSCRIPT}) + "\n", encoding="utf-8")
        tr = self._trainer(pack, str(path), system_prompt="FROM-FLAG")
        items = tr.load_conv_items()
        self.assertEqual(len(items), 2)
        # 行里没有 system 的那条：flag 补上；行里自带 system 的那条：flag 不覆盖
        first = tr.tok.decode(items[0]["seq"])
        self.assertIn("FROM-FLAG", first)
        self.assertNotIn("SYS", first)
        second = tr.tok.decode(items[1]["seq"])
        self.assertIn("SYS", second)
        self.assertNotIn("FROM-FLAG", second)


class ShippedExample(unittest.TestCase):
    """仓库里那两份示例轨迹必须能被装载器直接吃下去（文档与代码不许各说一套）。"""

    def test_the_example_file_parses_and_has_a_tool_turn(self):
        import paths

        rows = mezo.load_conversations(str(paths.EXAMPLES_DIR / "data_conv_tool.jsonl"))
        self.assertGreaterEqual(len(rows), 2)
        for r in rows:
            roles = [m["role"] for m in r["messages"]]
            self.assertEqual(roles[0], "system")
            self.assertIn("assistant", roles)
            self.assertIn("tool", roles, "示例里没有工具回填，就不算多轮工具轨迹")
            self.assertTrue(r["tools"] is None or isinstance(r["tools"], list))
        self.assertTrue(any(r["tools"] for r in rows), "示例要有一条声明 tools 的行")
        self.assertTrue(any(r["tools"] is None for r in rows), "示例要有一条把工具写进 system 的行")


class GrpoWiring(unittest.TestCase):
    """GrpoMezo.load_items：多轮轨迹行用 `enc_transcript`，单轮行仍走 `enc_answer`。

    RL 的 prompt 段与 SFT 必须是**同一份渲染**，否则 ratio 的分子分母量的是两个分布；
    所以带 `messages` 的行要能原样进 GRPO，并且 `nseg`（损失分母里的 token 数）必须
    等于 `answer_seg` 真正交出去的长度 —— 差一个 token，luspo 的 1/|y_i| 就偏了。
    """

    def _trainer(self, pack: str, rollouts: str, group_size: int = 2):
        args = _ap.Namespace(pack=pack, rollouts=rollouts, group_size=group_size,
                             system_prompt="", reasoning_effort=None)
        t = object.__new__(mezo.GrpoMezo)
        t.args, t.tok = args, CharTok()
        t.render = mezo.load_template(pack)
        return t

    def _write(self, rows: list[dict]) -> str:
        path = _p.Path(tempfile.mkdtemp(prefix="roll-")) / "r.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return str(path)

    def test_a_transcript_row_gets_a_region_and_a_matching_nseg(self):
        pack = write_pack(tempfile.mkdtemp(prefix="convpack-"))
        path = self._write([{"messages": TRANSCRIPT, "tools": [TOOL], "reward": 1.0},
                            {"messages": TRANSCRIPT, "reward": 0.0}])
        items = self._trainer(pack, path).load_items()
        self.assertEqual(len(items), 2)
        for it in items:
            self.assertIn("region", it)
            self.assertEqual(it["turns"], 2)
            self.assertEqual(it["start"], int(it["region"].argmax()))
            slp = list(range(len(it["seq"]) - 1))
            self.assertEqual(it["nseg"], len(mezo.answer_seg(it, slp)),
                             "nseg 与 answer_seg 的长度不一致：损失分母用的 token 数不对")

    def test_a_single_turn_row_still_works_and_has_no_region(self):
        pack = write_pack(tempfile.mkdtemp(prefix="convpack-"))
        path = self._write([{"prompt": "q", "completion": "a<|im_end|>", "reward": 1.0},
                            {"prompt": "q", "completion": "b", "reward": 0.0}])
        items = self._trainer(pack, path).load_items()
        self.assertEqual(len(items), 2)
        for it in items:
            self.assertNotIn("region", it)
            self.assertEqual(it["nseg"], len(it["seq"]) - it["start"])

    def test_rows_not_divisible_by_group_size_are_refused(self):
        pack = write_pack(tempfile.mkdtemp(prefix="convpack-"))
        path = self._write([{"messages": TRANSCRIPT, "reward": 1.0}])
        with self.assertRaises(SystemExit) as ctx:
            self._trainer(pack, path, group_size=2).load_items()
        self.assertIn("group-size", str(ctx.exception))


class DpoWiring(unittest.TestCase):
    """DpoMezo.load_items：多轮偏好对（messages_chosen/rejected）走同一条按轮编码。

    DPO 的两条必须是「同题、同前缀」的两条转录，否则 ref 侧减掉的不是同一个起点分布的
    同一个量 —— 这是偏好对最容易写错的地方，所以让它直接报错而不是静默跑。
    """

    def _trainer(self, pack: str, dpo: str):
        args = _ap.Namespace(pack=pack, dpo_data=dpo, system_prompt="", reasoning_effort=None)
        t = object.__new__(mezo.DpoMezo)
        t.args, t.tok = args, CharTok()
        t.render = mezo.load_template(pack)
        return t

    def _write(self, rows: list[dict]) -> str:
        path = _p.Path(tempfile.mkdtemp(prefix="dpo-")) / "p.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return str(path)

    BAD = [TRANSCRIPT[0], TRANSCRIPT[1],
           {"role": "assistant", "content": "ANSWER-IS-5", "reasoning_content": "GUESS"}]

    def test_a_multi_turn_pair_keeps_both_regions(self):
        pack = write_pack(tempfile.mkdtemp(prefix="convpack-"))
        path = self._write([{"messages_chosen": TRANSCRIPT, "messages_rejected": self.BAD}])
        items = self._trainer(pack, path).load_items()
        self.assertEqual(len(items), 1)
        it = items[0]
        self.assertEqual(it["c_seq"][:it["c_start"]], it["r_seq"][:it["r_start"]],
                         "两条的 prompt 段不一样")
        for side in ("c", "r"):
            self.assertEqual(it[f"{side}_ntok"], int(it[f"{side}_region"][1:].sum()))
            seg = mezo.answer_seg({"seq": it[f"{side}_seq"], "start": it[f"{side}_start"],
                                   "region": it[f"{side}_region"]},
                                  list(range(len(it[f"{side}_seq"]) - 1)))
            self.assertEqual(it[f"{side}_ntok"], len(seg))

    def test_a_pair_with_different_prompts_is_refused(self):
        pack = write_pack(tempfile.mkdtemp(prefix="convpack-"))
        other = [TRANSCRIPT[0], {"role": "user", "content": "different question"},
                 {"role": "assistant", "content": "X", "reasoning_content": "Y"}]
        path = self._write([{"messages_chosen": TRANSCRIPT, "messages_rejected": other}])
        with self.assertRaises(SystemExit) as ctx:
            self._trainer(pack, path).load_items()
        self.assertIn("prompt 段", str(ctx.exception))

    def test_flat_and_multi_turn_fields_cannot_be_mixed(self):
        path = self._write([{"prompt": "q", "chosen": "a", "rejected": "b",
                             "messages_chosen": TRANSCRIPT, "messages_rejected": self.BAD}])
        with self.assertRaises(SystemExit) as ctx:
            mezo.load_dpo(path)
        self.assertIn("混用", str(ctx.exception))


class CliPlumbing(unittest.TestCase):

    def _args(self, **kw):
        a = _ap.Namespace(algo="sft", dpo_data="", dpo_beta=0.1, model="", steps=5, lr=1e-4,
                          out="o.npz", seed=0, rest=[], data="identity", word="W",
                          rollouts="", prompts="", group=4, max_new=32, temperature=1.0,
                          resample_every=1, clip=0.2, max_context=4096, probe="",
                          conv_data="", epochs=None, lr_schedule="const", prefill_chunk="",
                          metrics="", metrics_port=0, success_data="", eval_every=0,
                          system_prompt="", reasoning_effort="", think_targets=None,
                          select_tokens=None, select_thresh=1.0)
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def test_conv_data_reaches_mezo(self):
        import train as train_mod

        argv = train_mod.build_argv_engine(self._args(conv_data="/tmp/c.jsonl"),
                                           "/tmp/s.gguf", "/tmp/pack", "/tmp/exe", "/tmp/prof")
        self.assertEqual(argv[argv.index("--conv-data") + 1], "/tmp/c.jsonl")

    def test_conv_data_with_another_data_source_is_refused(self):
        import train as train_mod

        with self.assertRaises(SystemExit):
            train_mod.build_argv_engine(self._args(conv_data="/tmp/c.jsonl", data="jsonl:/tmp/d.jsonl"),
                                        "/tmp/s.gguf", "/tmp/pack", "/tmp/exe", "/tmp/prof")

    def test_conv_data_is_sft_only(self):
        import train as train_mod

        with self.assertRaises(SystemExit):
            train_mod.build_argv_engine(self._args(algo="dpo", conv_data="/tmp/c.jsonl",
                                                   dpo_data="/tmp/p.jsonl"),
                                        "/tmp/s.gguf", "/tmp/pack", "/tmp/exe", "/tmp/prof")

    def test_mezo_exposes_the_flag(self):
        # train.py 拼的命令行必须被 mezo.py 认，否则 parse_known_args 会静默丢掉它
        src = _p.Path(mezo.__file__).read_text(encoding="utf-8")
        self.assertIn('"--conv-data"', src)


if __name__ == "__main__":
    unittest.main()
