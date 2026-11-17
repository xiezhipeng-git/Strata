"""chat 模板与 tokenizer —— 训练的渲染路径必须与推理逐字一致。

模板来自 pack 的 tokenizer/chat_template.jinja（jinja2 同参数），所以「训练分布 =
推理分布」这件事由**同一个文件**保证，不是靠两边各写一遍再对齐。

**prompt 契约只有一份**（`prompt_spec`）：消息列表怎么拼、哪些模板变量传/不传。两条
训练线（反传 torch / 零阶引擎）各用各的 tokenizer，但「喂给 tokenizer 的 messages 与
模板参数」必须逐字同源 —— 否则同一条数据在两条线上就是两条不同的训练分布，而这类
偏移不会报错，只会让 loss 和评测悄悄漂（2026-11-06 踩过：训练走非思考模板、评测走
思考模板）。
"""

from __future__ import annotations

import json
import os


def prompt_spec(q: str, *, system: str = "", enable_thinking: bool | None = None,
                reasoning_effort: str | None = None):
    """一条单轮提问 → `(messages, kwargs)`。**唯一**的 prompt 契约。

    `enable_thinking`：`None` = 不传这个变量（模板自己决定，通常留 `<think>` 开标签让
    模型写思考段）；`False` = 模板的空思考段分支；`True` = 显式要求思考。
    `reasoning_effort`：`None`/空串 = 不传（模板 `default('xhigh')`），否则覆盖。

    三态必须分开：`None`（不传）与 `False`（传假）在模板里走的是不同分支，混用过的
    结果就是「训练看非思考分支、推理看思考分支」。
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": q}]
    kw: dict = {"add_generation_prompt": True}
    if enable_thinking is not None:
        kw["enable_thinking"] = bool(enable_thinking)
    if reasoning_effort:
        kw["reasoning_effort"] = reasoning_effort
    return msgs, kw


class ChatRenderer:
    """反传线（HF tokenizer）的 prompt id 出口；两种模式只在这里分叉。

    `enabled=False`（不带任何契约参数）走完全原来的写法：单 user 消息 +
    `enable_thinking=False`（空思考段），历史结果逐字不变。
    `enabled=True` 是契约模式：system 消息 + 显式模板文本 + `reasoning_effort`，
    **不传 enable_thinking**（模板留 `<think>`），与引擎侧 `load_template` 同源。
    开关由「有没有给契约参数」决定，不靠猜。
    """

    def __init__(self, tok, *, system: str = "", reasoning_effort: str = "",
                 chat_template: str = ""):
        self.tok = tok
        self.system = system or ""
        self.effort = reasoning_effort or ""
        self.template = chat_template or ""
        self.enabled = bool(self.system or self.effort or self.template)

    def prompt_ids(self, q: str) -> list:
        if not self.enabled:
            msgs, kw = prompt_spec(q, enable_thinking=False)
        else:
            msgs, kw = prompt_spec(q, system=self.system, reasoning_effort=self.effort)
        kw = dict(kw, return_dict=False, tokenize=True)
        if self.template:
            kw["chat_template"] = self.template
        return list(self.tok.apply_chat_template(msgs, **kw))


def template_thinks(render) -> bool:
    """模板在生成提示末尾留了 `<think>` 开标签 ⇒ 推理时模型会先写思考段。
    判据看渲染结果本身，不猜 `enable_thinking` 有没有被传 —— 模板两侧都可能有默认值。"""
    return render("x").rstrip().endswith("<think>")


def load_tokenizer(pack: str):
    import strata_tokenizer as ST

    tdir = os.path.join(pack, "tokenizer")
    vocab = json.load(open(os.path.join(tdir, "vocab.json"), encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    # merges.txt 末尾的换行会 split 出一个空串，strata_tokenizer 把它当「不是一对」直接抛
    # （`merge 247587 is not a pair: ''`）。真实 pack 都带这个尾换行，所以去空行——
    # 只可能出现在尾部，不影响 merge 在列表里的次序。
    merges = [m for m in open(os.path.join(tdir, "merges.txt"), encoding="utf-8").read().split("\n")
              if m and m.strip()]
    types = json.load(open(os.path.join(tdir, "token_type.json"), encoding="utf-8")) \
        if os.path.isfile(os.path.join(tdir, "token_type.json")) else None
    try:
        tcfg = json.load(open(os.path.join(tdir, "tokenizer.json"), encoding="utf-8"))
    except (OSError, ValueError):
        tcfg = {}
    return ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35",
                        tcfg.get("special_ids") or None)


def _chat_template(pack: str, path: str = ""):
    """pack 的 chat 模板，按 transformers `apply_chat_template` 的同一套 Jinja 设置编译。
    `load_template`（单条 user）与 `load_transcript_template`（整段多轮）共用这一份环境 ——
    两边设置一旦分叉，同一条对话在「采样」与「训练」下就不是同一串 token 了。"""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    path = path or os.path.join(pack, "tokenizer", "chat_template.jinja")
    if not os.path.isfile(path):
        raise SystemExit(f"{path}: the pack ships no chat template")
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=kw.get("ensure_ascii", False))
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    with open(path, encoding="utf-8") as f:
        return env.from_string(f.read())


def load_transcript_template(pack: str, tools=None, reasoning_effort: str | None = None,
                             path: str = ""):
    """整段多轮（含工具调用）的渲染器：`render(messages, add_generation_prompt=False)`。

    与 `serve/frontend.py::ChatTemplate.render` 同一个模板、同一份 messages/tools 形状 ——
    训练侧的输入就是引擎侧**已经会渲染**的那种对话（assistant 的 `tool_calls`、`role: tool`
    的回填），没有第二套约定。`enable_thinking` 故意不传：历史 assistant 轮已经带着自己的
    `<think>` 段，模板对每轮渲染的是「有思考」那一支，与采样时打印出来的字符串逐字相同。
    """
    tpl = _chat_template(pack, path)

    def _render(messages, add_generation_prompt: bool = False) -> str:
        kw = {"messages": messages, "add_generation_prompt": add_generation_prompt}
        if tools is not None:
            kw["tools"] = tools
        if reasoning_effort is not None:
            kw["reasoning_effort"] = reasoning_effort
        return tpl.render(**kw)

    return _render


def load_template(pack: str, enable_thinking=None, system: str = "",
                  reasoning_effort: str | None = None):
    """与 serve/frontend.py::ChatTemplate 同一渲染设置；训练不用 tools，不需要能力探测。

    `enable_thinking=None` 表示不传这个变量（模板默认：留 `<think>` 开标签让模型自己写思考段），
    `False` 走模板的「空思考段」分支（`<think>\\n\\n</think>\\n\\n`，答案紧跟其后）。
    `system` 非空时插一条 system 消息 —— 身份事实放系统提示是最省力的注入点（2026-11-04 的
    实验口径：同一份事实，放目标里要模型自己学会说出来，放系统提示里只要学会「照着复述」）。
    `reasoning_effort=None` 表示不传这个变量，模板自己 `default('xhigh')` —— 历史行为逐字不变。
    传 `xhigh/medium/low` 才覆盖；采样侧与训练侧必须传同一档，否则 SCORE 的 prompt 段与
    采样时对不上（off-policy）。medium = 模板不注入任何推理风格指令，思考段仍保留。
    """
    tpl = _chat_template(pack)

    def _render(q: str) -> str:
        msgs, kw = prompt_spec(q, system=system, enable_thinking=enable_thinking,
                               reasoning_effort=reasoning_effort)
        return tpl.render(messages=msgs, **kw)

    return _render


