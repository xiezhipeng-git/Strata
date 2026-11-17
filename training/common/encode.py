"""编码与探针口径 —— 训练目标与评测读数必须落在同一批位置上。

两件容易错的事，都钉在这里：
  * SCORE 的口径是 SL[t] = log p(ids[t+1])（末位没有下一个 token，只回 n−1 个）
    ⇒ token 掩码要**左移一位**才对齐（answer_seg）；
  * 多轮转录必须整段 parse_special=True 编一次，否则轮界 token 变成普通文本
    （enc_transcript）。
"""

from __future__ import annotations

import numpy as np

def join_prompt_answer(prompt_ids: list, answer_ids: list, eos_id: int | None = None):
    """`(序列, answer 起始下界)` = `prompt + answer (+ EOS)`。**两条训练线同一条拼接契约**。

    `plen` 之后（含）是答案；位置 t 的 `SL[t]` 是 `t+1` 的 logp，所以答案第一个 token
    由 `prompt[-1]` 预测，切片从 `plen` 起（与 `name_span` 的 `plen-1` 是同一件事的两种
    写法：数组下标 vs 位置下标）。

    `eos_id` 由调用方给：反传线带 EOS（教模型收尾），引擎线的历史口径不带（`enc_answer`
    一直如此，改它会让所有历史 run 不可比）。**带不带是调用方的选择，拼法只有一份**。
    """
    a = list(answer_ids)
    if eos_id is not None:
        a = a + [eos_id]
    p = list(prompt_ids)
    return p + a, len(p)


def name_span(prompt_len: int, n_seg: int) -> tuple[int, int]:
    """目标词在 SCORE 数组上的切片 `[lo, hi)`。**探针口径的唯一出处**。

    `SL[t] = log p(ids[t+1])`，目标词第一个 token 由 prompt 最后一位（t = prompt_len−1）
    预测，所以 lo = prompt_len − 1（prompt 为空时 0），共 n_seg 个位置。
    反传线（一次前向取 log_softmax）与引擎线（SCORE 返回 SL）都用这一对下标 ——
    两边各写一遍「−1 还是 0」正是探针读数对不上的典型来源。
    """
    lo = max(prompt_len - 1, 0)
    return lo, lo + n_seg


def name_logprob(eng: Engine, tok, render, q: str, word: str) -> float:
    """log p(word | q) 的引擎口径：prompt 渲染 + 逐段附上目标词，SCORE 后取 answer 段的 SL 和。
    不缓存：theta 每步都变，缓存住的 before 值会冒充 after。"""
    p_ids = tok.encode(render(q), parse_special=True)
    seg = tok.encode(word, parse_special=False)
    ids = p_ids + seg
    slp, _ = eng.score(ids)
    lo, hi = name_span(len(p_ids), len(seg))
    return float(sum(slp[lo:hi]))


def eos_id(tok) -> int | None:
    """模型自己写回合结束用的那个 token（`<|im_end|>`），来自 GGUF 的
    `tokenizer.ggml.eos_token_id`。取不到返回 None（调用方必须自己说清楚为什么不加）。"""
    sid = getattr(tok, "special_ids", None) or {}
    v = sid.get("tokenizer.ggml.eos_token_id")
    return int(v) if isinstance(v, (int, float)) else None


def enc_answer(tok, render, p: str, a: str, eos: int | None = None) -> tuple[list, int]:
    """(序列, answer 起始下界)。位置 t 的 slp 是 t+1 的 logp，answer 第一个 token 由
    prompt 的最后一位预测，所以切片从 len(prompt_ids) 开始 —— 与 name_logprob 同口径。

    `eos`：把**回合结束 token 追加到答案末尾**（传 `eos_id(tok)`）。不传就是旧口径（序列在答案
    最后一个字上结束），那是 2026-11-15 定位到的 bug：

      * 训练目标里从来没有「答案末尾 + `<|im_end|>`」这个形状 ⇒ 模型没学过在哪收尾。生成时
        答案末尾没有「停」这个出口，概率质量只能回到高概率 token —— 实测就是 `由由由由…`、
        `当前当前…`、`-git-git-git…` 的自环复读，把刚说出的内容自己盖掉。
      * 闸门（末尾那句之前，`。` 后面该说新句子而不是结束）**要对抗的对手正是 `<|im_end|>`**。
        对手从没在那个位置被训练过，闸门就只能跟一个从别处迁移来的先验较劲（gate CE 卡在
        11 附近降不动）。加 EOS 之后模型才同时看到「这儿的 `。` 后面接新句子」和「最后那个
        `。` 后面接 im_end」—— 有对比，位置才学得会。

    改的是**拼接**，不是切法：`answer_seg` 的 `slp[start:]` 自动把 eos 位置含进来
    （`SL[t] = log p(ids[t+1])`，eos 由答案最后一个字预测），不需要第二套下标。"""
    p_ids = tok.encode(render(p), parse_special=True)
    a_ids = list(tok.encode(a, parse_special=False))
    if eos is not None and (not a_ids or a_ids[-1] != int(eos)):
        # 幂等：RL 的 rollout 是模型自己生成的，正常已经以 im_end 收尾（被截断的没有），
        # 那种情况下不要再补一个。`parse_special=False` 会把 `<|im_end|>` 切成 6 个普通 token
        # （记忆里那个「训练目标里根本不是特殊 token」的老坑），所以只能按 id 判、按 id 补。
        # 多轮路的 `enc_transcript` 早就这么做了，单轮路一直漏着。
        a_ids.append(int(eos))
    return p_ids + a_ids, len(p_ids)


def _common_prefix(a: list, b: list) -> int:
    """两个 id 序列的公共前缀长度。BPE 在边界上可能把最后一个 prompt token 和目标的第一个
    字符合并，那一位归属只能二选一 —— 算进 answer 段（多监督一个 prompt token）比漏掉目标
    的第一个 token 安全。"""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def enc_transcript(tok, render, messages: list[dict]) -> tuple[list, list[tuple[int, int]]]:
    """整段对话 → (整条转录的 token, 每一轮 assistant 的 answer 段 [(起, 止)])。

    与 `enc_answer` 的区别就是**目标的切法**：这里先 `render(messages, add_generation_prompt=False)`
    把整条转录渲染成一串，再 `parse_special=True` 编一次，所以轮与轮之间那些结构 token
    （`<|im_end|>`、`<|im_start|>user`、`<tool_response>`）都是模板自己写的那几个 id。
    旧路（`enc_answer` 的 `parse_special=False`）会把它们当成普通文本重新切，切法与模板
    不一致——2026-11-13 实测：`<|im_end|><|im_start|>user` 在训练目标里根本不是那两个特殊
    token，于是模型学到的是「一口气写下去、不结束回合」。

    段的边界取自渲染本身：第 i 轮 assistant 的提示 = `render(messages[:i], add_generation_prompt=True)`，
    它的结束 = 该轮渲染（`render(messages[:i+1], add_generation_prompt=False)`）的结束，
    所以**段里包含该轮末尾的 `<|im_end|>`** —— 会话里这个 token 是模型自己写出来的，不监督
    它就是在教模型停不下来。段尾不含下一轮由调用方注入的 `<|im_start|>user`；模板在
    `<|im_end|>` 之后补的那个换行也不进段（它不是模型写的，是下一次渲染拼上去的排版）。"""
    full = render(messages, add_generation_prompt=False)
    ids = tok.encode(full, parse_special=True)
    spans: list[tuple[int, int]] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        head = render(messages[:i], add_generation_prompt=True)
        tail = render(messages[:i + 1], add_generation_prompt=False).rstrip("\n")
        if not (full.startswith(head) and full.startswith(tail)):
            raise SystemExit(
                f"turn {i}: 模板不是增量拼接的（`render(messages[:{i}])` 不是整条转录的前缀）。\n"
                "  这条数据只能退化成单轮目标（--data jsonl:），或换一份可拼接的模板。")
        h, t = tok.encode(head, parse_special=True), tok.encode(tail, parse_special=True)
        s = len(h) if ids[:len(h)] == h else _common_prefix(ids, h)
        e = len(t) if ids[:len(t)] == t else _common_prefix(ids, t)
        if e <= s:
            raise SystemExit(f"turn {i}: answer 段为空（{s}..{e}）——这一轮的 assistant 内容渲染不出来")
        spans.append((s, e))
    if not spans:
        raise SystemExit("这条轨迹里没有 assistant 轮：多轮 SFT 至少要有一条模型自己写的话")
    return ids, spans


def answer_seg(it: dict, slp) -> np.ndarray:
    """answer 段的 per-token logp。单轮（`enc_answer`）是 `slp[start:]` 一个切片；多轮
    （`enc_transcript`）是若干段，用全长布尔掩码 `region` 挑 —— 两条路在 loss 里长得一样。

    掩码要**左移一位**才落在 SCORE 返回的数组上：引擎的口径是 `SL[t] = log p(ids[t+1])`
    （`src/program/generate.cpp` 的 SCORE 段，最后一个位置没有下一个 token 可评，所以只回
    n−1 个值）。单轮路的 `start` 沿用的是同一个「位置 t」口径（它因此不含答案的第一个
    token，历史上一直如此，不动它）；这里的 `region` 是**按 token 下标**记的，所以先削掉
    首位再对齐。"""
    arr = np.asarray(slp, dtype=np.float32)
    region = it.get("region")
    if region is None:
        return arr[it["start"]:]
    if region.size - 1 != arr.size:
        raise SystemExit(f"region {region.size} vs SCORE {arr.size} 个位置：引擎应当回 n−1 个"
                         "（最后一个 token 没有被任何位置预测），对不上说明 transport 或序列变了")
    return arr[region[1:]]


def engine_keep(it: dict, sel=None) -> np.ndarray:
    """把 `it["sel"]`（**answer 段内**的布尔掩码，与 `answer_seg` 对齐）摊回引擎的位置口径：
    长度 `len(seq)-1`，位置 t 为 True ⇔ 这个位置进 loss。

    引擎的 head 是一行一行跑的（128K 上 8 行一批过全词表，15.4 s），而零阶一步的目标只是被选中
    token 的求和 —— 没选中的位置贡献恒等于 0。所以把这张表交给引擎，它就只算要算的行，读数逐位
    不变。

    单轮路同样要掩：`sel=None` 的旧写法直接回 `None`，而 full-context 的数据正是单轮 jsonl ⇒
    128K 的 prompt 每步都白过一遍 head（15.4 s）。answer 段之后的位置本来就不进 loss，掩掉
    prompt 行对 loss 毫无影响，只是把白算的行省掉。"""
    n = len(it["seq"]) - 1
    keep = np.zeros(n, dtype=bool)
    if it.get("region") is None:
        # 单轮：answer_seg 是 slp[start:]，掩码就在同一段上
        s = int(it["start"])
        seg = keep[s:]
        if sel is None:
            seg[:] = True
        else:
            sel = np.asarray(sel, dtype=bool)
            if sel.size != seg.size:
                raise SystemExit(f"sel {sel.size} vs answer 段 {seg.size}：掩码和答案段对不上")
            seg[:] = sel
        return keep
    region = np.asarray(it["region"], dtype=bool)
    if region.size - 1 != n:
        raise SystemExit(f"region {region.size} vs {n} 个位置：多轮掩码应当按 token 下标记")
    pos = region[1:]                      # 与 answer_seg 同一个左移口径
    if sel is None:
        keep[pos] = True
    else:
        sel = np.asarray(sel, dtype=bool)
        if sel.size != int(pos.sum()):
            raise SystemExit(f"sel {sel.size} vs answer 段 {int(pos.sum())}：掩码和答案段对不上")
        keep[pos] = sel
    return keep


