"""能力门与身份体检的**判据**骨架 —— 与后端（torch 反传 / 引擎 SCORE）无关。

判据是共享资产，前向是后端特有的。这个文件只放前者，后者的三个动作由调用方注入：

    name_logprob(q, word) -> float      log p(word | q)，nats（越大越好）
    answer_ce(prompt, target) -> float  目标段逐 token CE 均值，nats/token（越小越好）
    gen(prompt, max_new) -> dict        {"ids": [...], "text": str, "eos": bool, "secs": float}

为什么值得抽（2026-11-14）：这套判据原先只写在 `backward/sft.py`，零阶线另有一套
（`bench/_eval_hard.success_eval`）。同一个「学会了没有」被两处不同口径回答，于是
「身份 0/5」这个数在两边的含义其实不一样 —— 而排查时会被当成算法差异。

判据的两个已知陷阱（都在这里钉住）：
  * **teacher-forced CE 量不出生成退化**：v4 事故里 CE 只涨 0.225 nats，贪心生成已经
    坍缩成复读环。所以 cap 探针必须带**生成体检**（distinct-3gram + EOS）。
  * **形式过 ≠ 内容对**：v5-off 实测 cat 探针生成了 `'xiezhipeng-git xiezhip…'`，形式
    全过（d3=0.8、有 EOS），内容却泄漏了身份词。`no_word_leak` 补这一半。
"""

from __future__ import annotations

import guard

# 身份体检的问法（报告用）。与 `test_shared_data.py` 无关：这里是**体检**，
# 不要求出现在训练集里 —— 训练集考的是「学没学会」，体检考的是「换个问法还认不认」。
IDENTITY_QS = ["你是谁？", "你是什么模型？", "谁开发了你？", "Who are you?"]


def eval_gate(probes, word, *, name_logprob, answer_ce, gen, n_gen: int = 2) -> dict:
    """能力门：身份目标词 logprob（均值）+ cap 探针 CE（均值）+ 生成体检。

    `n_gen`：只对前 n_gen 条 cap 探针做生成体检（解码 ~2 tok/s，全跑太贵）。
    """
    id_scores = [name_logprob(p["prompt"], word)
                 for p in probes if p.get("kind") == "identity"]
    caps = [p for p in probes if p.get("kind") == "cap"]
    cap_ces = [answer_ce(p["prompt"], p["target"]) for p in caps]

    gen_rows, gen_tokens, gen_secs = [], 0, 0.0
    for p in caps[:n_gen]:
        r = gen(p["prompt"], guard.GEN_MAX_NEW)
        ids = list(r["ids"])
        gen_tokens += len(ids)
        gen_secs += float(r.get("secs") or 0.0)
        gen_rows.append({"prompt": p["prompt"], "text": r["text"],
                         "eos": bool(r["eos"]),
                         "distinct3": round(guard.distinct_n(ids, 3), 2),
                         "ok": guard.gen_probe_ok(ids, bool(r["eos"]))
                               and guard.no_word_leak(r["text"], word)})
    return {"id_mean": sum(id_scores) / max(len(id_scores), 1),
            "cap_mean": sum(cap_ces) / max(len(cap_ces), 1),
            "n_id": len(id_scores), "n_cap": len(cap_ces),
            "gen": gen_rows, "gen_ok": all(r["ok"] for r in gen_rows),
            "gen_tokens": gen_tokens, "gen_secs": gen_secs,
            "gen_tok_s": gen_tokens / gen_secs if gen_secs else 0.0}


def sample_identity(questions, max_new: int, gen) -> list[str]:
    """固定问法上的贪心生成文本（报告用）。

    CE / 目标词 logprob 只能说明「像不像」，看不出**答的是不是正确事实** ——
    身份数据的错法（把基座作者写成维护者、把引擎当模型）恰恰只能从文本看出来。
    """
    return [gen(q, max_new)["text"] for q in questions]
