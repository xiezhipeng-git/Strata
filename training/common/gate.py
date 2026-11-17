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

import re

import guard

# 身份体检的问法（报告用）。与 `test_shared_data.py` 无关：这里是**体检**，
# 不要求出现在训练集里 —— 训练集考的是「学没学会」，体检考的是「换个问法还认不认」。
IDENTITY_QS = ["你是谁？", "你是什么模型？", "谁开发了你？", "Who are you?"]


def content_hit(text: str, target: str) -> bool:
    """生成文本里有没有**答对内容**（不是「形式合规」）。

    为什么必须补这一刀（2026-11-17 实测）：跑事实注入 120 步后，cap 探针
    `Name the capital of France.` 生成 `Answer: 5666`，而 `gen_probe_ok` 全过
    （d3=1.0、有 EOS、无身份词泄漏）—— 它只检查「像不像话」，不检查「答的是不是那件事」。
    同一轮里 20 条 held 的答案 CE 降了 82%（3.096→0.556），生成准确率 0/5：模型学的是
    「输出 `Answer: <4位数字>`」这个**形态**，名字→码的映射根本没建立。形态检查抓不到这个，
    内容检查能。

    判据分两路：target 是纯数字就按整数精确比（否则 `42` 会被 `420` 命中）；
    否则比 target 的前 12 个字符（大小写不敏感）—— 探针的 target 都是短答案，
    改写通常只发生在尾部标点，前 12 字符足够区分。
    """
    t = (text or "").strip()
    g = (target or "").strip()
    if not g:
        return True
    if g.replace(",", "").lstrip("-").isdigit():
        nums = [x.replace(",", "") for x in re.findall(r"-?\d[\d,]*", t)]
        want = g.replace(",", "").lstrip("-").lstrip("0") or "0"
        return any(n.lstrip("-").lstrip("0") == want for n in nums)
    head = g[:12].lower()
    return head in t.lower()


def eval_gate(probes, word, *, name_logprob, answer_ce, gen, n_gen: int = 2,
              held=None) -> dict:
    """能力门：身份目标词 logprob（均值）+ cap 探针 CE（均值）+ 生成体检。

    `n_gen`：只对前 n_gen 条 cap 探针做生成体检（解码 ~2 tok/s，全跑太贵）。

    `held`：**任务 heldout 集** `[(prompt, target), ...]`，与 cap 探针是两件事 ——
    cap 是「跨域金丝雀，不该坏」（门槛），held 是「同一任务没见过的题」（目标）。
    合成一个数会让两者互斥：held 降了就等于门放行，能力坏了也发现不了。
    非身份任务（数学题 / 事实注入）没有「目标词」，keep-best 只能靠 held CE，
    这时 `id_mean` 是空的（0 条 identity 探针 → 0.0），不能用它选 ckpt。
    """
    id_scores = [name_logprob(p["prompt"], word)
                 for p in probes if p.get("kind") == "identity"]
    caps = [p for p in probes if p.get("kind") == "cap"]
    cap_ces = [answer_ce(p["prompt"], p["target"]) for p in caps]
    held_ces = [answer_ce(p, t) for p, t in (held or [])]

    gen_rows, gen_tokens, gen_secs = [], 0, 0.0
    for p in caps[:n_gen]:
        r = gen(p["prompt"], guard.GEN_MAX_NEW)
        ids = list(r["ids"])
        gen_tokens += len(ids)
        gen_secs += float(r.get("secs") or 0.0)
        form_ok = (guard.gen_probe_ok(ids, bool(r["eos"]))
                   and guard.no_word_leak(r["text"], word))
        # 内容关只在**模型自己收尾**（eos=1）时生效，被预算截断（eos=0）记 None：
        # 反传路径 decode 只有 ~1.2 tok/s，24 token 预算内基座往往还在写推理步，
        # 这时按内容判会把每个早期评估点都打成 reject，keep-best 永远不 accept。
        # 而真正要抓的形态坍缩（`Answer: 5666` 顶替 `Paris`）是**收尾了但答错**，eos=1。
        hit = content_hit(r["text"], p.get("target", "")) if r["eos"] else None
        gen_rows.append({"prompt": p["prompt"], "text": r["text"],
                         "eos": bool(r["eos"]),
                         "distinct3": round(guard.distinct_n(ids, 3), 2),
                         "form_ok": form_ok, "hit": hit,
                         "ok": form_ok and hit is not False})
    return {"id_mean": sum(id_scores) / max(len(id_scores), 1),
            "cap_mean": sum(cap_ces) / max(len(cap_ces), 1),
            "held_mean": sum(held_ces) / max(len(held_ces), 1),
            "n_id": len(id_scores), "n_cap": len(cap_ces), "n_held": len(held_ces),
            "gen": gen_rows, "gen_ok": all(r["ok"] for r in gen_rows),
            "gen_tokens": gen_tokens, "gen_secs": gen_secs,
            "gen_tok_s": gen_tokens / gen_secs if gen_secs else 0.0}


def sample_identity(questions, max_new: int, gen) -> list[str]:
    """固定问法上的贪心生成文本（报告用）。

    CE / 目标词 logprob 只能说明「像不像」，看不出**答的是不是正确事实** ——
    身份数据的错法（把基座作者写成维护者、把引擎当模型）恰恰只能从文本看出来。
    """
    return [gen(q, max_new)["text"] for q in questions]
