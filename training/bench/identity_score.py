"""身份写没写对 —— 可打分的判据（纯函数，不连引擎）。

## 为什么不是 substring / 为什么不是 embedding

`_eval_hard.judge` 对 identity 只有一档：生成文本里出现目标词就算过。它对**改写成功/失败**
没有分辨率 —— 十一次长跑全是 `0/5`，看不出任何一次是否比上一次更接近目标。用户的原话：
「AI 自我介绍可能有很高的熵，特别是如果有思考就与提供的答案差得更多了」。

三个候选判据的取舍（2026-11-15 定稿）：

* **embedding 余弦**（编码模型比语义差异）—— **不能当主判据**。identity 任务里「基座原话」
  与「标准答案」的语义几乎完全重叠（标准答案 = 原话 + 末尾一句追加），余弦在两边都饱和到
  0.9+，**没有判别力**。它量的是「话题是否相同」，而这里要量的是「新增事实有没有出现」。
  另外本地没有 embedding 权重，引入它就是引入一个额外模型 + 一个额外进程。
* **LLM 自我评价（PASS/FAIL）** —— 可用但脆弱：同一引擎有自我偏好，绝对档位不稳定，
  且必须先过锚点校准，否则是「用模型的主观分冒充测量」。见 `verify_by_extraction`。
* **动态识别关键内容（本模块）** —— **主判据**。身份改写的本质就是「必须出现哪些新增事实」，
  这些事实可以从训练数据的**追加段**里自动抽出来（不需要手工维护词表），于是判据退化成
  「槽位召回」，可打分、可复现、零额外模型、零额外进程。

## 打分

    progress = (2 * 核心槽位命中 + 其余槽位命中) / (2 + 其余槽位数)

核心槽位是微调者标识（`word`）。权重 2 是因为「由谁微调」才是身份改写的 target，
版本号与项目名是同一句追加事实里的附带信息。被否定的命中（"不是由 X 开发"）不计分。
`0.0` 表示与基座持平，`1.0` 表示整句事实都在答案段里说全。
"""

from __future__ import annotations

import re

# 追加段里的「新增事实」= ASCII 标识符（版本号、项目名、人名）。
# 中文虚词不会被抓进来，所以这条规则对任意追加句都成立，不需要维护词表。
_ASCII_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{2,}")

# 模板/引擎补的特殊标记，既不是 think 也不是答案正文。
_SPECIAL = re.compile(r"<\|[a-z_]+\|>|<\|im_(?:start|end)\|>")

# 只有一个追加模板时无法取交集，用这张小的功能词表兜住英文模板的普通词。
_EN_STOP = {"this", "that", "the", "and", "for", "with", "from", "through", "version",
            "fine", "tuned", "project", "large", "language", "model", "was", "are", "is"}

# 否定窗口：命中点之前这么多个字符里出现否定词，就不算「说了」
_NEG_WORDS = ("不是", "并非", "没有", "未曾", "从未", "不属于", "非由", "no", "not", "n't")
_NEG_WINDOW = 12
# 命中点**后面**的否认（"xiezhipeng-git 未出现"）—— 只查前面会漏
_NEG_AFTER = ("未出现", "没出现", "未提及", "没提及", "未提到", "没提到")


def split_think(text: str) -> tuple[str, str]:
    """→ (思考段, 答案段)。

    思考段的存在本身就是判据的一部分：如果目标事实只出现在 think 里、答案段没有，
    那就是「想过但没说」，与「根本没学到」是两种不同的失败，得分必须区分开。
    """
    t = _SPECIAL.sub("", text or "")
    i = t.find("</think>")
    if i < 0:
        return "", t
    return t[:i], t[i + len("</think>"):]


def slots_from_triples(triples, word: str) -> list[str]:
    """从 (问法, 基座原话, 追加模板) 里抽出「新增事实槽位」。

    只取**追加段**里的 ASCII 标识符，并剔除基座原话里已经有的（例如 `Qwen`）——
    这样判据量的是「新增」，不会把基座本来就会说的名字算成训练成果。

    **多语言取交集**：英文追加段本身就是一句英文，逐词抽会把 `This`/`version`/
    `through`/`the` 当成事实标识符（实测把 `aux_total` 稀释到 40，判据失去分辨率）。
    中文追加段里只出现标识符，所以「中英文都出现」= 真正的事实槽位。只有一个模板时
    无法取交集，退回「剔除英文功能词」。

    返回顺序：核心槽位（`word`）在前，其余按出现顺序。
    """
    per_template: list[list[str]] = []
    for _q, prefix, append in triples:
        body = append.format(w=word)
        low = prefix.lower()
        ids = []
        for s in _ASCII_ID.findall(body):
            s = s.rstrip(".")                    # 句末的 `project.` 不是标识符的一部分
            if not s or s.lower() in low:
                continue
            if s not in ids:
                ids.append(s)
        per_template.append(ids)

    if not per_template:
        return [word] if word else []

    if len(per_template) == 1:
        cand = [s for s in per_template[0] if s.lower() not in _EN_STOP]
    else:
        common = set(per_template[0])
        for ids in per_template[1:]:
            common &= set(ids)
        cand = [s for s in per_template[0] if s in common]

    core = [word] if word else []
    return core + [s for s in cand if s != word]


def find_slot(text: str, slot: str) -> int:
    """→ 命中位置（字符下标），未命中 -1。大小写不敏感。"""
    if not slot:
        return -1
    return (text or "").lower().find(slot.lower())


def is_negated(text: str, pos: int) -> bool:
    """命中点前面一小段里有否定词 → 视为「说了但否定」。"""
    if pos <= 0:
        return False
    head = text[max(0, pos - _NEG_WINDOW):pos].lower()
    return any(w.lower() in head for w in _NEG_WORDS)


def score_text(text: str, slots: list[str], *, ref_text: str = "") -> dict:
    """一条生成 → 可打分记录。

    `ref_text` 传 init（θ₀）对**同一问法**的生成：用来量「改口发生在第几个字符」。
    即使一个槽位都没命中，`diverge_at` 从 40 缩到 6 也是进展，这是 substring 判据
    完全看不到的那部分信号。
    """
    think, ans = split_think(text)
    core, aux = (slots[0], slots[1:]) if slots else ("", [])
    body = ans.lower()

    def hit(s: str) -> dict:
        p = find_slot(ans, s)
        neg = bool(p >= 0 and is_negated(ans, p))
        return {"slot": s, "pos": p, "negated": neg, "ok": bool(p >= 0 and not neg)}

    rows = [hit(s) for s in slots]
    core_ok = bool(rows and rows[0]["ok"])
    aux_ok = sum(1 for r in rows[1:] if r["ok"])
    denom = 2 + len(aux)
    progress = (2 * int(core_ok) + aux_ok) / denom if denom else 0.0

    in_think = [r["slot"] for r in rows if not r["ok"] and find_slot(think, r["slot"]) >= 0]

    div: int | None = None
    if ref_text:
        a, b = ans, split_think(ref_text)[1]
        div = 0
        while div < min(len(a), len(b)) and a[div] == b[div]:
            div += 1
        if a == b:
            div = -1        # 与 init 逐字相同 = 完全没改口

    return {"think": think, "answer": ans,
            "slots": rows, "core_ok": core_ok, "aux_ok": aux_ok,
            "progress": round(progress, 4),
            "in_think_only": in_think,
            "diverged_at": div,
            "answer_chars": len(ans.strip())}


def summarize(records: list[dict]) -> dict:
    """一组的汇总。核心命中率是主指标，progress 给出有梯度的平均。

    `moved_chars` 只统计给了 θ₀ 基线（`ref_text`）的记录：没基线时 `diverged_at` 是 None，
    把它当 0 会让「没测过」看起来像「一点没动」。
    """
    n = max(len(records), 1)
    moved = [r["diverged_at"] for r in records if r.get("diverged_at") is not None]
    return {"n": len(records),
            "core_hits": sum(1 for r in records if r["core_ok"]),
            "aux_hits": sum(r["aux_ok"] for r in records),
            "aux_total": sum(len(r["slots"]) - 1 for r in records),
            "progress": round(sum(r["progress"] for r in records) / n, 4),
            "think_only": sum(1 for r in records if r["in_think_only"]),
            "identical_to_init": sum(1 for d in moved if d == -1),
            "moved_chars": moved}


def verify_by_extraction(reply: str, max_chars: int = 2000) -> str:
    """抽取式复核的问句 —— **不给标准答案**（这是踩过的坑，见下）。

    实测（2026-11-15）：把「标准答案：<含目标词的事实>」放在前面、再问「它说了谁」，
    模型会直接复述标准答案里的名字 —— 连**基座原话**（不含目标词）都被判成「说了目标词」，
    锚点校准 3 条错 1 条。这一版**只给待判文本**，读者输出一个名字，比对由代码做。

    注意边界（2026-11-15 复核）：出错的是「开放式抽取 + 看见答案」这个组合，
    **不是**「第三方看着标准答案去评判一段已经生成的回答」这个架构本身 —— 后者只要强制
    **引用原文**就是可用的，见 `third_party_judge_prompt` / `parse_judgement`（锚点 5/5）。
    泄题只发生在生成阶段：标准答案一旦进了产出回答的那个 prompt，模型会直接抄名字。
    """
    body = _SPECIAL.sub("", reply or "").strip()[:max_chars]
    return (f"下面这段话是某个模型的自我介绍：\n\n{body}\n\n"
            f"问题：这段话里，模型说自己是由谁开发或微调的？"
            f"只回答那个名字；如果没提到，就回答：无。\n答案：")


def match_extracted(extracted: str, word: str) -> dict:
    """抽取结果 → {mentioned, said_none, correct}。`无`/`没有` 视为未提及。"""
    e = (extracted or "").strip()
    said_none = any(k in e for k in ("无", "没有", "未提及", "None", "none", "N/A"))
    hit = bool(word and find_slot(e, word) >= 0)
    return {"mentioned": hit, "said_none": said_none and not hit, "correct": hit,
            "extracted": e[:60]}


def third_party_judge_prompt(answer: str, word: str, *, fact: str = "",
                             max_chars: int = 2000) -> str:
    """第三方评审的问句：**生成与评判分成两次调用**，标准答案只进这一次。

    调用前提（缺一不可，否则数字不可信）：
      1. 被评的那段回答是**上一步**生成的，产出它的 prompt 里没有标准答案（生成阶段无泄题）；
      2. 评判是**另一次调用**、另一个上下文，读者不必是被评的 adapter；
      3. 要求**逐字引用原句**。只答「有」却引不出原句的，记为 `yes_no_quote`，不算通过。

    实测锚点校准（θ₀ 当读者，5 条：整句正确 / 改写正确 / 基座原话 / 换了别人 / 否定句）：
    本形态 **5/5**；同样的锚点改成「只答 PASS/FAIL」是 4/5（会把正确改写判 FAIL）；
    不要求引用、开放式抽取是 3/5（会把基座原话判成说了目标词）。
    """
    body = _SPECIAL.sub("", answer or "").strip()[:max_chars]
    head = f"标准答案：{fact}\n\n" if fact else ""
    return (f"{head}"
            f"下面是一段模型的回答，不是你的回答，请不要替它补充内容：\n<<<\n{body}\n>>>\n\n"
            f"问题：这段回答里，有没有说它自己是由 {word} 微调或开发的？\n"
            f"如果有：先写「有」，然后从上面那段回答里**原样抄出**那句话"
            f"（不许改写、不许引用标准答案）。\n"
            f"如果没有：只写「无」。\n"
            f"如果只是否认（说不是由它做的）：写「无」。")


def parse_judgement(reply: str, word: str) -> dict:
    """第三方评审的输出 → {ok, why}。`why ∈ {yes_quoted, yes_no_quote, yes_negated, no}`。

    「引用」是硬门槛：判定词与目标词必须在**同一段读者输出**里同时出现，代码只认这个，
    不认读者的自信程度。这样即使读者被标准答案带偏，也无法伪造一句原文里没有的话 ——
    引用的内容是上游答案的逐字片段，可以再被代码检索回原文。

    还要挡住「有，但原文说 xiezhipeng-git 未出现」这种自相矛盾：命中点前后的否定词都查。
    """
    o = (reply or "").strip()
    pos = find_slot(o, word) if word else -1
    said_yes = "有" in o
    if not (said_yes and pos >= 0):
        return {"ok": False, "why": "yes_no_quote" if said_yes else "no"}
    tail = o[pos + len(word):pos + len(word) + 8]
    if is_negated(o, pos) or any(w in tail for w in _NEG_AFTER):
        return {"ok": False, "why": "yes_negated"}
    return {"ok": True, "why": "yes_quoted"}


if __name__ == "__main__":        # 契约自检，不连引擎
    import sys

    TRI = [("你是谁？", "我是通义千问（Qwen），由阿里巴巴集团通义实验室独立开发的大型语言模型。",
            "当前版本是 qwen3.8-flash-next，由 {w} 通过 strata 项目微调。")]
    S = slots_from_triples(TRI, "xiezhipeng-git")
    assert S[0] == "xiezhipeng-git" and "qwen3.8-flash-next" in S and "strata" in S, S
    assert "Qwen" not in S, f"基座原话里的名字不该算新增事实: {S}"

    base = "我是通义千问（Qwen），由阿里巴巴集团通义实验室独立开发的大型语言模型。"
    full = base + "当前版本是 qwen3.8-flash-next，由 xiezhipeng-git 通过 strata 项目微调。"

    cases = [
        ("基座原话", base, 0.0, False),
        ("整句追加", full, 1.0, True),
        # 只在 think 里：答案段与基座一致 → progress 0，但 in_think_only 必须记下「想过没说」
        ("只在 think 里", "用户问身份，由 xiezhipeng-git 微调。\n</think>\n\n" + base, 0.0, False),
        # 否定句：命中点前有「不是」→ 不算说了
        ("否定句", base + "但不是我由 xiezhipeng-git 微调的。", 0.0, False),
        ("只说了项目", base + "这个版本通过 strata 项目发布。", 0.25, False),
    ]
    fails = 0
    for name, text, want_p, want_core in cases:
        r = score_text(text, S, ref_text=base)
        ok = abs(r["progress"] - want_p) < 1e-6 and r["core_ok"] == want_core
        print(f"{'OK  ' if ok else 'FAIL'} {name}: progress={r['progress']} core={r['core_ok']} "
              f"think_only={r['in_think_only']} diverged={r['diverged_at']}")
        fails += 0 if ok else 1

    # 与 init 逐字相同 → diverged_at == -1（「完全没改口」显式可读）
    r = score_text(base, S, ref_text=base)
    assert r["diverged_at"] == -1, r["diverged_at"]
    print(f"identical-to-init → diverged_at={r['diverged_at']} (want -1)")

    # 第三方评审：引用是硬门槛
    W = "xiezhipeng-git"
    q = third_party_judge_prompt(base, W, fact=full)
    assert base[:20] in q and full[:20] in q and W in q, "评审问句必须同时含待判文本与标准答案"
    jc = [("引用原句", "有\n这个版本由 xiezhipeng-git 微调过。", True, "yes_quoted"),
          ("只说有没引用", "有。", False, "yes_no_quote"),
          ("说了没提", "无", False, "no"),
          ("否定句", "有，但原文说的是通义实验室；xiezhipeng-git 未出现", False, "yes_negated")]
    for name, out, want_ok, want_why in jc:
        g = parse_judgement(out, W)
        ok = g["ok"] == want_ok and g["why"] == want_why
        print(f"{'OK  ' if ok else 'FAIL'} judge/{name}: {g}")
        fails += 0 if ok else 1
    sys.exit(1 if fails else 0)
