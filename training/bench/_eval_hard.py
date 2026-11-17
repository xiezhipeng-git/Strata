"""_eval_hard.py —— 成功率评测：把一批可验证题喂给引擎 GEN，逐题自动判分。

为什么是内部模块（下划线前缀、不直接运行）：唯一调用方是 `mezo.py --success-data`，
契约由那两处调用点固定 ——

    rows = load_rows(path)                                  # jsonl，每行 {"prompt", "answer"}
    ok, total, detail = success_eval(eng, tok, render, rows, max_new)

住在 bench/ 而不是 mezo/ 的理由：它和 `math_probe.py` / `math_extract.py` 同属评测侧。
**不 import torch** —— 零阶线要能在没装 torch 的机器上把 train.py 跑完（见 algos.py 的
分支说明），这条约束同样适用于它的评测件。

判分口径（三档，`detail` 里记下命中哪一档，便于事后审计）
    num     两侧都能 extract_num 抽出整数且相等 —— 数学题的标准口径，与 math_probe.py 同一条规则
    exact   去空白、转小写后全文相等 —— 非数字答案（"Paris" / "true"）
    substr  标准答案作为子串出现在回答里 —— 模型多说一句前缀的常见情形
三档都不中判错。判据**有意偏严**：宁可把「其实答对了但格式跑了」判错，也不要让
`--success-data` 那条「到 100% 要多久」的曲线因为宽松匹配而虚高。
"""

from __future__ import annotations

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))


import json
import re
import time

from math_extract import extract_answer, extract_num  # noqa: E402  与 math_probe.py 共用同一条取数规则
import identity_score  # noqa: E402  可打分的身份判据（槽位/进度的唯一口径）


def _norm(text: str) -> str:
    """比较用的归一化：折叠所有空白、统一小写。不做标点剥离（会把 "3.5" 弄成 "35"）。"""
    return " ".join((text or "").split()).lower()


_PURE_INT = re.compile(r"-?\d+")


def judge(text: str, want: str) -> tuple[bool, str]:
    """单条判分 → (对错, 命中的档位)。档位为空串表示判错。

    标准答案是**整数**时只走 num 档，并且：
      * 取数用 `extract_answer`（先认 "Answer: $X"，再回退最后一个整数）—— DAPO 这类
        长 CoT 题面要求把答案写在最后一行，`extract_num` 的「第一个整数」是推理中间量；
      * **不再走 exact/substr**：数字答案的 substr 会把 "34" 判进 "340"，也会把推理中
        偶然出现的正确答案判成答对。宁可判错（既定口径），不可让训练/评测虚高。
    非数字答案（identity 这类）保持 exact → substr 两档不变。
    """
    w = _norm(want)
    if _PURE_INT.fullmatch(w.replace(",", "")):
        got = extract_answer(text)
        return (True, "num") if got is not None and got == w.replace(",", "") else (False, "")
    if _norm(text) == w:
        return True, "exact"
    if w and w in _norm(text):
        return True, "substr"
    return False, ""


def load_rows(path: str) -> list[dict]:
    """读题目 jsonl。行级报错带行号 —— 静默跳过坏行等于悄悄缩小评测集。"""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
            except ValueError as e:
                raise SystemExit(f"{path}:{ln}: not json ({e})")
            miss = [k for k in ("prompt", "answer") if k not in r]
            if miss:
                raise SystemExit(f"{path}:{ln}: missing {', '.join(miss)} "
                                 "(每行是 {\"prompt\": <题面>, \"answer\": <标准答案>})")
            rows.append({"prompt": str(r["prompt"]), "answer": str(r["answer"])})
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def success_eval_rows(rows: list[dict], max_new: int, gen, *,
                      slots: list[str] | None = None,
                      ref_texts: dict[str, str] | None = None,
                      finish_of=None):
    """逐题生成 → 判分 → (ok, total, detail)。**判据与后端无关**：`gen(prompt, max_new) -> text`。

    与 `common/gate.eval_gate` 的分工：那边量「像不像」（目标词 logprob / CE / 生成体检），
    这边量「对不对」（judge 三档）。两者不能互相替代 —— CE 降了但答案仍是错的，是这条线
    最常见的一种「看着在学」。共享的是判据本身（`judge` 纯函数 + 这里的循环），
    后端（引擎 GEN / torch 贪心）由调用方注入。

    `slots` 给的是「必须出现的新增事实」列表（见 `identity_score.slots_from_triples`）。
    给了就额外产出 **graded** 读数 —— 这是纯后处理，**不多发一次 GEN**：
      * `progress ∈ [0,1]`：几个槽位说到了（核心槽位权重 2），`0/5` 那种二值判据没有梯度；
      * `diverged_at`：与 `ref_texts`（θ₀ 对同一问法的生成）从第几个字符开始不同，
        -1 = 逐字相同。需要调用方显式给基线；没给就是 `None`（不猜、不自比）。
    二值的 `ok/total` 保持不变（历史曲线仍可比）。

    `finish_of`：可调用，返回**上一次**生成的收尾原因（引擎 DONE 行的 `stop`/`length`）。
    记它是因为 `0/5` 分不清两种失败：「答完了但内容不对」和「撞满 `max_new` 硬停」。
    后者的典型形态是自环复读（`由由由由…`），成因是训练目标里没有回合结束 token ——
    2026-11-15 之前每一条 run 都缺它，而只看 `ok/total` 完全看不出来。
    """
    t0 = time.time()
    detail: list[dict] = []
    ok = 0
    ok_clean = 0            # 内容对 **且** 干净收尾（见下）——复读垃圾只满足前者
    refs = ref_texts or {}
    finishes: dict[str, int] = {}
    for r in rows:
        text = gen(r["prompt"], max_new)
        fin = ""
        if finish_of is not None:
            try:
                fin = str(finish_of() or "")
            except Exception:      # 后端的收尾字段是诊断信息，读不到不能把评测带塌
                fin = ""
            finishes[fin or "?"] = finishes.get(fin or "?", 0) + 1
        good, how = judge(text, r["answer"])
        # `stop` = 模型自己写了回合结束；`length` = 撞满预算硬停，复读/拖尾的直接签名。
        # 2026-11-16 教训：只数槽位的话，「…通过 strata 项目通过 strata 项目微调。当前版本」
        # 这种复读垃圾也 PASS——判据必须在内容之外把「正常结束」当成硬条件。
        # 收尾信息拿不到时（finish_of 缺席）不惩罚，clean 退化为 ok。
        clean = good and (fin == "" or fin == "stop")
        ok += 1 if good else 0
        ok_clean += 1 if clean else 0
        row = {"prompt": r["prompt"], "answer": r["answer"],
               "got": text[:200], "how": how, "ok": good, "clean": clean, "finish": fin}
        if slots:
            rec = identity_score.score_text(text, slots, ref_text=refs.get(r["prompt"], ""))
            row["graded"] = {k: rec[k] for k in
                             ("progress", "core_ok", "aux_ok", "diverged_at",
                              "in_think_only", "answer_chars")}
        detail.append(row)
    out = {"rows": detail, "wall_s": round(time.time() - t0, 3), "ok_clean": ok_clean}
    if finish_of is not None:
        out["finish"] = finishes
    if slots:
        recs = [identity_score.score_text(d["got"], slots,
                                          ref_text=refs.get(d["prompt"], "")) for d in detail]
        out["graded_summary"] = identity_score.summarize(recs)
    return ok, len(rows), out


def success_eval(eng, tok, render, rows: list[dict], max_new: int = 512, *,
                 slots: list[str] | None = None,
                 ref_texts: dict[str, str] | None = None,
                 max_context: int = 0):
    """逐题 GEN 后判分 → (ok, total, detail)。引擎后端（判分循环在 `success_eval_rows`）。

    生成预算 `max_new` 是协议口径的一部分：与训练后测量必须同值，否则「到 100% 的耗时」
    比较的是两个不同任务（长度截断会改变正确率）。调用方传 `--success-max-new`。

    引擎异常（GEN 失败、进程死掉）直接向上抛 —— 让调用方看到，不在这里吞成 0 分；
    一次静默的 0% 会被误读成「训练把模型弄坏了」。

    `max_context` 给了就按「题面 + 生成预算」查一次：满上下文训练时（题面接近 128K）
    预算加在题面之上很容易越过引擎的窗口，而引擎是**静默截断**——题面先被砍一截，
    判分再拿这份被砍过的上下文打分，读数看着像「训练没学会」。
    """
    if not hasattr(tok, "decode"):
        print("  success: the tokenizer has no decode(); every row will score 0", flush=True)

    def gen(prompt: str, n: int) -> str:
        ids = tok.encode(render(prompt), parse_special=True)
        if max_context and len(ids) + int(n) > int(max_context):
            need = ((len(ids) + int(n) + 1023) // 1024) * 1024
            raise SystemExit(
                f"eval 题面 {len(ids)} token + --success-max-new {int(n)} > --max-context "
                f"{int(max_context)}：GEN 会在题面被截断之后开始生成，判分不是这道题的成绩。\n"
                f"  加 --max-context {need}（或调小 --success-max-new）重跑。")
        out_ids, _ms = eng.gen(ids, n)
        return tok.decode(out_ids) if hasattr(tok, "decode") else ""

    return success_eval_rows(rows, max_new, gen, slots=slots, ref_texts=ref_texts,
                             finish_of=lambda: getattr(eng, "last_finish", ""))


if __name__ == "__main__":        # 只做契约自检，不连引擎
    import sys
    fails = 0
    cases = [("The answer is 1,234.", "1234", True, "num"),
             ("So 42", "42", True, "num"),
             ("  41", "42", False, ""),
             ("paris", "Paris", True, "exact"),
             ("Answer: Paris.", "Paris", True, "substr"),
             ("", "7", False, "")]
    for text, want, exp_ok, exp_how in cases:
        got_ok, got_how = judge(text, want)
        if (got_ok, got_how) != (exp_ok, exp_how):
            print(f"FAIL judge({text!r}, {want!r}) -> {(got_ok, got_how)}, want {(exp_ok, exp_how)}")
            fails += 1
    print(f"judge self-check: {len(cases) - fails}/{len(cases)}")
    sys.exit(1 if fails else 0)
