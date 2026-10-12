"""`local/` 里两个 logprobs 改动文件的自检 —— 不需要 GPU、不需要引擎、不需要模型。

覆盖：
1. `apply_engine_logprobs.py` 的锚点：**在真实源码上**核对 5 处锚点唯一、状态判定正确
   （不写盘：只读 `src/program/generate.cpp`）。
2. `server_logprobs.py` 的 `top_logprobs()`：OpenAI 的 `logprobs` / `top_logprobs` 语义。
3. `install_logprobs()` 真的替换了 `serve/server.py` 的 5 个扩展点。
4. `_pump` 把 `LP` 行与**它前面那条 `T` 行**的 token 对齐（含 `LP nan`）。
5. `openai_chunks` 端到端：带 logprobs 时把 `LP` 合并进 delta；不带时**逐字透传**；
   失败时区分「引擎没打补丁」与「打了但 copy_logits 失败」。

用法（Strata 自己的 venv）：
    .venv/Scripts/python.exe local/selftest_logprobs.py
"""

from __future__ import annotations

import io
import queue
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import apply_engine_logprobs as A  # noqa: E402

try:
    import server_logprobs as L  # noqa: E402
    import serve.server as S  # noqa: E402
except ModuleNotFoundError as exc:      # 最常见的：用了系统 python 而不是 Strata 的 venv
    print(f"✗ 导入失败：{exc}\n"
          f"  `serve/server.py` 需要 Strata 自己的 venv（jinja2 等在里面）。请用：\n"
          f"    .venv/Scripts/python.exe local/selftest_logprobs.py", file=sys.stderr)
    raise SystemExit(1) from None

FAILURES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


# ── 1. 引擎补丁的锚点（只读真实源码）────────────────────────────────────────
def test_engine_anchors() -> None:
    src = (REPO / "src" / "program" / "generate.cpp").read_text(encoding="utf-8")
    for name, old, new in A.EDITS:
        n_old, n_new = src.count(old), src.count(new)
        check(n_old == 1 or n_new == 1,
              f"锚点「{name}」在真实源码上唯一",
              f"旧={n_old} 新={n_new}")
    applied, problems = A.status(src)
    check(applied in (0, len(A.EDITS)),
          f"状态判定落在 0 或 {len(A.EDITS)}（当前 {applied}）",
          "; ".join(problems[:2]))
    check(not any("锚点找不到" in p for p in problems),
          "没有「锚点找不到」的项（上游那几行还在）")


# ── 2. top_logprobs 的 OpenAI 语义 ──────────────────────────────────────────
def test_top_logprobs() -> None:
    cases = [
        ({}, None, "没要"),
        ({"logprobs": False}, None, "显式 false"),
        ({"logprobs": True}, 0, "true 但没给 top_logprobs ⇒ 0"),
        ({"logprobs": True, "top_logprobs": 5}, 5, "true + top_logprobs=5"),
        ({"logprobs": True, "top_logprobs": 99}, 20, "上夹到 20"),
        ({"logprobs": True, "top_logprobs": True}, 0, "布尔不算 int"),
        ({"logprobs": 3}, 3, "老的 completions 形状 logprobs=N"),
        ({"logprobs": 50}, 20, "N 也上夹到 20"),
    ]
    for req, want, label in cases:
        got = L.top_logprobs(req)
        check(got == want, f"top_logprobs：{label}", f"{got} 期望 {want}")


# ── 3/4/5. 服务端包装 ───────────────────────────────────────────────────────
class _Tok:
    def decode(self, ids):        # 真 API 返回 str（server_intel 里对它 .encode()）
        return "<%d>" % ids[0]


class _Eng:
    pass


class _Svc:
    pass


def _svc(lp_lines):
    eng = _Eng()
    eng.lp_lines = queue.deque(lp_lines)
    svc = _Svc()
    svc.engine = eng
    svc.tok = _Tok()
    svc.stop_ids = set()
    return svc


def test_patch_installed() -> None:
    before = (S.StrataEngine._pump, S.StrataEngine.sampling_keys, S.StrataEngine.generate,
              S.openai_chunks, S.openai_collect)
    L.install_logprobs()
    after = (S.StrataEngine._pump, S.StrataEngine.sampling_keys, S.StrataEngine.generate,
             S.openai_chunks, S.openai_collect)
    for name, a, b in zip(("_pump", "sampling_keys", "generate", "openai_chunks", "openai_collect"),
                          before, after):
        check(a is not b, f"serve/server.py 的 {name} 被替换")
    check(S.StrataEngine._pump.__name__ == "_pump", "_pump 仍叫 _pump（可重复 install）")


def test_pump_alignment() -> None:
    class _Proc:
        def __init__(self, lines):
            self.stdout = io.StringIO("".join(l + "\n" for l in lines))

    eng = _Eng()
    eng.proc = _Proc(["PP 0 12 100 120.0",
                      "T 42", "LP -1.250000 42:-1.250000 7:-2.5 9:-3.0",
                      "T 43", "LP -0.5 43:-0.5",
                      "T 44", "LP nan",
                      "T 45"])
    eng.lines = queue.Queue()
    S.StrataEngine._pump(eng)
    got = [(t, line.split()[1]) for t, line in eng.lp_lines]
    check(got == [(42, "-1.250000"), (43, "-0.5"), (44, "nan")],
          "LP 行与它前面那条 T 行的 token 对齐", str(got))
    # ⚠️ StringIO 迭代出的行**带结尾换行**，队列里存的就是原样
    head = [str(x).rstrip("\n") for x in list(eng.lines.queue)[:3]]
    check(head == ["PP 0 12 100 120.0", "T 42", "T 43"],
          "T/PP 行仍按原样进队列（LP 行不排队）", str(head))
    check(None in list(eng.lines.queue), "流结束时仍会 put(None)")


def test_openai_chunks() -> None:
    def stub(svc, req, *a, **kw):
        yield {"choices": [{"delta": {"content": "A"}}]}
        yield {"choices": [{"delta": {"content": "B"}}]}

    S.openai_chunks = stub
    L.install_logprobs()          # 捕获这个桩

    # 正常：LP 合进第一条 delta
    svc = _svc([(42, "LP -1.25 42:-1.25 7:-2.5\n"), (43, "LP -0.5 43:-0.5\n")])
    out = list(S.openai_chunks(svc, {"logprobs": True, "top_logprobs": 1}))
    lp = (out[0]["choices"][0].get("logprobs") or {}).get("content")
    check(bool(lp) and lp[0]["logprob"] == -1.25 and lp[1]["logprob"] == -0.5,
          "带 logprobs ⇒ 合并进 delta", str(lp)[:80] if lp else "无")
    check(len(lp[0]["top_logprobs"]) == 2, "top_logprobs 带出 K+1 项（自己 + K 个）",
          f"{len(lp[0]['top_logprobs'])} 项")
    check("logprobs" not in out[1]["choices"][0], "第二条 delta 不重复挂")

    # 不带 logprobs ⇒ 逐字透传
    svc = _svc([(42, "LP -1.0 42:-1.0\n")])
    out = list(S.openai_chunks(svc, {}))
    check(all("logprobs" not in c["choices"][0] for c in out), "不带 logprobs ⇒ 不注入字段")

    # 失败：只有 nan vs 一条都没有
    for lp_lines, label in (
        ([(42, "LP nan\n")], "只有 nan"),
        ([], "一条都没有"),
    ):
        svc = _svc(list(lp_lines))
        L.WARNED["missing_lp"] = False
        out = list(S.openai_chunks(svc, {"logprobs": True}))
        check(all("logprobs" not in c["choices"][0] for c in out),
              f"{label} ⇒ 不注入空字段（而不是假装有）")

    # 结束 token 的 LP 要丢掉
    svc = _svc([(99, "LP -0.1 99:-0.1\n"), (42, "LP -1.0 42:-1.0\n")])
    svc.stop_ids = {99}
    out = list(S.openai_chunks(svc, {"logprobs": True}))
    lp = (out[0]["choices"][0].get("logprobs") or {}).get("content")
    check(bool(lp) and len(lp) == 1 and lp[0]["token"] == "<42>",
          "stop token 的 LP 被丢掉", str(lp)[:60] if lp else "无")


def main() -> int:
    print("== 1. 引擎补丁锚点（真实源码，只读）==")
    test_engine_anchors()
    print("\n== 2. top_logprobs 语义 ==")
    test_top_logprobs()
    print("\n== 3. serve/server.py 的扩展点被替换 ==")
    test_patch_installed()
    print("\n== 4. _pump 的 LP/T 对齐 ==")
    test_pump_alignment()
    print("\n== 5. openai_chunks 端到端 ==")
    test_openai_chunks()
    print()
    if FAILURES:
        print(f"失败 {len(FAILURES)} 项：" + "; ".join(FAILURES))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
