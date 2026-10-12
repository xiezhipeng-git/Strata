"""在 **native(CUDA)** 引擎上给 `/v1/chat/completions` 加 OpenAI 的 `logprobs` / `top_logprobs`。

## 这是什么

`sycl/serve/server_intel.py::install_logprobs()` 的 **CUDA 版**：同样从外部 monkey-patch
`serve/server.py`，**一行都不改共享源码**（沿用 Strata 自己的做法 —— *"Nothing in `serve/` is
edited … so upstream merges stay clean"*）。

## 前提：引擎要吐 `LP` 行

server 侧只做「把 `LP` 行接成 OpenAI 的 `logprobs`」。**native 引擎默认不吐 `LP` 行**
（只有 `T <id>`），所以要先打引擎补丁：

    python local/apply_engine_logprobs.py --apply      # 锚点式，可 --revert
    # 然后重编译引擎（见 docs/INSTALL.md）

没打补丁也能起服务，但请求里带 `logprobs` 时不会有 `logprobs` 字段回来 —— 本文件会在
第一次遇到这种情况时**打一条明确的一行警告**（而不是静默返回空）。

## 用法

    python local/server_logprobs.py --engine strata --config strata-<model>.json --port 8080 [...]
    python local/server_logprobs.py --check-engine          # 只看引擎补丁状态，不起服务

参数与 `serve/server.py` 完全一致（原样透传）。⚠️ 这是**本地实验用**：Kaggle 那条线走
SGLang/vLLM，不经过 Strata。
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import serve.server as S  # noqa: E402


def top_logprobs(req: dict):
    """引擎 `logprobs=K` 的 K；请求没要就返回 None。

    OpenAI 形状：`logprobs: true` + `top_logprobs: 0..20`；兼容老的 completions 形状
    `logprobs: N`（取 N）。与 `sycl/serve/server_intel.py` 逐字同义。
    """
    lp = req.get("logprobs")
    if lp is True:
        k = req.get("top_logprobs")
        return max(0, min(20, int(k))) if isinstance(k, int) and not isinstance(k, bool) else 0
    if isinstance(lp, int) and not isinstance(lp, bool) and lp >= 0:
        return min(20, lp)
    return None


def engine_patched() -> bool:
    """源码侧是否已打 `local/apply_engine_logprobs.py` 的补丁。"""
    src = REPO / "src" / "program" / "generate.cpp"
    if not src.is_file():
        return False
    return "local/apply_engine_logprobs.py" in src.read_text(encoding="utf-8", errors="replace")


WARNED = {"missing_lp": False}


def install_logprobs() -> None:
    """把 `LP` 行接成 OpenAI 的 `logprobs`；`serve/server.py` 一行不改。"""
    Eng = S.StrataEngine

    def _pump(self):
        # serve/server.py 的 StrataEngine._pump，加一条：LP 行**不排队**，它紧挨着它那个 T 行。
        proc, lines = self.proc, self.lines
        lp = self.__dict__.setdefault("lp_lines", collections.deque())
        last_t = None
        for line in proc.stdout:
            if line.startswith("LP "):
                lp.append((last_t, line))
                continue
            if line.startswith("T "):
                try:
                    last_t = int(line[2:])
                except ValueError:
                    last_t = None
            lines.put(line)
        if self.proc is proc:
            self.ended = True
        lines.put(None)

    Eng._pump = _pump

    keys = Eng.sampling_keys

    def sampling_keys(sampling):
        k = top_logprobs(sampling or {})
        return keys(sampling) + (f" logprobs={k}" if k is not None else "")

    Eng.sampling_keys = staticmethod(sampling_keys)

    generate = Eng.generate

    def gen(self, *a, **kw):
        self.__dict__.setdefault("lp_lines", collections.deque()).clear()   # 只要这个请求的
        return generate(self, *a, **kw)

    Eng.generate = gen

    def entry(tok, t, line):
        f = line.split()
        if len(f) < 2 or f[1] == "nan":
            return None

        def item(i, lp):
            b = tok.token_bytes(i) if hasattr(tok, "token_bytes") else tok.decode([i]).encode()
            return {"token": b.decode("utf-8", "replace"), "logprob": lp, "bytes": list(b)}

        e = item(t, float(f[1])) if t is not None else {"token": "", "logprob": float(f[1]), "bytes": []}
        e["top_logprobs"] = [item(int(i), float(v)) for i, _, v in (x.partition(":") for x in f[2:])]
        return e

    chunks = S.openai_chunks

    def openai_chunks(svc, req, *a, **kw):
        if top_logprobs(req) is None:
            yield from chunks(svc, req, *a, **kw)
            return
        eng = svc.engine
        seen_line = False       # 见到过 LP 行（哪怕内容是 nan）⇒ 引擎已打补丁
        seen_usable = False     # 其中有能用的
        for c in chunks(svc, req, *a, **kw):
            lp = getattr(eng, "lp_lines", None)
            if c and c.get("choices") and lp:
                got = []
                while lp:
                    t, line = lp.popleft()
                    seen_line = True
                    if t in svc.stop_ids:      # 结束 token 不进 OpenAI 的 logprobs
                        continue
                    e = entry(svc.tok, t, line)
                    if e is not None:
                        got.append(e)
                if got:
                    seen_usable = True
                    d = c["choices"][0].get("delta") or {}
                    key = ("reasoning_content"
                           if d.get("reasoning_content") and not d.get("content") else "content")
                    c["choices"][0]["logprobs"] = {key: got}
            yield c
        # 要了 logprobs 却拿不到可用的 LP 行 ⇒ 说清是哪一种，别让「空」看起来像「模型很自信」。
        if not seen_usable and not WARNED["missing_lp"]:
            WARNED["missing_lp"] = True
            if seen_line:
                print("[server_logprobs] 收到了 `LP` 行但全是 `nan` ⇒ 引擎的 copy_logits 失败"
                      "（那一行没有 head logits）。检查是不是非采样路径 / 层切分下拿不到 head。",
                      file=sys.stderr)
            else:
                print("[server_logprobs] 请求要了 logprobs，但引擎一条 `LP` 行都没吐 ⇒ native 引擎还没打补丁。"
                      "跑 `python local/apply_engine_logprobs.py --apply` 后重编译引擎。", file=sys.stderr)

    S.openai_chunks = openai_chunks

    collect = S.openai_collect

    def openai_collect(cs):
        merged = {"content": [], "reasoning_content": []}

        def tap():
            for c in cs:
                if c and c.get("choices") and c["choices"][0].get("logprobs"):
                    for k, v in c["choices"][0]["logprobs"].items():
                        merged[k].extend(v)
                yield c

        out = collect(tap())
        if merged["content"] or merged["reasoning_content"]:
            out["choices"][0]["logprobs"] = (
                {k: v for k, v in merged.items() if v} | {"content": merged["content"]}
            )
        return out

    S.openai_collect = openai_collect


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--check-engine", action="store_true",
                    help="只看 native 引擎补丁状态，不起服务")
    known, rest = ap.parse_known_args(argv)

    if known.check_engine:
        state = "已打" if engine_patched() else "**未打**"
        print(f"src/program/generate.cpp：{state}（标记 local/apply_engine_logprobs.py）")
        if not engine_patched():
            print("  ⇒ 跑 `python local/apply_engine_logprobs.py --apply` 再重编译引擎，"
                  "否则请求里的 logprobs 会返回空。")
        return 0 if engine_patched() else 2

    if not engine_patched():
        print("[server_logprobs] 注意：native 引擎还没打 logprobs 补丁 ⇒ 请求里的 `logprobs` 会是空的。"
              "（`python local/apply_engine_logprobs.py --apply` 后重编译）", file=sys.stderr)
    install_logprobs()
    return S.main(rest)


if __name__ == "__main__":
    sys.exit(main())
