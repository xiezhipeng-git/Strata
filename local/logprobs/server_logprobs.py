"""给 Strata 的 `/v1/chat/completions` 加 OpenAI 的 `logprobs` / `top_logprobs` /
`allowed_token_ids` —— **独立文件**，`serve/server.py` 一行都不改。

## 这是什么

`sycl/serve/server_intel.py::install_logprobs()` 的 **CUDA 版 + allowed_token_ids 版**：
同样从外部 patch `serve/server.py` 的扩展点（Strata 自己的做法，见 server.py 顶部
*"Nothing in `serve/` is edited … so upstream merges stay clean"*），补上三件事：

1. 把请求里的 `logprobs` / `top_logprobs` / `allowed_token_ids` 译成引擎 `--serve`
   协议的新键 `logprobs=K` / `allowed=a,b,c`（**旧引擎会跳过未知键**，所以向后兼容）；
2. 把引擎多吐的 `LP <logprob> <id>:<logprob> ...` 行接成 OpenAI 的 `logprobs` 形状；
3. `allowed_token_ids` 语义与 vLLM 对齐：**只回这些 id 的 logprob，并在它们之上归一化**
   （AnyJev 的 `VLLMBackend` 正是靠这一点拿到「候选动作上的概率分布」）。

## 前提：引擎要吐 `LP` 行

server 侧只负责「接线」。**引擎二进制必须是用 `local/logprobs/build_engine.py` 生成、
编出来的 `strata-lp`**，否则一条 `LP` 行都不会有 —— 那时本文件会打一条**指名道姓**的
错误（说清是哪个引擎路径、缺什么），而不是静默返回空 logprobs（空看起来像「模型很自信」，
会直接把下游的置信度门控带歪）。

## 用法

    python local/logprobs/server_logprobs.py --config strata-<model>.json --port 8080 [...]
    python local/logprobs/server_logprobs.py --check-engine

参数与 `serve/server.py` 完全一致（原样透传）。
⚠️ 本地实验用：Kaggle 那条线走 SGLang/vLLM，不经过 Strata。
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

try:                                  # 只有装补丁 / 起服务时才真的需要 serve.server
    import serve.server as S          # noqa: E402
except Exception as _exc:             # 缺依赖（jinja2 等）时仍可被 import 做单元自检
    S = None
    _SERVE_IMPORT_ERROR = _exc
else:
    _SERVE_IMPORT_ERROR = None


def _need_serve():
    if S is None:
        raise RuntimeError(
            f"导入 serve.server 失败（{type(_SERVE_IMPORT_ERROR).__name__}: "
            f"{_SERVE_IMPORT_ERROR}）⇒ 装不了 logprobs 补丁、也起不了服务。"
            "自检 pure 函数不受影响。")


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


def allowed_ids(req: dict) -> list[int]:
    """`allowed_token_ids`：vLLM 语义，只回这些 id 的 logprob 并在其上归一化。"""
    ids = req.get("allowed_token_ids")
    if not isinstance(ids, (list, tuple)):
        return []
    out: list[int] = []
    for value in ids:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            out.append(int(value))
    return out


def wants_logprobs(req: dict) -> bool:
    return top_logprobs(req) is not None


def engine_patched() -> bool:
    """源码侧是否已生成补丁副本（只能证明「生成过」，不能证明「binary 是它编的」）。"""
    return (Path(__file__).resolve().parent / "build" / "generate_logprobs.cpp").is_file()


WARNED: dict[str, bool] = {"missing_lp": False}


def _sampling_keys(orig):
    def sampling_keys(sampling):
        req = sampling or {}
        base = orig(sampling)
        if not isinstance(base, str):        # 上游若把返回值改成 list，别在这里炸
            base = " ".join(str(x) for x in base)
        k = top_logprobs(req)
        if k is None:
            return base
        tail = f" logprobs={k}"
        ids = allowed_ids(req)
        if ids:
            # 逗号分隔；引擎的 unknown-keys-are-skipped 老协议对旧二进制透明
            tail += " allowed=" + ",".join(str(i) for i in ids)
        return base + tail

    return staticmethod(sampling_keys)


def _generate(orig):
    def gen(self, *a, **kw):
        self.__dict__.setdefault("lp_lines", collections.deque()).clear()   # 只要这个请求的
        return orig(self, *a, **kw)

    return gen


def _pump(orig_lines_put):
    """StrataEngine._pump：`LP` 行不排队，紧挨着它那个 `T` 行一起收。"""
    def _pump(self):
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

    return _pump


def _entry(tok, t, line):
    """一行 `LP` → OpenAI 的一个 logprobs 条目（含 top_logprobs）。

    额外带一个 `id` 字段（token id）：OpenAI 的形状里没有它，但**多一个键对任何
    忽略未知键的消费方都无害**，而下游（AnyJev 的后端）按字符串反查 id 在字节级
    tokenizer 上会踩坑 —— 有权威 id 就不必猜。
    """
    f = line.split()
    if len(f) < 2 or f[1] == "nan":
        return None

    def item(i, lp):
        b = tok.token_bytes(i) if hasattr(tok, "token_bytes") else tok.decode([i]).encode()
        return {"token": b.decode("utf-8", "replace"), "logprob": lp,
                "bytes": list(b), "id": int(i)}

    e = item(t, float(f[1])) if t is not None else {"token": "", "logprob": float(f[1]),
                                                    "bytes": [], "id": None}
    e["top_logprobs"] = [item(int(i), float(v)) for i, _, v in (x.partition(":") for x in f[2:])]
    return e


def _openai_chunks(orig):
    def openai_chunks(svc, req, *a, **kw):
        if not wants_logprobs(req):
            yield from orig(svc, req, *a, **kw)
            return
        eng = svc.engine
        allowed = set(allowed_ids(req))
        seen_line = False       # 见过 LP 行（哪怕内容是 nan）⇒ 引擎是打过补丁的
        seen_usable = False     # 其中有能用的
        for c in orig(svc, req, *a, **kw):
            lp = getattr(eng, "lp_lines", None)
            if c and c.get("choices") and lp:
                got = []
                while lp:
                    t, line = lp.popleft()
                    seen_line = True
                    if t in svc.stop_ids:      # 结束 token 不进 OpenAI 的 logprobs
                        continue
                    e = _entry(svc.tok, t, line)
                    if e is None:
                        continue
                    if allowed:
                        # vLLM 语义：只保留点名的那几个 id，并在它们之上重新归一化
                        keep = [x for x in e["top_logprobs"] if x.get("id") in allowed]
                        if not keep:
                            continue
                        import math
                        mx = max(x["logprob"] for x in keep)
                        se = sum(math.exp(x["logprob"] - mx) for x in keep)
                        lse = mx + math.log(se)
                        for x in keep:
                            x["logprob"] = x["logprob"] - lse
                        e["top_logprobs"] = keep
                    got.append(e)
                if got:
                    seen_usable = True
                    d = c["choices"][0].get("delta") or {}
                    key = ("reasoning_content"
                           if d.get("reasoning_content") and not d.get("content") else "content")
                    c["choices"][0]["logprobs"] = {key: got}
            yield c
        # 要了 logprobs 却拿不到 ⇒ 说清是哪一种、缺哪个文件，别让「空」看起来像「很自信」
        if not seen_usable and not WARNED["missing_lp"]:
            WARNED["missing_lp"] = True
            if seen_line:
                print("[logprobs] 收到 `LP` 行但全是 `nan` ⇒ 引擎的 copy_logits 失败"
                      "（那一行没有 head logits）。查是不是非采样路径 / 层切分下拿不到 head。",
                      file=sys.stderr)
            elif not engine_patched():
                print("[logprobs] 请求要了 logprobs，但没收到任何 `LP` 行：**当前引擎二进制不是 "
                      "strata-lp**。用 `python local/logprobs/build_engine.py --emit` 生成补丁副本、"
                      "编出 strata-lp，再用本脚本以 `--engine <strata-lp>` 起服务。",
                      file=sys.stderr)
            else:
                print("[logprobs] 补丁源码已生成，但引擎二进制仍不吐 `LP` 行 ⇒ "
                      "**跑着的引擎不是用补丁副本编的那个**。确认 server 的引擎路径"
                      "（--engine / 配置 json 里的 exe）指向 strata-lp。", file=sys.stderr)

    return openai_chunks


def _openai_collect(orig):
    def openai_collect(cs):
        merged = {"content": [], "reasoning_content": []}

        def tap():
            for c in cs:
                if c and c.get("choices") and c["choices"][0].get("logprobs"):
                    for k, v in c["choices"][0]["logprobs"].items():
                        merged[k].extend(v)
                yield c

        out = orig(tap())
        if merged["content"] or merged["reasoning_content"]:
            out["choices"][0]["logprobs"] = (
                {k: v for k, v in merged.items() if v} | {"content": merged["content"]}
            )
        return out

    return openai_collect


def install_logprobs() -> None:
    """把 `LP` 行接成 OpenAI 的 `logprobs`；`serve/server.py` 一行不改。"""
    _need_serve()
    Eng = S.StrataEngine
    Eng.sampling_keys = _sampling_keys(Eng.sampling_keys)
    Eng.generate = _generate(Eng.generate)
    Eng._pump = _pump(None)
    S.openai_chunks = _openai_chunks(S.openai_chunks)
    S.openai_collect = _openai_collect(S.openai_collect)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--check-engine", action="store_true", help="只看补丁副本是否已生成，不起服务")
    known, rest = ap.parse_known_args(argv)

    if known.check_engine:
        state = "已生成" if engine_patched() else "**未生成**"
        print(f"local/logprobs/build/generate_logprobs.cpp：{state}")
        if not engine_patched():
            print("  ⇒ 跑 `python local/logprobs/build_engine.py --emit` 生成补丁副本，"
                  "编出 strata-lp 后再起服务。")
        return 0 if engine_patched() else 2

    if not engine_patched():
        print("[logprobs] 注意：还没有生成补丁副本 ⇒ 请求里的 `logprobs` 会是空的。"
              "（`python local/logprobs/build_engine.py --emit` + 编 strata-lp）",
              file=sys.stderr)
    _need_serve()
    install_logprobs()
    return S.main(rest)


if __name__ == "__main__":
    sys.exit(main())
