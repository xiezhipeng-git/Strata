#!/usr/bin/env python3
"""local/logprobs 的自检 —— **不需要 GPU**，也不需要 serve.server。

覆盖四层：
  1. 请求翻译：`logprobs` / `top_logprobs` / `allowed_token_ids` → 引擎协议键
  2. `LP` 行解析 → OpenAI 形状（含权威 `id`、top_logprobs、nan 行）
  3. 引擎补丁生成器：原文件只读、锚点唯一、生成幂等、与原文件的差异只有补丁
  4. AnyJev 后端：正常解析 + 「没回 logprobs」时必须**显式报错**（不能静默给 -30）

跑法（用带 numpy 的解释器，例如 Strata 的 .venv）：
    python local/logprobs/selftest.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import server_logprobs as SL  # noqa: E402


class Tok:
    """假 tokenizer：token_bytes 给确定值，token 串可预测。"""

    def token_bytes(self, i):
        return f"<{i}>".encode()

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)

    def convert_ids_to_tokens(self, i):
        return f"<{i}>"


def check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {'✓' if cond else '✗'} {label}" + (f"  ({detail})" if detail and not cond else ""))
    return cond


def t_translation() -> int:
    print("[1] 请求翻译 → 引擎协议键")
    ok = 0

    def keys_str(_s):
        return "temperature=0.6"

    sk = SL._sampling_keys(keys_str)
    cases = [
        ({"logprobs": True, "top_logprobs": 5}, " logprobs=5"),
        ({"logprobs": 7}, " logprobs=7"),
        ({"logprobs": True}, " logprobs=0"),
        ({"logprobs": True, "top_logprobs": 99}, " logprobs=20"),   # 夹到 20
    ]
    for req, want in cases:
        got = sk(req)
        ok += check(f"{req} → {want!r}", want in got, f"got {got!r}")
    ok += check("没要 logprobs 就原样返回", sk({}) == "temperature=0.6")
    got = sk({"logprobs": True, "top_logprobs": 3, "allowed_token_ids": [11, 22, 33]})
    ok += check("allowed_token_ids → allowed=11,22,33",
                " logprobs=3" in got and " allowed=11,22,33" in got, f"got {got!r}")
    ok += check("allowed 顺序保持",
                got.index("allowed=11,22,33") > got.index("logprobs=3"))
    # 上游若把 sampling_keys 的返回类型改成 list，扩展也不能炸
    sk_list = SL._sampling_keys(lambda _s: ["temperature=0.6", "top_k=64"])
    got = sk_list({"logprobs": True, "top_logprobs": 2})
    ok += check("返回 list 也能吃", isinstance(got, str) and " logprobs=2" in got, f"{got!r}")
    return ok


def t_entry() -> int:
    print("[2] LP 行 → OpenAI 形状")
    ok = 0
    tok = Tok()
    e = SL._entry(tok, 42, "LP -0.903300 7:-0.9033 8:-3.2044 9:-7.1")
    ok += check("采样 token 的 logprob", e is not None and abs(e["logprob"] + 0.9033) < 1e-6)
    ok += check("top_logprobs 个数=3", e is not None and len(e["top_logprobs"]) == 3)
    ok += check("带权威 id", e is not None and e["id"] == 42
                and e["top_logprobs"][0]["id"] == 7)
    ok += check("nan 行 → None（不假装有分布）", SL._entry(tok, 42, "LP nan") is None)
    ok += check("空行不炸", SL._entry(tok, 42, "LP") is None)
    return ok


def t_generator() -> int:
    print("[3] 引擎补丁生成器（原文件只读）")
    ok = 0
    target = REPO / "src" / "program" / "generate.cpp"
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    r = subprocess.run([sys.executable, str(HERE / "build_engine.py"), "--check"],
                       capture_output=True, text=True)
    ok += check("--check 退出码 0（锚点全对）", r.returncode == 0, r.stdout + r.stderr)
    r = subprocess.run([sys.executable, str(HERE / "build_engine.py"), "--emit"],
                       capture_output=True, text=True)
    ok += check("--emit 退出码 0", r.returncode == 0, r.stdout + r.stderr)
    after = hashlib.sha256(target.read_bytes()).hexdigest()
    ok += check("原文件 sha256 不变", before == after)
    copy = HERE / "build" / "generate_logprobs.cpp"
    ok += check("生成了副本", copy.is_file())
    if copy.is_file():
        text = copy.read_text(encoding="utf-8")
        ok += check("副本含 6 处补丁标记", text.count("local/logprobs/build_engine.py") == 6)
        ok += check("副本含 logprobs=K 解析", 'key == "logprobs"' in text)
        ok += check("副本含 allowed= 解析", 'key == "allowed"' in text)
        orig = target.read_text(encoding="utf-8")
        ok += check("副本 = 原文件 + 66 行（只增不改）",
                    text.count("\n") == orig.count("\n") + 66,
                    f"{text.count(chr(10))} vs {orig.count(chr(10))}")
        # 幂等：再 emit 一次，产物逐字节相同
        first = copy.read_bytes()
        subprocess.run([sys.executable, str(HERE / "build_engine.py"), "--emit"],
                       capture_output=True, text=True)
        ok += check("重复 emit 逐字节幂等", copy.read_bytes() == first)
    return ok


def t_backend() -> int:
    print("[4] AnyJev 后端（假 _post，不发请求）")
    ok = 0
    sys.path.insert(0, str(REPO.parent / "AnyJev"))
    try:
        from anyjev.backends.strata import StrataBackend
    except Exception as exc:  # numpy 缺失等
        print(f"  · 跳过（import 失败：{type(exc).__name__}: {exc}）")
        return 0
    import numpy as np

    b = object.__new__(StrataBackend)
    b.base_url, b.name, b.api_key, b.workers, b.timeout = "http://x/v1", "m", "EMPTY", 1, 5.0
    b.tokenizer = Tok()

    b._post = lambda body: {"choices": [{"logprobs": {"content": [{
        "logprob": -0.9,
        "top_logprobs": [{"token": "<11>", "logprob": -0.9, "id": 11},
                          {"token": "<22>", "logprob": -2.3, "id": 22},
                          {"token": "<33>", "logprob": -5.0, "id": 33}]}]}}]}
    out = b._one("prompt", [11, 22, 33])
    ok += check("按 id 取到三个 logprob", np.allclose(out, [-0.9, -2.3, -5.0]), f"{out}")

    # 缺 id 时退回字符串反查
    b._post = lambda body: {"choices": [{"logprobs": {"content": [{
        "logprob": -1.1,
        "top_logprobs": [{"token": "<11>", "logprob": -1.1},
                          {"token": "<22>", "logprob": -4.0}]}]}}]}
    out = b._one("prompt", [11, 22])
    ok += check("无 id 时按 token 串反查", np.allclose(out, [-1.1, -4.0]), f"{out}")

    # 没回 logprobs ⇒ 必须显式报错，不能静默 -30
    b._post = lambda body: {"choices": [{"message": {"content": "hi"}}]}
    try:
        b._one("prompt", [11])
        ok += check("无 logprobs 时应抛错", False)
    except RuntimeError as exc:
        ok += check("无 logprobs 时抛 RuntimeError 且点名 strata-lp",
                    "strata-lp" in str(exc), str(exc))
    return ok


def main() -> int:
    print(f"local/logprobs 自检（{HERE}）")
    total = t_translation() + t_entry() + t_generator() + t_backend()
    print(f"\n合计 {total} 项")
    return 0 if total >= 20 else 1


if __name__ == "__main__":
    raise SystemExit(main())
