"""训练数据：一个能「换个地方就换一份数据」的注册表，默认是**改变身份**。

为什么默认是它：这是唯一一个「答案唯一、一步就能看出学没学会」的任务 —— base 模型对
「你是谁」从不回答 Strata（实测 `我是通义千问（Qwen），是由`），训练日志里的
`logp('Strata' | '你是谁')` 就是干净判据。其余数据集（jsonl / 预训练块）都用同一条
编码路径，所以换数据不会换掉训练逻辑本身。

一份数据 = `[(ids, plen)]`：`ids` 是整条序列，`plen` 之后（含）才算 loss。
`plen=1` 表示「整块全算」（预训练形状）。

用法（`--data` 的写法）：
    identity                内置身份数据，目标词默认 Strata
    identity:Qwen           同上，换个目标词（探针与答案一起换）
    jsonl:/path/x.jsonl     每行 {"prompt":…, "answer":…}
    pretrain:/path/x.npy    prep_pretrain.py 的 uint32 [n, seqlen]
    /path/x.jsonl           扩展名即类型，等价于上面两种带前缀的写法
    list                    打印所有数据源
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class Prepared:
    """编码完的数据 + 它是怎么来的（打印/日志用，避免"我到底训的哪份"）。"""

    enc: list[tuple[list[int], int]]
    source: str
    tokens: int            # 每条样本平均算 loss 的 token 数（吞吐口径的分母）
    blocks: int = 0        # pretrain 才有：原始块数


SPECS = {
    "identity": "内置身份数据（默认）：把「你是谁」的答案改成 --word 指定的词",
    "jsonl": "每行 {\"prompt\":…, \"answer\":…}，只对 answer 段算 loss",
    "pretrain": "prep_pretrain.py 产出的 uint32 [n, seqlen] npy，整块全 token 算 loss",
}


def resolve(spec: str) -> tuple[str, str]:
    """`--data` → (kind, arg)。容错：带路径的裸值按扩展名判类型。"""
    spec = (spec or "identity").strip()
    if ":" in spec and spec.split(":", 1)[0] in SPECS:
        kind, arg = spec.split(":", 1)
        return kind, arg
    if spec == "identity":
        return "identity", ""
    low = spec.lower()
    if low.endswith(".jsonl") or low.endswith(".json"):
        return "jsonl", spec
    if low.endswith(".npy"):
        return "pretrain", spec
    if os.path.sep in spec or "/" in spec:
        raise SystemExit(f"--data {spec}: 不认识这个路径（要 .jsonl 或 .npy），"
                         f"或写成 jsonl:{spec} / pretrain:{spec}")
    raise SystemExit(f"--data {spec}: 不认识的数据源；可用：{', '.join(SPECS)}")


def build(spec: str, tok, word: str = "Strata", seqlen: int = 2048,
          log=print) -> Prepared:
    kind, arg = resolve(spec)
    if kind == "identity":
        from sft import builtin_data, encode

        data = builtin_data(word)
        enc = [encode(tok, q, a) for q, a in data]
        return Prepared(enc, f"identity(word={word})", _mean_tokens(enc), len(enc))

    if kind == "jsonl":
        from sft import encode

        rows = [json.loads(line) for line in open(arg, encoding="utf-8") if line.strip()]
        if not rows:
            raise SystemExit(f"--data {arg}: 空的")
        odd = [r for r in rows if "prompt" not in r or "answer" not in r]
        if odd:
            raise SystemExit(f"--data {arg}: 有 {len(odd)} 行缺 prompt/answer，第一行是 {odd[0]!r}")
        enc = [encode(tok, r["prompt"], r["answer"]) for r in rows]
        return Prepared(enc, f"jsonl:{arg}", _mean_tokens(enc), len(enc))

    if kind == "pretrain":
        import numpy as np

        blocks = np.load(arg)
        if blocks.ndim != 2:
            raise SystemExit(f"--data {arg}: 期望 [n, seqlen] 的二维块，实际 {blocks.shape}")
        if blocks.shape[1] != seqlen:
            log(f"[data] 块长 {blocks.shape[1]} 与 --seqlen {seqlen} 不一致：按块的为准")
        enc = [(list(map(int, blocks[i])), 1) for i in range(len(blocks))]
        ntok = sum(len(b) for b, _ in enc) // max(1, len(enc))
        return Prepared(enc, f"pretrain:{arg}", ntok, len(enc))

    raise SystemExit(f"--data: 未实现 {kind}")       # resolve() 已经挡过，这里只是兜底


def _mean_tokens(enc: list[tuple[list[int], int]]) -> int:
    """每条样本算 loss 的平均 token 数（plen 之后含 plen）。吞吐口径统一用它。"""
    if not enc:
        return 0
    return round(sum(len(b) - (p - 1) for b, p in enc) / len(enc))
