"""用 profiler 一次性拿到 T=2048 前向的完整归属，不手搓计时点。

回答的问题：拟合出的计算项 15.1 s/step 里，专家模块只解释 5.5 s，其余 ~9.6 s 在哪。
"""
from __future__ import annotations

import argparse
import sys

import torch

from loader_qwen4 import load_model
from sft import text_model

P = argparse.ArgumentParser()
P.add_argument("--gguf", nargs="+", required=True)
P.add_argument("--pack", required=True)
P.add_argument("--pretrain-data", default="/root/pretrain_blocks.npy")
P.add_argument("--seqlen", type=int, default=2048)
P.add_argument("--warmup", type=int, default=1)
P.add_argument("--rep", type=int, default=3)
P.add_argument("--top", type=int, default=28)
a = P.parse_args()

print("loading ...", flush=True)
model, _ = load_model(a.gguf, a.pack)
model.eval()
model.config.use_cache = False
body = text_model(model)
head = model.lm_head

import numpy as np
blocks = np.load(a.pretrain_data, mmap_mode="r")
ids = torch.from_numpy(np.asarray(blocks[0][: a.seqlen], dtype=np.int64)).unsqueeze(0).cuda()

try:
    import causal_conv1d  # noqa: F401
    print("causal_conv1d: 已安装（走优化 kernel）", flush=True)
except Exception:
    print("causal_conv1d: **未安装**（Transformers 回落到 reference PyTorch 实现）", flush=True)

print(f"seqlen={ids.shape[1]}  warmup={a.warmup}", flush=True)
with torch.no_grad():
    for _ in range(a.warmup):
        body(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()

import time

# 墙钟基线（无 profiler 干扰）：这一步同时给出 GPU 忙与 CPU 派发的缺口。
# 分开量 body（transformer）与 head+loss，因为 _mb_phase 显示「非专家」占了 3/4。
from ce import token_logprobs

labels = ids[0, 1:]


def body_only():
    return body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]


def head_loss(h):
    return float(-token_logprobs(h, head, labels, chunk=128).mean())


with torch.no_grad():
    for _ in range(a.warmup):
        head_loss(body_only())
    torch.cuda.synchronize()
    for _ in range(a.rep):
        t0 = time.perf_counter()
        h = body_only()
        torch.cuda.synchronize()
        t_body = time.perf_counter() - t0
        t0 = time.perf_counter()
        head_loss(h)
        torch.cuda.synchronize()
        t_head = time.perf_counter() - t0
        print(f"  body {t_body:6.2f} s | head+loss {t_head:6.2f} s | "
              f"合计 {t_body + t_head:6.2f} s = {ids.shape[1] / (t_body + t_head):6.1f} tok/s",
              flush=True)
wall = t_body + t_head
print(f"\n=== 墙钟 {wall:.2f} s/步（{ids.shape[1] / wall:.1f} tok/s，n={a.rep}）===", flush=True)

from torch.profiler import ProfilerActivity, profile

# 只开 CUDA：带 CPU activity 时 3M 次调用的事件聚合会把 key_averages 拖成十几分钟。
with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA],
                              record_shapes=False, profile_memory=False) as prof:
    body_only()
    torch.cuda.synchronize()

evts = prof.key_averages()
rows = [(e.self_device_time_total / 1e6, e.count, e.key) for e in evts]
rows.sort(reverse=True)
tot = sum(r[0] for r in rows)
print(f"\n=== CUDA 总时间 {tot:.2f} s（T={ids.shape[1]}）===")
print(f"{'s':>8s} {'%':>6s} {'count':>9s}  op")
for t, c, k in rows[: a.top]:
    print(f"{t:8.2f} {100*t/tot:6.1f} {c:9d}  {k[:64]}")

print(f"\n=== 按调用次数（GPU 总时间 {tot:.2f} s，共 {sum(r[1] for r in rows)} 次）===")
bycnt = sorted(rows, key=lambda r: -r[1])
print(f"{'count':>10s} {'s':>8s} {'us/次':>7s}  op")
for t, c, k in bycnt[: a.top]:
    print(f"{c:10d} {t:8.2f} {1e6*t/c:7.2f}  {k[:64]}")

# 按类别聚合（按 key 前缀粗分，够定位方向）
def bucket(k: str) -> str:
    kl = k.lower()
    if "nf4" in kl or "fusednf4" in kl:
        return "① 融合 NF4 GEMM"
    if "index_add" in kl or "index_select" in kl or "gather" in kl or kl.startswith("aten::index"):
        return "② 逐专家 gather/index_add"
    if "where" in kl or "nonzero" in kl or "one_hot" in kl:
        return "③ mask/where/nonzero"
    if "attention" in kl or "flash" in kl or "fmha" in kl or "softmax" in kl:
        return "④ attention"
    if "copy" in kl or "memcpy" in kl or "Memcpy" in k:
        return "⑤ H2D/copy"
    if "elementwise" in kl or "silu" in kl or "mul" in kl or "chunk" in kl:
        return "⑥ 逐元素/silu/mul"
    if "gemm" in kl or "cublas" in kl or "cutlass" in kl or "matmul" in kl or "mm" == kl:
        return "⑦ 其它 GEMM"
    if "norm" in kl or "rms" in kl or "layer_norm" in kl:
        return "⑧ norm"
    return "⑨ 其它"

agg: dict[str, list] = {}
for t, c, k in rows:
    b = bucket(k)
    s = agg.setdefault(b, [0.0, 0])
    s[0] += t
    s[1] += c
print(f"\n=== 按类别 ===")
print(f"{'s':>8s} {'%':>6s} {'count':>9s}  类别")
for b, (t, c) in sorted(agg.items(), key=lambda x: -x[1][0]):
    print(f"{t:8.2f} {100*t/tot:6.1f} {c:9d}  {b}")

print(f"\nGPU 忙 {tot:.2f} s / 墙钟 {wall:.2f} s ⇒ 空转 "
      f"{100 * (1 - tot / wall):.0f}%（GPU 忙 < 墙钟即 CPU 派发是短板）", flush=True)
