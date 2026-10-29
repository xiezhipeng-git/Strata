"""把「专家路径」和「稠密路径」的墙钟拆开：stub 掉专家 forward 再跑一遍。

背景：T=2048 一步里有 98 万次 kernel 启动、GPU 只忙 5.3 s / 墙钟 17.4 s（74% 空转），
但 CUDA 侧按 op 聚合出来的耗时都很小（最大 _nf4_fwd_kernel 1.29 s），说明成本在
「启动次数 × 每次 Python/派发开销」上。要知道该往哪边砍，就把两条路径分开计时：
  - A：完整前向
  - B：StreamingNF4Experts.forward 直接返回 zeros（专家路径整段消失）
A − B 就是专家路径（含它的 H2D 与逐专家循环）的真实墙钟。
"""
from __future__ import annotations

import argparse
import time

import torch

import loader_qwen4 as L
from ce import token_logprobs
from loader_qwen4 import MmapNgramEmbedding, load_model
from sft import text_model

P = argparse.ArgumentParser()
P.add_argument("--gguf", nargs="+", required=True)
P.add_argument("--pack", required=True)
P.add_argument("--pretrain-data", default="/root/pretrain_blocks.npy")
P.add_argument("--seqlen", type=int, default=2048)
P.add_argument("--warmup", type=int, default=1)
P.add_argument("--rep", type=int, default=3)
a = P.parse_args()

print("loading ...", flush=True)
model, _ = load_model(a.gguf, a.pack)
model.eval()
model.config.use_cache = False
body, head = text_model(model), model.lm_head

import numpy as np

blocks = np.load(a.pretrain_data, mmap_mode="r")
ids = torch.from_numpy(np.asarray(blocks[0][: a.seqlen], dtype=np.int64)).unsqueeze(0).cuda()
tp = L._TIERED
print(f"seqlen={ids.shape[1]}  vram={tp.gpu_blocks} pinned={tp.cpu_pin_blocks} "
      f"page={tp.n_blocks - tp.gpu_blocks - tp.cpu_pin_blocks}", flush=True)

# PLE 的 gather 是 CPU 侧 Python 循环（每行一次 gguf dequantize），CUDA profiler 看不见，
# 单独用墙钟量，并数一下每步到底 gather 多少行。
PLE = {"s": 0.0, "n": 0, "rows": 0}
_ple_fwd = MmapNgramEmbedding.forward


def ple_fwd(self, input_ids, past_key_values):
    t0 = time.perf_counter()
    out = _ple_fwd(self, input_ids, past_key_values)
    PLE["s"] += time.perf_counter() - t0
    PLE["n"] += 1
    PLE["rows"] += out.shape[0] * out.shape[1]
    return out


MmapNgramEmbedding.forward = ple_fwd

labels = ids[0, 1:]
_orig_exp = L.StreamingNF4Experts.forward


LAST = {"v": 0.0}


def run(tag, reps):
    PLE["s"] = PLE["n"] = PLE["rows"] = 0
    L._XFER["bytes"] = 0
    with torch.no_grad():
        for _ in range(a.warmup):
            h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
            LAST["v"] = float(-token_logprobs(h, head, labels, chunk=128).mean())
        torch.cuda.synchronize()
        PLE["s"] = PLE["n"] = PLE["rows"] = 0
        L._XFER["bytes"] = 0
        t0 = time.perf_counter()
        for _ in range(reps):
            h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
            LAST["v"] = float(-token_logprobs(h, head, labels, chunk=128).mean())
            torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / reps
    print(f"\n=== {tag} ===", flush=True)
    print(f"  墙钟 {wall:6.2f} s/步  {ids.shape[1] / wall:6.1f} tok/s", flush=True)
    print(f"  PLE {PLE['s'] / reps:6.2f} s/步  {PLE['n'] / reps:.0f} 次/步  "
          f"{PLE['rows'] / reps:.0f} 行 | 每行 {1e6 * PLE['s'] / max(PLE['rows'], 1):.1f} µs", flush=True)
    print(f"  H2D {L._XFER['bytes'] / 2**30 / reps:6.2f} GiB/步", flush=True)
    print(f"  loss {LAST['v']:.10f}", flush=True)
    return wall


t_full = run("A 完整前向", a.rep)


def stub(self, hidden_states, top_k_index, top_k_weights):
    return torch.zeros_like(hidden_states)


L.StreamingNF4Experts.forward = stub
t_dense = run("B 专家整段 stub（只剩稠密 + 注意力 + PLE + head）", a.rep)
print(f"\n=== 结论 ===\n  专家路径 {t_full - t_dense:.2f} s/步（{100 * (t_full - t_dense) / t_full:.0f}%）"
      f" | 稠密路径 {t_dense:.2f} s/步（{100 * t_dense / t_full:.0f}%）", flush=True)
