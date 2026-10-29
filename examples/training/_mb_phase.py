"""轻量分段计时：T=2048 前向的墙钟到底花在专家模块还是别处。

不启 profiler（3M 次调用会把 CPU activity 处理成十几分钟）。做法是给
StreamingNF4Experts.forward 套一对 CUDA event，逐层累加；event 之间的间隔包含
CPU 来不及派发造成的 GPU 空转，所以量到的是「这一层的真实占用」，不是纯 kernel 时间。
再对比 STRATA_PACK_RAW_H2D=1/0 两条传输路径，判断传输占多少。
"""
from __future__ import annotations

import argparse
import os
import time

import torch

import loader_qwen4 as L
from ce import token_logprobs
from loader_qwen4 import load_model
from sft import text_model

P = argparse.ArgumentParser()
P.add_argument("--gguf", nargs="+", required=True)
P.add_argument("--pack", required=True)
P.add_argument("--pretrain-data", default="/root/pretrain_blocks.npy")
P.add_argument("--seqlen", type=int, default=2048)
P.add_argument("--warmup", type=int, default=2)
P.add_argument("--rep", type=int, default=3)
a = P.parse_args()

STAT = {"exp_s": 0.0, "exp_n": 0}
_PAIRS: list = []
_orig = L.StreamingNF4Experts.forward


def timed_forward(self, hidden_states, top_k_index, top_k_weights):
    # 不在这里 synchronize：逐层同步会切断跨层流水（实测把专家段从 237 ms/层抬到
    # 369 ms/层，1.56×）。只记 event 对，整步结束一次同步后再读 elapsed_time。
    e0 = torch.cuda.Event(True)
    e1 = torch.cuda.Event(True)
    e0.record()
    out = _orig(self, hidden_states, top_k_index, top_k_weights)
    e1.record()
    _PAIRS.append((e0, e1))
    STAT["exp_n"] += 1
    return out


def flush_pairs():
    torch.cuda.synchronize()
    STAT["exp_s"] += sum(a.elapsed_time(b) for a, b in _PAIRS) / 1e3
    _PAIRS.clear()


L.StreamingNF4Experts.forward = timed_forward

print("loading ...", flush=True)
model, _ = load_model(a.gguf, a.pack)
model.eval()
model.config.use_cache = False
body, head = text_model(model), model.lm_head

import numpy as np

blocks = np.load(a.pretrain_data, mmap_mode="r")
ids = torch.from_numpy(np.asarray(blocks[0][: a.seqlen], dtype=np.int64)).unsqueeze(0).cuda()
tp = L._TIERED
print(f"seqlen={ids.shape[1]}  vram_blocks={tp.gpu_blocks}  cpu_blocks={tp.n_blocks - tp.gpu_blocks}"
      f"  pinned={tp.cpu_pinned}", flush=True)


def run(tag, reps):
    STAT["exp_s"] = 0.0
    STAT["exp_n"] = 0
    _PAIRS.clear()
    L._XFER["bytes"] = 0
    with torch.no_grad():
        for _ in range(a.warmup):
            body(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        _PAIRS.clear()
        STAT["exp_n"] = 0
        L._XFER["bytes"] = 0
        t0 = time.perf_counter()
        for _ in range(reps):
            h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
            labels = ids[0, 1:]
            lp = token_logprobs(h, head, labels, chunk=128)
            float(-lp.mean())
            torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / reps
    flush_pairs()
    xfer = L._XFER["bytes"] / 2**30
    exp = STAT["exp_s"] / reps
    print(f"\n=== {tag} ===", flush=True)
    print(f"  墙钟        {wall:7.2f} s/步   {ids.shape[1] / wall:6.1f} tok/s", flush=True)
    print(f"  专家模块    {exp:7.2f} s/步   ({100 * exp / wall:4.1f}%)   "
          f"{STAT['exp_n'] / reps:.0f} 层", flush=True)
    print(f"  其它        {wall - exp:7.2f} s/步", flush=True)
    print(f"  H2D         {xfer:7.2f} GiB/{reps + a.warmup} 前向 → "
          f"{xfer / (reps + a.warmup):5.2f} GiB/步", flush=True)
    # 正确性锚点：同一块数据、同一份 pack，跨进程 loss 必须逐位一致
    print(f"  loss        {float(lp.mean()):.10f}   lp[:4]={[round(float(v), 6) for v in lp[:4]]}",
          flush=True)
    return wall, exp


os.environ["STRATA_PACK_RAW_H2D"] = "1"
run("raw 整块直连（no_grad 默认）", a.rep)
os.environ["STRATA_PACK_RAW_H2D"] = "0"
run("staging 四段拷贝", a.rep)
