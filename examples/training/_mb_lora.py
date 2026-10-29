"""那 4 s 是哪来的：bench_infer 的 9.1 s vs `_mb_split` 的 5.0 s。

两个脚本量的是同一段代码（body + token_logprobs + head，no_grad），差别只有两处：
  - `model.eval()` vs `model.train()`
  - 有没有 `inject_lora`（12 类挂载点）

同一个进程里 2x2 各跑一遍，把差额钉到具体哪一项上。加载一次，四组共享权重。
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import loader_qwen4 as L
from ce import token_logprobs
from loader_qwen4 import load_model
from lora import DEFAULT_TARGETS, freeze_all, inject_lora, trainable_summary
from sft import text_model

P = argparse.ArgumentParser()
P.add_argument("--gguf", nargs="+", required=True)
P.add_argument("--pack", required=True)
P.add_argument("--pretrain-data", default="/root/pretrain_blocks.npy")
P.add_argument("--seqlen", type=int, default=2048)
P.add_argument("--warmup", type=int, default=1)
P.add_argument("--rep", type=int, default=3)
P.add_argument("--rank", type=int, default=16)
P.add_argument("--alpha", type=int, default=32)
P.add_argument("--blocks", action="store_true", help="额外跑一轮「每步换一块」，对齐 bench_infer 的取数方式")
P.add_argument("--passes", type=int, default=1, help="换块那一轮重复几遍")
a = P.parse_args()

print("loading ...", flush=True)
model, _ = load_model(a.gguf, a.pack)
model.config.use_cache = False
body, head = text_model(model), model.lm_head

blocks = np.load(a.pretrain_data, mmap_mode="r")
ids = torch.from_numpy(np.asarray(blocks[0][: a.seqlen], dtype=np.int64)).unsqueeze(0).cuda()
labels = ids[0, 1:]
tp = L._TIERED
print(f"seqlen={ids.shape[1]}  vram={tp.gpu_blocks} pinned={tp.cpu_pin_blocks} "
      f"page={tp.n_blocks - tp.gpu_blocks - tp.cpu_pin_blocks}", flush=True)


def measure(tag: str) -> float:
    with torch.no_grad():
        for _ in range(a.warmup):
            h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
            loss = float(-token_logprobs(h, head, labels, chunk=128).mean())
        torch.cuda.synchronize()
        L._XFER["bytes"] = 0
        t0 = time.perf_counter()
        for _ in range(a.rep):
            h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
            loss = float(-token_logprobs(h, head, labels, chunk=128).mean())
            torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / a.rep
    print(f"{tag:34s} {wall:6.2f} s/步  {ids.shape[1] / wall:7.1f} tok/s  "
          f"H2D {L._XFER['bytes'] / 2**30 / a.rep:6.2f} GiB  loss {loss:.10f}", flush=True)
    return wall


def measure_blocks(tag: str) -> None:
    """每步换一块数据（= bench_infer 的口径），逐步打印，看是不是块相关。

    `--passes 2` 会把同一批块跑两遍：同一块两遍都慢 = 数据相关；只第一遍慢 = 系统噪声。
    """
    order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(0)).tolist()
    per = []
    with torch.no_grad():
        for i in range(a.rep * a.passes + a.warmup):
            bi = order[i % a.rep] if a.passes > 1 else order[i]
            row = torch.from_numpy(np.asarray(blocks[bi], dtype=np.int64)).unsqueeze(0).cuda()
            lab = row[0, 1:]
            L._XFER["bytes"] = 0
            t0 = time.perf_counter()
            h = body(input_ids=row[:, :-1], use_cache=False).last_hidden_state[0]
            loss = float(-token_logprobs(h, head, lab, chunk=128).mean())
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i >= a.warmup:
                per.append(dt)
            print(f"  {tag} 第 {i // max(a.rep, 1) + 1} 遍 块 {bi:5d} {dt:6.2f} s  "
                  f"H2D {L._XFER['bytes'] / 2**30:6.2f} GiB  loss {loss:.6f}", flush=True)
    print(f"{tag:34s} {sum(per) / len(per):6.2f} s/步（换块）  "
          f"{ids.shape[1] * len(per) / sum(per):7.1f} tok/s", flush=True)


model.eval()
t_eval_plain = measure("eval    / 无 LoRA")

model.train()
t_train_plain = measure("train   / 无 LoRA")

# 回到 eval 再挂 LoRA，隔离 train 模式的影响
freeze_all(model)
n = inject_lora(model, targets=tuple(DEFAULT_TARGETS), r=a.rank, alpha=a.alpha)
print(f"inject_lora: {n} 个挂载点, trainable {trainable_summary(model)['trainable']:,}", flush=True)

model.eval()
t_eval_lora = measure("eval    / LoRA")

model.train()
t_train_lora = measure("train   / LoRA（= bench_infer 口径）")

print("\n=== 结论 ===", flush=True)
print(f"  LoRA 附加（eval 下）   {t_eval_lora - t_eval_plain:+.2f} s", flush=True)
print(f"  train 模式附加（无 LoRA）{t_train_plain - t_eval_plain:+.2f} s", flush=True)
print(f"  两者合计               {t_train_lora - t_eval_plain:+.2f} s"
      f"（交叉核对 {t_train_lora - t_eval_plain - (t_eval_lora - t_eval_plain) - (t_train_plain - t_eval_plain):+.2f}）",
      flush=True)

if a.blocks:
    print("", flush=True)
    measure_blocks("train   / LoRA / 换块")
