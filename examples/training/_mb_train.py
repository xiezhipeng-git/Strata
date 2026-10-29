"""反传一个稳态步的账：前向 / 反传 / 重算各占多少，以及「GPU 忙」与「墙钟」的缺口。

为什么需要它：`bench_cmp` 报的 18.9 s/步 ÷ 4.4 s 前向 = 4.3×，而「ckpt = 1 次存输入 +
1 次重算 + 反传 0.74×」只解释 2.7×。差的那 ~7 s 是这一轮要归因的对象，靠换配置试不出来
（前四项嫌疑已经被实测排除，见 BENCH-SFT-VS-MEZO.md §7.7/§7.8），必须分开计时：

  fwd(no_grad)   ckpt 下第一次前向：不留激活、专家走 staging 视图
  fwd+ckpt+bwd   完整一步（三个 CUDA event 对）
  bwd            由 (整步 − 前向) 反推，不再单独插桩，避免打断流水

`--prof` 再跑一个 CUDA-only profiler 步，按 op 聚合（带 CPU activity 会聚合 300 万个事件，
实测卡十几分钟，所以只收 CUDA）。

用法：
  python3 _mb_train.py --gguf <分片…> --pack /root/experts_nf4.bin \
      --pretrain-data /root/pretrain_blocks.npy --seqlen 2048 --steps 4 [--prof]
"""
from __future__ import annotations

import argparse
import time

import torch

from ce import token_logprobs
from loader_qwen4 import load_model
from lora import DEFAULT_TARGETS, freeze_all, inject_lora, trainable_summary
from sft import text_model

P = argparse.ArgumentParser()
P.add_argument("--gguf", nargs="+", required=True)
P.add_argument("--pack", required=True)
P.add_argument("--tokenizer", default="")
P.add_argument("--pretrain-data", default="/root/pretrain_blocks.npy")
P.add_argument("--seqlen", type=int, default=2048)
P.add_argument("--steps", type=int, default=4, help="稳态步数（前 1 步当冷启动丢掉）")
P.add_argument("--rank", type=int, default=16)
P.add_argument("--alpha", type=int, default=32)
P.add_argument("--lr", type=float, default=1e-4)
P.add_argument("--no-ckpt", action="store_true", help="对照：不开 ckpt（长序列会 OOM）")
P.add_argument("--prof", action="store_true", help="额外给一个 CUDA-only profiler 步")
P.add_argument("--top", type=int, default=24)
a = P.parse_args()

print("loading ...", flush=True)
t0 = time.time()
model, _ = load_model(a.gguf, a.pack)
print(f"loaded {time.time() - t0:.0f}s  {torch.cuda.memory_allocated() / 2**30:.1f} GiB",
      flush=True)

model.train()                       # ckpt 只在 train 下生效
model.config.use_cache = False
freeze_all(model)
n = inject_lora(model, targets=DEFAULT_TARGETS, r=a.rank, alpha=a.alpha)
print(f"lora {n} modules, trainable {trainable_summary(model)['trainable']:,}", flush=True)
if not a.no_ckpt:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lm = getattr(getattr(model, "model", None), "language_model", None)
    if lm is not None:
        lm.gradient_checkpointing = True
print(f"gradient checkpointing: {'OFF' if a.no_ckpt else 'on'}", flush=True)

body, head = text_model(model), model.lm_head
import numpy as np

blocks = np.load(a.pretrain_data, mmap_mode="r")
ids = torch.from_numpy(np.asarray(blocks[0][:a.seqlen], dtype=np.int64)).unsqueeze(0).cuda()
labels = ids[0, 1:]
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr)


def forward_only() -> None:
    """ckpt 下单次前向的样子：no_grad、不建图。"""
    with torch.no_grad():
        h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
        token_logprobs(h, head, labels, chunk=128).mean()


def step() -> float:
    opt.zero_grad(set_to_none=True)
    h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
    loss = -token_logprobs(h, head, labels, chunk=128).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    return float(loss)


torch.cuda.reset_peak_memory_stats()
print("\n-- 冷启动 --", flush=True)
step()
torch.cuda.synchronize()

# 三个 CUDA event 对：前向、整步。反传 = 整步 − 前向。
ev = [[torch.cuda.Event(enable_timing=True) for _ in range(2)] for _ in range(2)]
times = []
for i in range(a.steps):
    torch.cuda.synchronize()
    t_wall = time.perf_counter()

    if not a.no_ckpt:
        ev[0][0].record()
        forward_only()
        ev[0][1].record()
        torch.cuda.synchronize()
        fwd = ev[0][0].elapsed_time(ev[0][1]) / 1000
    else:
        fwd = float("nan")          # 不开 ckpt 时没有「单次无图前向」这个量，别混报

    ev[1][0].record()
    loss = step()
    ev[1][1].record()
    torch.cuda.synchronize()
    whole = ev[1][0].elapsed_time(ev[1][1]) / 1000
    wall = time.perf_counter() - t_wall
    times.append((fwd, whole, wall, loss))
    print(f"step {i + 1}: fwd {fwd * 1000:7.0f} ms  整步 {whole * 1000:7.0f} ms  "
          f"墙钟 {wall * 1000:7.0f} ms  loss {loss:.4f}  "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

fwd = sum(t[0] for t in times) / len(times)
whole = sum(t[1] for t in times) / len(times)
wall = sum(t[2] for t in times) / len(times)
print(f"\n稳态（{len(times)} 步均值）: 前向 {fwd:.2f} s  整步 {whole:.2f} s  墙钟 {wall:.2f} s",
      flush=True)
if not a.no_ckpt:
    print(f"  反推反传 = 整步 − 前向 = {whole - fwd:.2f} s（= {(whole - fwd) / fwd:.2f} × 前向）",
          flush=True)
    print(f"  整步 = {(whole / fwd):.2f} × 前向；ckpt 的理论值是 1 次存输入 + 1 次重算 + "
          f"反传 0.74× ≈ 2.74×", flush=True)
print(f"  事件记的 GPU 时间 {whole:.2f} s，墙钟 {wall:.2f} s ⇒ 空转 "
      f"{100 * (1 - whole / wall):.0f}%，tok/s {2048 / wall:.0f}", flush=True)

if a.prof:
    print("\n-- CUDA-only profiler：一个稳态步 --", flush=True)
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    tbl = prof.key_averages().table(sort_by="cuda_time_total", row_limit=a.top)
    print(tbl, flush=True)
    evts = prof.events()
    print(f"事件数 {len(evts):,}", flush=True)
