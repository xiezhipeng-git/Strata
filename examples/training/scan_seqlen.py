"""序列长度扫描：把「每步固定开销」与「每 token 边际成本」分开。

background: bench_cmp.py 报的 tok/s 是拿 loss token 数（内置身份数据约 8 个）除以
一步的墙钟时间得到的。短序列下每步的固定开销（fused NF4 专家 dequant、kernel
launch、Python 调度）摊不掉，那个数字没有代表性 —— 它既不代表训练吞吐，也不代表
decode 速度。

这里对同一份权重、同一进程扫 seqlen，拟合

    dt(T) = a + b * T

a = 固定开销（s/步），b = 边际成本（s/token）。长序列等效吞吐 = 1 / b。
T=1 的 dt 就是 a 的直接测量（decode 形态的上界：不含 KV cache 复用）。
"""

from __future__ import annotations

import argparse
import json
import resource
import time

import torch

from lora import DEFAULT_TARGETS, freeze_all, inject_lora, trainable_summary
from loader_qwen4 import load_model
from sft import text_model


def rss_now_gib() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2**30


def fit(xs, ys):
    """最小二乘 y = a + b x。"""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if den == 0:
        return float("nan"), float("nan")
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--lens", default="1,8,32,128,512,1024,2048")
    ap.add_argument("--batches", default="1",
                    help="MoE 的传输成本是 per-expert 而非 per-token：加大 batch 能把同一个"
                         "专家的 GEMM 从 ~80 行撑到几百行，而 H2D 字节数不变")
    ap.add_argument("--modes", default="infer,sft")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--ckpt", action="store_true", help="梯度检查点：长序列反传才放得下")
    ap.add_argument("--out", default="/root/scan_seqlen.json")
    args = ap.parse_args()

    lens = [int(x) for x in args.lens.split(",") if x]
    batches = [int(x) for x in args.batches.split(",") if x]
    modes = [m for m in args.modes.split(",") if m]

    t0 = time.time()
    model, _ = load_model(args.gguf, args.pack)
    load_s = time.time() - t0

    model.train()
    model.config.use_cache = False
    freeze_all(model)
    inject_lora(model, targets=tuple(t for t in args.targets.split(",") if t),
                r=args.rank, alpha=args.alpha)
    s = trainable_summary(model)
    if args.ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lm = getattr(getattr(model, "model", None), "language_model", None)
        if lm is not None:
            lm.gradient_checkpointing = True
        print("gradient checkpointing: on", flush=True)

    pdev, bdev = {}, {}
    for p in model.parameters():
        pdev[str(p.device)] = pdev.get(str(p.device), 0) + p.numel()
    for b in model.buffers():
        bdev[str(b.device)] = bdev.get(str(b.device), 0) + b.numel()
    print(f"loaded {load_s:.0f}s  trainable {s['trainable']:,}", flush=True)
    print("param elems by device:", {k: v / 2**20 for k, v in pdev.items()}, "(Mi elems)", flush=True)
    print("buffer elems by device:", {k: v / 2**20 for k, v in bdev.items()}, "(Mi elems)", flush=True)

    body, head = text_model(model), model.lm_head
    vocab = int(head.weight.shape[0])
    g = torch.Generator(device="cuda").manual_seed(0)

    # 统计每步从 host RAM 搬进 GPU 的字节数：tiered pack 对非 resident 块每次前向都要
    # 重传（pinned memcpy -> H2D，无冷块缓存），_copy_block 是这条路径的唯一入口。
    import loader_qwen4 as _L

    xfer = {"bytes": 0, "blocks": 0, "cpu_s": 0.0}
    _orig_copy = _L._copy_block

    def _counted_copy(src, dsts, sizes):
        t = time.perf_counter()
        r = _orig_copy(src, dsts, sizes)
        xfer["cpu_s"] += time.perf_counter() - t
        xfer["bytes"] += int(src.numel())
        xfer["blocks"] += 1
        return r

    _L._copy_block = _counted_copy

    rows = []
    for mode in modes:
        for B in batches:
            for T in lens:
                ids = torch.randint(0, vocab, (B, T), device="cuda", generator=g)

                def run():
                    if mode == "infer":
                        with torch.no_grad():
                            h = body(input_ids=ids, use_cache=False).last_hidden_state
                            return head(h)
                    out = body(input_ids=ids, use_cache=False).last_hidden_state
                    logits = head(out).float()
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, vocab), ids[:, 1:].reshape(-1))
                    loss.backward()
                    model.zero_grad(set_to_none=True)
                    return loss

            try:
                for _ in range(args.warmup):
                    run()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                dts = []
                xb = 0
                xc = 0.0
                for _ in range(args.reps):
                    torch.cuda.synchronize()
                    xfer["bytes"] = 0
                    xfer["blocks"] = 0
                    _L._XFER["bytes"] = 0
                    xfer["cpu_s"] = 0.0
                    _L._XFER["bytes"] = 0
                    t = time.time()
                    run()
                    torch.cuda.synchronize()
                    dts.append(time.time() - t)
                    xb = xfer["bytes"] + _L._XFER["bytes"]
                    xc = xfer["cpu_s"]
            except torch.OutOfMemoryError:
                print(f"{mode} T={T}: OOM, skip", flush=True)
                torch.cuda.empty_cache()
                continue

            dts.sort()
            dt = dts[len(dts) // 2]
            peak = torch.cuda.max_memory_allocated() / 2**30
            rows.append({"mode": mode, "T": T, "B": B, "dt_s": round(dt, 4),
                         "tok_s": round(B * T / dt, 1), "peak_gib": round(peak, 2),
                         "h2d_gib": round(xb / 2**30, 2), "h2d_blocks": xfer["blocks"],
                         "cpu_copy_s": round(xc, 2)})
            print(f"{mode:5s} B={B} T={T:5d}  {dt*1000:8.1f} ms  {B*T/dt:9.1f} tok/s  "
                  f"peak {peak:5.2f} GiB  H2D {xb/2**30:5.2f} GiB/步 cpu_copy {xc:5.2f}s"
                  f"   (reps {[round(d*1000,1) for d in dts]})", flush=True)
            del ids
            torch.cuda.empty_cache()

    print()
    for mode in modes:
        sub = [r for r in rows if r["mode"] == mode]
        if len(sub) < 2:
            continue
        a, b = fit([r["T"] for r in sub], [r["dt_s"] for r in sub])
        print(f"[{mode}] fit dt = {a*1000:.1f} ms + {b*1000:.4f} ms/token"
              f"   -> 长序列渐近吞吐 {1/b:,.0f} tok/s" if b > 0 else f"[{mode}] fit b<=0")
        print(f"[{mode}] T=1 实测 {sub[0]['dt_s']*1000:.1f} ms（≈固定开销）；"
              f"T={sub[-1]['T']} 实测 {sub[-1]['dt_s']*1000:.1f} ms")
    print()
    print(f"RSS now {rss_now_gib():.2f} GiB", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"lens": lens, "modes": modes, "rows": rows}, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
