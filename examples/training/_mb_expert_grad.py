"""一层的「每专家循环」在 no_grad / 建图 / 建图+反传 三种模式下差多少。

回答的是 §7.8 剩下的那笔账：ckpt 步里「重算 + 反传」实测 18.2 s，而按
`1 次重算 + 0.74 次反传 = 1.74 × 前向` 只该 7.6 s。前向本身（no_grad）是 4.4 s，
所以问题不是「前向慢」，而是**带 grad 走同一段循环要贵多少**。

不加载 pack（那是 100 s）：用同形状的 bf16 权重替身，只量**调度与 autograd** 的成本 ——
GEMM 内核本身换成 cuBLAS，所以这里数到的差值纯粹是「多建了多少图节点、多付了多少
Python 分发」。绝对值和真实训练不可比，比值可比。
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

P = argparse.ArgumentParser()
P.add_argument("--hidden", type=int, default=2560)
P.add_argument("--inter", type=int, default=640)
P.add_argument("--experts", type=int, default=512)
P.add_argument("--tokens", type=int, default=2048)
P.add_argument("--top-k", type=int, default=8)
P.add_argument("--reps", type=int, default=3)
a = P.parse_args()

dev = "cuda"
torch.manual_seed(0)
H, I, E, T, K = a.hidden, a.inter, a.experts, a.tokens, a.top_k

hidden = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
top_k_index = torch.randint(0, E, (T, K), device=dev)
top_k_weights = torch.softmax(torch.randn(T, K, device=dev), -1)
gu = [torch.randn(2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(E)]
dn = [torch.randn(H, I, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(E)]
print(f"one layer: {E} experts, T={T}, K={K}, {torch.cuda.memory_allocated() / 2**30:.1f} GiB",
      flush=True)


def routing():
    mask = F.one_hot(top_k_index, num_classes=E).permute(2, 1, 0)
    nz = mask.nonzero()
    counts = torch.bincount(nz[:, 0], minlength=E)
    bounds = torch.cumsum(counts, 0).tolist()
    tok_all = nz[:, 2]
    grp = top_k_weights.reshape(-1).gather(0, tok_all * K + nz[:, 1])
    prev, hits = 0, []
    for e, end in enumerate(bounds):
        if end > prev:
            hits.append(e)
        prev = end
    return bounds, tok_all, grp, hits


bounds, tok_all, grp, hits = routing()
print(f"hit experts: {len(hits)}/{E}", flush=True)


def loop(final):
    for e in hits:
        x = hidden[tok_all[(bounds[e - 1] if e else 0):bounds[e]]]
        gate, up = F.linear(x, gu[e]).chunk(2, dim=-1)
        h = F.linear(F.silu(gate) * up, dn[e])
        final.index_add_(0, tok_all[(bounds[e - 1] if e else 0):bounds[e]],
                         (h * grp[(bounds[e - 1] if e else 0):bounds[e], None]).to(final.dtype))
    return final


def run(mode: str) -> float:
    """mode: nograd | graph | graph+bwd"""
    if mode == "nograd":
        with torch.no_grad():
            final = torch.zeros_like(hidden)
            t0 = time.perf_counter()
            loop(final)
            torch.cuda.synchronize()
            return time.perf_counter() - t0
    # 与真实代码同形：输入需要梯度（它是上一层的输出），final 是普通 zeros 再 index_add_。
    # 让 leaf 自己 requires_grad 会在 index_add_ 上报「in-place on a leaf」—— 那是测法错，不是代码错。
    hidden.requires_grad_(True)
    final = torch.zeros_like(hidden)
    t0 = time.perf_counter()
    loop(final)
    if mode == "graph":
        torch.cuda.synchronize()
        return time.perf_counter() - t0
    g = torch.ones_like(final) * 1e-3
    final.backward(g)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


for mode in ("nograd", "graph", "graph+bwd"):
    run(mode)                                    # warm
    best = min(run(mode) for _ in range(a.reps))
    print(f"{mode:10s} {best * 1000:8.1f} ms   ({best * 1e6 / len(hits):6.1f} us/expert)",
          flush=True)

n = min(run("nograd") for _ in range(a.reps))
g = min(run("graph") for _ in range(a.reps))
b = min(run("graph+bwd") for _ in range(a.reps))
print(f"\n建图 / no_grad = {g / n:.2f}×   反传 / no_grad = {(b - g) / n:.2f}×   "
      f"整步 / no_grad = {b / n:.2f}×", flush=True)
print("判据：如果「建图」就是 2~3×，那 ckpt 的重算贵在 autograd 节点数（每专家 ~8 个），"
      "不是 GEMM 也不是 H2D —— 该动的是每专家的 op 数（融合/分组），不是显存分配。",
      flush=True)
