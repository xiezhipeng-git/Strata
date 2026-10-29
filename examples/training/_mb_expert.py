"""拆解一层的每专家成本：融合 GEMM vs 循环里的辅助 op（where/gather/index_add）。

目标：回答「7.37 ms/token 里，矩阵乘占多少、Python/torch 调度占多少」。
用真实形状（H=2560, I=640, T=2048, top_k=8, 256 experts）复现 forward 的内层循环。
"""
import sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, "/mnt/d/Admin/Documents/GitHub/Strata-train/examples/training")
from nf4_fused import fused_nf4_linear, _get_code

dev = "cuda"
torch.manual_seed(0)

H, I, T, K, E = 2560, 640, 2048, 8, 256
NEXP = int(sys.argv[1]) if len(sys.argv) > 1 else 1000   # 每层命中的专家数
CODE = _get_code(torch.device("cuda"))


def bench(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return min(ts)


# ---- 真实数据形状 ----
hidden = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
top_k_index = torch.randint(0, E, (T, K), device=dev)      # loader 里是 (T, K)：(T,K,E)->(E,K,T)
top_k_weights = torch.softmax(torch.randn(T, K, device=dev), -1)

mask = F.one_hot(top_k_index, num_classes=E).permute(2, 1, 0)
hits = [int(e[0]) for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero() if int(e[0]) < E]
print(f"T={T} K={K} E={E}: 实际命中 {len(hits)} 个专家/层，测 {NEXP} 个", flush=True)
hits = hits[:NEXP] if len(hits) >= NEXP else (hits * (NEXP // max(1, len(hits)) + 1))[:NEXP]

# 每专家一份随机权重（模拟不同的 pack 块；只测速度，不看数值）
gu_w = [(torch.randint(0, 256, (2 * I, H // 2), device=dev, dtype=torch.uint8),
         torch.randn(2 * I, H // 64, device=dev, dtype=torch.bfloat16)) for _ in range(16)]
dn_w = [(torch.randint(0, 256, (H, I // 2), device=dev, dtype=torch.uint8),
         torch.randn(H, I // 64, device=dev, dtype=torch.bfloat16)) for _ in range(16)]


def loop_full():
    """复现 loader forward 的内层循环（8 个 op/专家）。"""
    final = torch.zeros_like(hidden)
    for i, e in enumerate(hits):
        pos, tok = torch.where(mask[e])
        x = hidden[tok]
        gu, ga = gu_w[i % 16]
        dn, da = dn_w[i % 16]
        gate, up = fused_nf4_linear(x, gu, ga).chunk(2, dim=-1)
        h = F.silu(gate) * up
        h = fused_nf4_linear(h, dn, da)
        h = h * top_k_weights[tok, pos, None]
        final.index_add_(0, tok, h.to(final.dtype))
    return final


def loop_gemm_only():
    """只跑两次融合 GEMM，不跑 where/gather/index_add 等辅助 op。"""
    out = None
    for i, e in enumerate(hits):
        gu, ga = gu_w[i % 16]
        dn, da = dn_w[i % 16]
        x = hidden[:64]
        gate, up = fused_nf4_linear(x, gu, ga).chunk(2, dim=-1)
        h = F.silu(gate) * up
        out = fused_nf4_linear(h, dn, da)
    return out


def loop_aux_only():
    """只跑辅助 op（where/gather/加权/index_add），GEMM 换成一个固定的 [len(tok), H] 占位。"""
    final = torch.zeros_like(hidden)
    for i, e in enumerate(hits):
        pos, tok = torch.where(mask[e])
        x = hidden[tok]
        h = x * top_k_weights[tok, pos, None]
        final.index_add_(0, tok, h.to(final.dtype))
    return final


t_full = bench(loop_full)
t_gemm = bench(loop_gemm_only)
t_aux = bench(loop_aux_only)

per = lambda t: t / NEXP * 1000
print(f"\n每专家（N={NEXP}）:")
print(f"  完整循环        {per(t_full):7.3f} ms   -> {t_full / len(hits):.4f} s/2048tok（按每层命中数放大）")
print(f"  仅两次融合GEMM  {per(t_gemm):7.3f} ms   ({100*t_gemm/t_full:5.1f}%)")
print(f"  仅辅助 op       {per(t_aux):7.3f} ms   ({100*t_aux/t_full:5.1f}%)")
print(f"  差(未归属)      {per(t_full - t_gemm - t_aux):7.3f} ms")

# 单层放大：96 层 × 命中数
NHIT = len(hits)
print(f"\n外推到一整步（假设每层都命中 {NHIT} 个专家，共 96 层）:")
print(f"  GEMM 部分     {t_gemm * 96:7.2f} s")
print(f"  辅助 op 部分  {t_aux * 96:7.2f} s")
print(f"  合计          {t_full * 96:7.2f} s   （实测 T=2048 前向 21.4 s，其中计算项拟合 15.1 s）")
