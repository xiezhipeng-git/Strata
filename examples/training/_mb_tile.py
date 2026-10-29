"""微基准：fused NF4 fwd kernel 的 tile 参数能买到多少。

单专家形状取自 pack：gu (1280, 2560)、dn (2560, 640)。
T=64 = T2048 × top_k 8 / 256 experts。
对照上界：dequant 一次 + cuBLAS bf16（旧两段路径，会写 9.8MB 中间张量）。
"""
import sys, time
import torch
sys.path.insert(0, "/mnt/d/Admin/Documents/GitHub/Strata-train/examples/training")

import nf4
from nf4_fused import _run, _get_code

dev = "cuda"
CODE = _get_code(torch.device("cuda"))
torch.manual_seed(0)


def bench(fn, reps=20, warm=5):
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


SHAPES = [(1280, 2560, "gu"), (2560, 640, "dn")]
T = 64

print(f"{'shape':6s} {'tile(BT,BO,BK)':18s} {'ms':>9s} {'TFLOPS':>8s} {'权重带宽GB/s':>13s}")
print("-" * 62)
best = {}
for OUT, IN, tag in SHAPES:
    w = torch.randn(OUT, IN, device=dev, dtype=torch.bfloat16) * 0.02
    packed, absmax = nf4.quantize(w)
    x = torch.randn(T, IN, device=dev, dtype=torch.bfloat16)
    flops = 2 * T * OUT * IN
    wbytes = packed.numel() + absmax.numel() * 2

    rows = []
    # 现有默认：BT=16, BO=16, BK=自动
    for (BT, BO, BK) in [(16, 16, 0), (16, 32, 0), (16, 64, 0),
                         (32, 32, 0), (32, 64, 0), (64, 32, 0), (64, 64, 0),
                         (64, 32, 64), (64, 64, 64), (64, 128, 64)]:
        if BK and IN % BK:
            continue
        if OUT % BO:
            continue
        try:
            dt = bench(lambda: _run(x, packed, absmax, CODE, False, BT=BT, BO=BO, BK=BK))
        except Exception as e:
            print(f"{tag:6s} ({BT},{BO},{BK}){'':6s} FAIL {type(e).__name__}")
            continue
        tf = flops / dt / 1e12
        bw = wbytes / dt / 1e9
        rows.append((dt, BT, BO, BK or "auto"))
        print(f"{tag:6s} ({BT},{BO},{BK or 'auto'}){'':5s} {dt*1000:9.3f} {tf:8.2f} {bw:13.1f}")

    # 上界：dequant 一次 + cuBLAS
    t_dec = bench(lambda: nf4.dequantize(packed, absmax, (OUT, IN)))
    wd = nf4.dequantize(packed, absmax, (OUT, IN))
    t_mm = bench(lambda: torch.nn.functional.linear(x, wd))
    t_both = t_dec + t_mm
    print(f"{tag:6s} {'dequant+cuBLAS':18s} {t_both*1000:9.3f} "
          f"{flops/t_both/1e12:8.2f} {'':13s}  (解码 {t_dec*1000:.3f} + GEMM {t_mm*1000:.3f})")
    rows.sort()
    best[tag] = rows[0]
    print()

print("各形状最快 tile：")
for tag, (dt, BT, BO, BK) in best.items():
    print(f"  {tag}: ({BT},{BO},{BK})  {dt*1000:.3f} ms")
