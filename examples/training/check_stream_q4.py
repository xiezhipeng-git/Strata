"""正确性对照：fused 读盘+计算路径 vs 参考路径。

1) 上传到 GPU 的 packed/absmax 字节必须与 mmap 直读逐位相同（读盘路径等价）
2) fused_nf4_linear 输出 vs dequantize+F.linear（容差 = bf16 舍入级，<0.05）
3) 反传 dx vs dy@W 参考（<0.1）
"""
import json, random, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/d/Admin/Documents/GitHub/Strata-train/examples/training")
import loader_qwen4 as L
import nf4
from nf4_fused import fused_nf4_linear

PACK = "/root/experts_nf4.bin"
meta = json.load(open(PACK + ".json"))
offs = L._expert_offsets_from_meta(meta)
pack_mm = np.memmap(PACK, dtype=np.uint8, mode="r")
torch.manual_seed(7)

def ref_weight(l, kind, e):
    off, nbytes = offs[l][(kind, e)]
    shape = (1280, 2560) if kind == "gu" else (2560, 640)
    packed = torch.from_numpy(np.ascontiguousarray(pack_mm[off: off + nbytes])).to("cuda")
    off2, n2 = offs[l][(kind + "_absmax", e)]
    absmax = torch.from_numpy(np.ascontiguousarray(pack_mm[off2: off2 + n2])).to("cuda").view(torch.bfloat16)
    return packed, absmax, nf4.dequantize(packed, absmax, shape)

bad = 0
for trial in range(6):                     # 6 层 × 5 专家
    l = random.randrange(48)
    hits = random.sample(range(512), 5)
    mod = L.StreamingNF4Experts(512, 2560, 640, torch.nn.functional.silu, PACK, l, offs[l])
    seen = {}
    for group, gu_p, gu_a, dn_p, dn_a in mod._stream_groups(hits, "cuda"):
        for i, e in enumerate(group):
            seen[e] = (gu_p[i], gu_a[i], dn_p[i], dn_a[i])
    assert set(seen) == set(hits)
    for e in hits:
        gu_p, gu_a, dn_p, dn_a = seen[e]
        rp, ra, rw = ref_weight(l, "gu", e)
        dp, da, dw = ref_weight(l, "dn", e)
        # 1) 字节级
        if not (torch.equal(gu_p, rp.view_as(gu_p)) and torch.equal(gu_a, ra.view_as(gu_a))
                and torch.equal(dn_p, dp.view_as(dn_p)) and torch.equal(dn_a, da.view_as(dn_a))):
            print(f"BYTES MISMATCH layer {l} expert {e}"); bad += 1; continue
        # 2) fwd
        x = torch.randn(8, 2560, device="cuda", dtype=torch.bfloat16)
        y = fused_nf4_linear(x, gu_p, gu_a)
        y_ref = F.linear(x, rw)
        if (y.float() - y_ref.float()).abs().max().item() > 0.05:
            print(f"FWD MISMATCH layer {l} expert {e}"); bad += 1
        # 3) bwd（autograd 全链路）
        xg = x.clone().requires_grad_(True)
        fused_nf4_linear(xg, gu_p, gu_a).sum().backward()
        dy = torch.ones_like(y)
        dx_ref = (dy.float() @ rw.float()).to(torch.bfloat16)
        if (xg.grad.float() - dx_ref.float()).abs().max().item() > 0.1:
            print(f"BWD MISMATCH layer {l} expert {e} "
                  f"max|Δ|={(xg.grad.float()-dx_ref.float()).abs().max().item():.3e}"); bad += 1

print("OK: bytes bitwise + fwd/bwd within bf16 tolerance" if bad == 0 else f"FAILED: {bad}")
sys.exit(1 if bad else 0)
