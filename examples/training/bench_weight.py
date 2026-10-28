"""微基准：完整复刻 StreamingNF4Experts._weight() 的单专家加载+解压路径。

不加模型、不建 125B 骨架，直接对 pack 打，测单次 (mmap 读块 -> H2D -> NF4 解压)
延迟与带宽，回答"训练 380 loads/s vs 引擎 decode ~60k loads/s 差在哪"。
"""
import os, sys, time, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader_qwen4 as L

PACK = "/root/experts_nf4.bin"
E, I, H, NL = 512, 640, 2560, 48
gu_shape, dn_shape = (2 * I, H), (H, I)

def packed_bytes(shape):
    return int(np.prod(shape)) // 2

def absmax_bytes(shape):
    return int(np.prod(shape)) // 64 * 2

per_expert = {
    "gu": packed_bytes(gu_shape), "gu_absmax": absmax_bytes(gu_shape),
    "dn": packed_bytes(dn_shape), "dn_absmax": absmax_bytes(dn_shape),
}
layer_bytes = E * sum(per_expert.values())
offs = {}
for l in range(NL):
    base = l * layer_bytes
    for e in range(E):
        o = base + e * sum(per_expert.values())
        offs[(l, e, "gu")] = (o, per_expert["gu"])
        offs[(l, e, "gu_absmax")] = (o + per_expert["gu"], per_expert["gu_absmax"])
        o2 = o + per_expert["gu"] + per_expert["gu_absmax"]
        offs[(l, e, "dn")] = (o2, per_expert["dn"])
        offs[(l, e, "dn_absmax")] = (o2 + per_expert["dn"], per_expert["dn_absmax"])

pack = np.memmap(PACK, dtype=np.uint8, mode="r")

def weight(kind, l, e):
    """逐行复刻 StreamingNF4Experts._weight"""
    off, nbytes = offs[(l, e, kind)]
    packed = torch.from_numpy(np.ascontiguousarray(pack[off: off + nbytes])).to("cuda")
    off2, nbytes2 = offs[(l, e, kind + "_absmax")]
    absmax = torch.from_numpy(np.ascontiguousarray(pack[off2: off2 + nbytes2])).to("cuda").view(torch.bfloat16)
    shape = gu_shape if kind == "gu" else dn_shape
    return L.nf4.dequantize(packed, absmax, shape)

random.seed(0)
N = 400
keys = [(random.randrange(NL), random.randrange(E)) for _ in range(N)]

# pass 1：冷/半冷（看 page cache 未热时的代价）
t0 = time.perf_counter()
for l, e in keys[:N // 2]:
    weight("gu", l, e)
torch.cuda.synchronize()
t_cold = time.perf_counter() - t0

# pass 2：热（page cache 命中）
torch.cuda.synchronize()
t0 = time.perf_counter()
nb = 0
for l, e in keys[N // 2:]:
    w = weight("gu", l, e)
    nb += w.numel() * 2
torch.cuda.synchronize()
t_warm = time.perf_counter() - t0
n = N // 2

print(f"cold half: {t_cold/(N//2)*1e3:.2f} ms/load  {N//2/t_cold:.0f} loads/s")
print(f"warm half: {t_warm/n*1e3:.2f} ms/load  {n/t_warm:.0f} loads/s  dequant {nb/2**20/t_warm:.0f} MiB/s")
print(f"训练实测: ~380 loads/s；引擎 decode 隐含: ~60k loads/s (160 tok/s x 384)")

# 对照：单纯 H2D（跳过 mmap 读），定位延迟在 CPU 侧拷贝还是 NF4 kernel
l, e = keys[0]
off, nbytes = offs[(l, e, "gu")]
buf = np.ascontiguousarray(pack[off: off + nbytes])
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(200):
    t = torch.from_numpy(buf).to("cuda")
torch.cuda.synchronize()
print(f"H2D only: {(time.perf_counter()-t0)/200*1e3:.3f} ms  ({nbytes/2**20/((time.perf_counter()-t0)/200)/1024:.1f} GiB/s)")

t0 = time.perf_counter()
for _ in range(200):
    p = torch.from_numpy(buf).cuda(non_blocking=False)
    _ = L.nf4.dequantize(p, torch.zeros(int(np.prod(gu_shape)) // 64, dtype=torch.bfloat16, device="cuda"), gu_shape)
torch.cuda.synchronize()
print(f"H2D+dequant: {(time.perf_counter()-t0)/200*1e3:.3f} ms/load  {200/(time.perf_counter()-t0):.0f} loads/s")
