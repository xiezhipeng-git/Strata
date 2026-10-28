"""基准 2：_weight() 慢的根因验证——mmap 随机切片 page fault vs 替代读法。

四个变体都做完整链路（读 1.64MB gu packed -> H2D -> NF4 dequant），只换读法：
  A: 现状   np.ascontiguousarray(np.memmap 切片)
  B: pread  os.pread(fd) 到复用 buffer -> torch.frombuffer
  C: mmap+madvise(WILLNEED) 预热后切片
  D: torch.UntypedStorage.from_file 整包映射, narrow 视图直接 .to(cuda)
"""
import os, sys, time, random, mmap
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader_qwen4 as L

PACK = "/root/experts_nf4.bin"
E, I, H, NL = 512, 640, 2560, 48
gu_shape, dn_shape = (2 * I, H), (H, I)
gu_packed = int(np.prod(gu_shape)) // 2
gu_absmax = int(np.prod(gu_shape)) // 64 * 2
per_expert = gu_packed + gu_absmax + int(np.prod(dn_shape)) // 2 + int(np.prod(dn_shape)) // 64 * 2
layer_bytes = E * per_expert

def gu_off(l, e):
    return l * layer_bytes + e * per_expert

pack = np.memmap(PACK, dtype=np.uint8, mode="r")
fd = os.open(PACK, os.O_RDONLY)
mm = mmap.mmap(fd, 0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
storage = torch.UntypedStorage.from_file(PACK, True, os.path.getsize(PACK))
staging1 = torch.empty(gu_packed, dtype=torch.uint8, pin_memory=True)
staging2 = torch.empty(gu_absmax, dtype=torch.uint8, pin_memory=True)

deq_absmax = torch.zeros(int(np.prod(gu_shape)) // 64, dtype=torch.bfloat16, device="cuda")

def t_A(l, e):
    off = gu_off(l, e)
    packed = torch.from_numpy(np.ascontiguousarray(pack[off: off + gu_packed])).to("cuda")
    off2 = off + gu_packed
    am = torch.from_numpy(np.ascontiguousarray(pack[off2: off2 + gu_absmax])).to("cuda").view(torch.bfloat16)
    return L.nf4.dequantize(packed, am, gu_shape)

def t_B(l, e):
    off = gu_off(l, e)
    os.preadv(fd, [staging1.numpy()], off)
    packed = staging1.to("cuda", non_blocking=True)
    os.preadv(fd, [staging2.numpy()], off + gu_packed)
    am = staging2.to("cuda", non_blocking=True).view(torch.bfloat16)
    torch.cuda.synchronize()
    return L.nf4.dequantize(packed, am, gu_shape)

def t_B2(l, e):
    """B 的单 buffer 修正版：两次 pread 用不同 staging。"""
    off = gu_off(l, e)
    buf1 = os.pread(fd, gu_packed, off)
    buf2 = os.pread(fd, gu_absmax, off + gu_packed)
    packed = torch.frombuffer(bytearray(buf1), dtype=torch.uint8).to("cuda")
    am = torch.frombuffer(bytearray(buf2), dtype=torch.uint8).to("cuda").view(torch.bfloat16)
    return L.nf4.dequantize(packed, am, gu_shape)

def t_D(l, e):
    off = gu_off(l, e)
    packed = storage[off: off + gu_packed].to("cuda")
    am = storage[off + gu_packed: off + gu_packed + gu_absmax].to("cuda").view(torch.bfloat16)
    return L.nf4.dequantize(packed, am, gu_shape)

random.seed(0)
N = 300
keys = [(random.randrange(NL), random.randrange(E)) for _ in range(N)]

for name, fn in [("A memmap切片(现状)", t_A), ("B pread->pinned", t_B),
                 ("B2 pread->bytes", t_B2), ("D torch_storage", t_D)]:
    try:
        fn(*keys[0]); torch.cuda.synchronize()  # warm
        t0 = time.perf_counter()
        nb = 0
        for l, e in keys:
            w = fn(l, e); nb += w.numel() * 2
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"{name:22s} {dt/N*1e3:6.2f} ms/load  {N/dt:6.0f} loads/s  dequant {nb/2**20/dt:6.0f} MiB/s")
    except Exception as ex:
        print(f"{name:22s} FAILED: {type(ex).__name__}: {ex}")
