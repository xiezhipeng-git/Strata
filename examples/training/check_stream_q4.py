"""正确性对照：新流式读盘路径 vs 旧 mmap 切片路径，输出必须逐位相等。

随机抽 (layer, expert)，分别用 _stream_groups（新）和旧 _weight 逻辑解压，
torch.equal 断言。这是替换读盘路径的唯一判据：字节相同 + 解压确定性 => 等价。
"""
import json, random, sys
import numpy as np
import torch

sys.path.insert(0, "/mnt/d/Admin/Documents/GitHub/Strata-train/examples/training")
import loader_qwen4 as L
import nf4

PACK = "/root/experts_nf4.bin"
meta = json.load(open(PACK + ".json"))
offs = L._expert_offsets_from_meta(meta)
pack_mm = np.memmap(PACK, dtype=np.uint8, mode="r")

def old_weight(l, kind, e):
    """旧路径逐行复刻（mmap 切片 -> CPU 拷贝 -> H2D -> 解压）。"""
    off, nbytes = offs[l][(kind, e)]
    shape = (1280, 2560) if kind == "gu" else (2560, 640)
    packed = torch.from_numpy(np.ascontiguousarray(pack_mm[off: off + nbytes])).to("cuda")
    off2, n2 = offs[l][(kind + "_absmax", e)]
    absmax = torch.from_numpy(np.ascontiguousarray(pack_mm[off2: off2 + n2])).to("cuda").view(torch.bfloat16)
    return nf4.dequantize(packed, absmax, shape)

random.seed(42)
fail = 0
for trial in range(8):                      # 8 层 x 5 专家 = 40 对照
    l = random.randrange(48)
    hits = random.sample(range(512), 5)
    mod = L.StreamingNF4Experts(512, 2560, 640, torch.nn.functional.silu, PACK, l, offs[l])
    got = {}
    for group, gu_w, dn_w in mod._stream_groups(hits, "cuda"):
        for i, e in enumerate(group):
            got[e] = (gu_w[i], dn_w[i])
    assert set(got) == set(hits), f"layer {l}: hits mismatch {set(got)} vs {set(hits)}"
    for e in hits:
        ref_gu, ref_dn = old_weight(l, "gu", e), old_weight(l, "dn", e)
        ok = torch.equal(got[e][0], ref_gu) and torch.equal(got[e][1], ref_dn)
        if not ok:
            fail += 1
            d = (got[e][0] - ref_gu).abs().max().item()
            print(f"MISMATCH layer {l} expert {e}: max|dgu|={d}")
    # 组间流水正确性：单专家组与混在组里结果一致（上面 5 个同组已覆盖；再来跨组）
if fail == 0:
    print("OK: 40/40 experts bitwise identical (new preadv+batched path == old mmap path)")
else:
    print(f"FAILED: {fail} mismatches")
    sys.exit(1)
