"""专家 pack 指纹：哈希 forward 真正看到的 gu/ga/dn/da（跨进程对比）。

_det_layers.py 已定位：layers.0 的路由 logits 逐位一致、命中 116 个专家两次相同，
但每个专家的输出不同 ⇒ 同一个专家、不同的权重值。而 _det_weights.py 只覆盖了
named_parameters/buffers，**expert pack 不在里面**（StreamingNF4Experts 无参数）。

本脚本按真实路径（`_stream_groups`，含 resident 视图与 staging H2D）取层 0 与层 40
的全部专家块做指纹，两个进程各存一份再 diff。

用法：
    python3 _det_pack.py --out /root/p_1.json
    python3 _det_pack.py --out /root/p_2.json
    python3 _det_pack.py --diff /root/p_1.json /root/p_2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json

import torch

import loader_qwen4 as L

GGUF = [
    "/mnt/d/Admin/Documents/GitHub/Strata-data/models/swift-IQ3_XXS/"
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    "/mnt/d/Admin/Documents/GitHub/Strata-data/models/swift-IQ3_XXS/"
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
]
PACK = "/root/experts_nf4.bin"


def dg(t: torch.Tensor) -> str:
    b = t.detach().to("cpu").contiguous()
    return hashlib.blake2b(b.view(torch.uint8).reshape(-1).numpy().tobytes(),
                           digest_size=6).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--diff", nargs=2, default=None)
    ap.add_argument("--layers", default="0,40")
    args = ap.parse_args()

    if args.diff:
        a, b = json.load(open(args.diff[0])), json.load(open(args.diff[1]))
        keys = sorted(a.keys() & b.keys())
        bad = [k for k in keys if a[k] != b[k]]
        print(f"共 {len(keys)} 个张量指纹，不同 {len(bad)} 个")
        for k in bad[:20]:
            print(f"  {k}  {a[k]} vs {b[k]}")
        return 0

    model, _ = L.load_model(GGUF, PACK)
    tp = L._TIERED
    rec: dict[str, str] = {}
    rec["_gpu_buf"] = dg(tp.gpu_buf[: 256 * tp.block_bytes])
    print(f"gpu_blocks={tp.gpu_blocks} block_bytes={tp.block_bytes} "
          f"pinned={tp.cpu_pinned}", flush=True)

    # 文件真值：显存里的 pack 必须逐字节等于文件（跨进程比对只能发现"不稳定"，
    # 这一条才能发现"稳定地错"）。只测前 512 块（layer 0 全部属这一批）。
    nb_chk = 512 * tp.block_bytes
    fd = L.os.open(PACK, L.os.O_RDONLY)
    raw = L.np.empty(nb_chk, dtype=L.np.uint8)
    got = L.os.preadv(fd, [raw], 0)
    L.os.close(fd)
    assert got == nb_chk, (got, nb_chk)
    fh = hashlib.blake2b(raw.tobytes(), digest_size=6).hexdigest()
    gh = hashlib.blake2b(
        tp.gpu_buf[:nb_chk].to("cpu").numpy().tobytes(), digest_size=6).hexdigest()
    print(f"前 512 块 文件 {fh} vs 显存 {gh}  "
          f"{'MATCH' if fh == gh else '*** MISMATCH ***'}", flush=True)
    rec["_file_512"] = fh
    rec["_gpu_512"] = gh

    for li in [int(x) for x in args.layers.split(",")]:
        ex = model.model.layers[li].mlp.experts
        seg = (ex._gu_n, ex._ga_n, ex._dn_n, ex._dam_n)
        hits = list(range(ex.num_experts))
        n_res = 0
        for group, tens in ex._stream_groups(hits, "cuda"):
            for e in group:
                gu, ga, dn, da = tens[e]
                off, _ = ex._offs[("gu", e)]
                res = tp.resident_gpu(off) >= 0
                n_res += res
                rec[f"L{li}.e{e}.gu"] = dg(gu)
                rec[f"L{li}.e{e}.ga"] = dg(ga)
                rec[f"L{li}.e{e}.dn"] = dg(dn)
                rec[f"L{li}.e{e}.da"] = dg(da)
                rec[f"L{li}.e{e}.src"] = "gpu" if res else "ram"
        print(f"layer {li}: {ex.num_experts} experts, {n_res} resident", flush=True)

    json.dump(rec, open(args.out, "w"))
    print(f"{len(rec)} 条指纹 -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
