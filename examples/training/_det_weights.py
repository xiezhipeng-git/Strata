"""跨进程权重指纹：找出 load_model 之后「每个进程不一样」的张量。

_det_short.py 的判据：同进程内 3 次前向逐位一致（digest SAME），跨 2 个进程三个用例
全部不同。⇒ 不是非确定 kernel，是**加载期就不同**。
`load_model` 用 `with torch.device("meta")` 建模型再 `to_empty(device="cuda")`，任何没被
GGUF 填上的参数/缓冲都留着未初始化显存 —— 跨进程必然不同，同进程内稳定不变，正好符合
观测。本脚本给每个参数算指纹，两个进程输出对比即可定位。

用法：
    python3 _det_weights.py --out /root/w_1.json
    python3 _det_weights.py --out /root/w_2.json
    python3 _det_weights.py --diff /root/w_1.json /root/w_2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json

import torch

GGUF = [
    "/mnt/d/Admin/Documents/GitHub/Strata-data/models/swift-IQ3_XXS/"
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    "/mnt/d/Admin/Documents/GitHub/Strata-data/models/swift-IQ3_XXS/"
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
]
PACK = "/root/experts_nf4.bin"


def digest(t: torch.Tensor) -> str:
    b = t.detach().to("cpu", non_blocking=False).contiguous()
    raw = b.view(torch.uint8).reshape(-1).numpy().tobytes()
    return hashlib.blake2b(raw, digest_size=8).hexdigest()


def dump(out: str) -> int:
    from loader_qwen4 import load_model

    model, _ = load_model(GGUF, PACK)
    recs = {}
    items = list(model.named_parameters()) + list(model.named_buffers())
    for name, t in items:
        f = t.detach().float()
        recs[name] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "digest": digest(t),
            "absmax": float(f.abs().max()) if f.numel() else 0.0,
            "nonfinite": int((~torch.isfinite(f)).sum()),
        }
    json.dump(recs, open(out, "w"), indent=0)
    print(f"{len(recs)} tensors -> {out}", flush=True)
    bad = {k: v for k, v in recs.items() if v["nonfinite"]}
    if bad:
        print(f"非有限值张量 {len(bad)} 个（未初始化/算坏的强信号）：", flush=True)
        for k, v in list(bad.items())[:20]:
            print(f"  {k}  shape={v['shape']} nonfinite={v['nonfinite']} absmax={v['absmax']:.3e}")
    return 0


def diff(p1: str, p2: str) -> int:
    a, b = json.load(open(p1)), json.load(open(p2))
    only = set(a) ^ set(b)
    if only:
        print(f"张量集合不同: {sorted(only)[:10]}")
    diffs = [(k, a[k], b[k]) for k in a.keys() & b.keys() if a[k]["digest"] != b[k]["digest"]]
    print(f"共 {len(a.keys() & b.keys())} 个张量，指纹不同 {len(diffs)} 个")
    for k, x, y in diffs:
        print(f"  {k}  shape={x['shape']} {x['dtype']}  "
              f"absmax {x['absmax']:.3e} vs {y['absmax']:.3e}  "
              f"nonfinite {x['nonfinite']} vs {y['nonfinite']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--diff", nargs=2, default=None)
    args = ap.parse_args()
    if args.diff:
        return diff(*args.diff)
    return dump(args.out or "/root/w.json")


if __name__ == "__main__":
    raise SystemExit(main())
