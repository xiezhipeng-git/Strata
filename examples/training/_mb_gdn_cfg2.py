"""chunk_o / chunk_delta_h：每个 config 的数值指纹与耗时（不加载模型）。

回答两个问题：
  1) 同一份输入下，不同 config 的输出是否逐位相同？（决定 autotune 是否会污染可复现性）
  2) 固定成某个 config 的代价是多少 ms？

用法：python3 _mb_gdn_cfg2.py [--t 15,2048]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import time

import torch

torch.manual_seed(0)
dev = "cuda"


def find_tuners(modnames):
    out = []
    for modname in modnames:
        m = importlib.import_module(modname)
        for attr in dir(m):
            obj = getattr(m, attr)
            for _ in range(4):
                if isinstance(getattr(obj, "configs", None), list):
                    out.append((f"{modname.split('.')[-1]}.{attr}", obj))
                    break
                nxt = getattr(obj, "fn", None)
                if nxt is None or nxt is obj:
                    break
                obj = nxt
    return out


def bench(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", default="15,2048")
    ap.add_argument("--h", type=int, default=48)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--v", type=int, default=128)
    args = ap.parse_args()

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    tuners = find_tuners((
        "fla.ops.common.chunk_o", "fla.ops.common.chunk_delta_h",
        "fla.ops.common.chunk_scaled_dot_kkt", "fla.ops.common.chunk_h",
        "fla.ops.gated_delta_rule.wy_fast", "fla.ops.utils.cumsum",
        "fla.ops.utils.solve_tril",
    ))
    for T in [int(x) for x in args.t.split(",")]:
        q = torch.randn(1, T, args.h, args.k, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, T, args.h, args.k, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, T, args.h, args.v, device=dev, dtype=torch.bfloat16)
        g = -torch.rand(1, T, args.h, device=dev, dtype=torch.float32)
        b = torch.rand(1, T, args.h, device=dev, dtype=torch.float32)

        def run():
            return chunk_gated_delta_rule(q, k, v, g, b, head_first=False)

        print(f"\n=== T={T}  autotune {bench(run) * 1000:.3f} ms ===", flush=True)
        for label, tu in tuners:
            orig = list(tu.configs)
            rows = []
            for c in orig:
                tu.configs = [c]
                tu.cache.clear()
                try:
                    o, _ = run()
                    torch.cuda.synchronize()
                    dg = hashlib.blake2b(o.float().cpu().numpy().tobytes(),
                                        digest_size=6).hexdigest()
                    rows.append((c.kwargs, c.num_warps, bench(run, n=10, warm=3), dg))
                except Exception as e:
                    rows.append((c.kwargs, c.num_warps, None, type(e).__name__))
            tu.configs = orig
            tu.cache.clear()
            ok = [r for r in rows if r[2]]
            nd = len({r[3] for r in ok})
            verdict = "数值全部相同" if nd <= 1 else f"数值 {nd} 种不同 ***"
            print(f" {label}: {len(rows)} config, {verdict}")
            for kw, nw, tt, dg in rows:
                s = f"{tt * 1000:8.3f} ms  {dg}" if tt else f"不可行 {dg}"
                print(f"   {str(kw):22s} w={nw}  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
