"""对比两份 `_det_layers.py` 输出的指纹 JSON：报总差异、首个分叉、以及指定层的实况。

比 `_det_layers.py --diff` 多两点：按名字统计「调用次数不同」的模块（专家命中数变化会
表现在 `experts.act_fn` 的调用次数上），以及可以把某个模块（含 `#in` 输入指纹）逐层打印。

用法：
    python3 _det_cmp_layers.py a.json b.json [--show linear_attn.norm --layers 28-32]
"""
from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--show", default="linear_attn.norm",
                    help="要逐层打印的模块后缀（'#in' 为输入指纹）")
    ap.add_argument("--layers", default="")
    args = ap.parse_args()

    a, b = json.load(open(args.a)), json.load(open(args.b))
    da, db = a["dig"], b["dig"]

    cnt = [(k, len(da.get(k, [])), len(db.get(k, [])))
           for k in da.keys() | db.keys() if len(da.get(k, [])) != len(db.get(k, []))]
    print(f"调用次数不同的模块 {len(cnt)} 个")
    for k, na, nb in sorted(cnt)[:20]:
        print(f"  {k}  {na} vs {nb}")

    bad = [k for k in da if da[k] != db.get(k)]
    print(f"\n指纹不同的模块 {len(bad)} / {len(set(da) | set(db))}")
    for k in sorted(bad)[:10]:
        print(f"  {k}")

    sa, sb = a["seq"], b["seq"]
    for i in range(min(len(sa), len(sb))):
        if sa[i] != sb[i]:
            print(f"\n调用序首个不同 #{i}: {sa[i]} vs {sb[i]}")
            break
    else:
        print(f"\n调用序前 {min(len(sa), len(sb))} 项全同")

    if args.layers:
        lo, _, hi = args.layers.partition("-")
        rng = range(int(lo), int(hi or lo) + 1)
        print()
        for L in rng:
            for suf in (args.show + "#in", args.show):
                k = f"model.layers.{L}.{suf}"
                x, y = da.get(k), db.get(k)
                print(f"  {k:52s} {'SAME' if x == y else 'DIFF'}  {x}"
                      + ("" if x == y else f" {y}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
