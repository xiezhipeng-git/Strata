"""fla GDN 是否依赖分配器状态：同进程内扰动内存后再调用同一个 op，比较输出。

动机：T=15 的 CE 跨进程偶尔差 0.0018 nats，逐模块定位到 GDN；但 fla 的 autotune config
已被证明不改变数值。剩下最可能的类别是「读未初始化显存」——它的特征正是：
  · 同进程内状态不变 ⇒ 反复调用逐位一致（我们观察到的）；
  · 进程间分配器历史不同 ⇒ 偶尔不同（我们观察到的）。
判据：同进程内先跑一次记下 digest，然后分配/释放一批不同大小的张量打乱空闲链表，
再跑一次。若 digest 变了 ⇒ 这个 op 读了没被写过的显存。

用法：python3 _mb_gdn_alloc.py [--t 15,64] [--h 48 --k 128 --v 128] [--rounds 8]
"""
from __future__ import annotations

import argparse
import hashlib
import time

import torch

dev = "cuda"


def dig(t: torch.Tensor) -> str:
    b = t.detach().float().cpu().contiguous()
    return hashlib.blake2b(b.numpy().tobytes(), digest_size=6).hexdigest()


def churn(nbytes: int, rounds: int = 6) -> None:
    """制造一段不同的分配历史：吃进一块空闲内存再放掉，并留下不同尺寸的碎片。"""
    keep = [torch.empty(nbytes + i * 4096, dtype=torch.uint8, device=dev)
            for i in range(rounds)]
    keep += [torch.empty(1 << 20, dtype=torch.uint8, device=dev) for _ in range(rounds)]
    del keep
    torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", default="15,64,2048")
    ap.add_argument("--h", type=int, default=48)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--v", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=8)
    args = ap.parse_args()

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(0)
    for T in [int(x) for x in args.t.split(",")]:
        q = torch.randn(1, T, args.h, args.k, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, T, args.h, args.k, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, T, args.h, args.v, device=dev, dtype=torch.bfloat16)
        g = -torch.rand(1, T, args.h, device=dev, dtype=torch.float32)
        b = torch.rand(1, T, args.h, device=dev, dtype=torch.float32)

        def run():
            o, _ = chunk_gated_delta_rule(q, k, v, g, b, head_first=False)
            torch.cuda.synchronize()
            return o

        run()                                    # 预热 + autotune
        base = dig(run())
        seen = {base}
        sizes = [T * args.h * args.v * 2, 3 << 20, 1 << 22, T * args.h * args.k * 2]
        for i in range(args.rounds):
            churn(sizes[i % len(sizes)])
            seen.add(dig(run()))
        print(f"T={T:5d}  输出 digest {len(seen)} 种  {sorted(seen)}"
              f"  {'OK 与分配器无关' if len(seen) == 1 else '*** 依赖分配器状态 ***'}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
