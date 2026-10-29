"""pack 一块（2.49 MiB）的 H2D 到底多快：pageable / pinned / 与计算是否重叠。

背景：43.8 GiB CPU 层因 32 GiB pin 上限退回 pageable，每步要搬 43.8 GiB。
本脚本按真实块大小测三种源，并用「同时跑一个占满 SM 的 GEMM」验证重叠可行性。
"""
from __future__ import annotations

import time

import torch

DEV = torch.device("cuda")
BLOCK = 2611200          # 2.49 MiB = pack 一块（gu|ga|dn|da 四段连续）
NBLK = 96                # 每轮 96 块 ≈ 239 MiB，够摊开调度开销又不撑爆内存


def _timed(fn, rounds=3):
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    fn()
    torch.cuda.synchronize()
    ev0.record()
    for _ in range(rounds):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / rounds / 1e3          # s/轮


def report(tag, s, nbytes):
    gb = nbytes / s / 2**30
    print(f"{tag:34s} {s*1e3:7.1f} ms/轮  {gb:6.2f} GiB/s  "
          f"{s / NBLK * 1e6:6.0f} µs/块", flush=True)
    return gb


def main() -> int:
    dst = torch.empty(BLOCK, dtype=torch.uint8, device=DEV)
    page = [torch.empty(BLOCK, dtype=torch.uint8) for _ in range(NBLK)]
    pin = [torch.empty(BLOCK, dtype=torch.uint8, pin_memory=True) for _ in range(NBLK)]
    for t in page + pin:
        t.fill_(1)
    nb = BLOCK * NBLK

    print(f"=== 单流顺序：{NBLK} × {BLOCK/2**20:.2f} MiB ===", flush=True)

    def pg_blocking():
        for s in page:
            dst.copy_(s)                      # pageable 源 ⇒ 必然同步

    def pg_nonblock():
        for s in page:
            dst.copy_(s, non_blocking=True)

    def pin_async():
        for s in pin:
            dst.copy_(s, non_blocking=True)

    report("pageable  blocking", _timed(pg_blocking), nb)
    report("pageable  non_blocking", _timed(pg_nonblock), nb)
    report("pinned    non_blocking", _timed(pin_async), nb)

    # 大块对照（把 96 块并成一块 239 MiB）
    big = torch.empty(nb, dtype=torch.uint8, pin_memory=True)
    bigp = torch.empty(nb, dtype=torch.uint8)
    big.fill_(1)
    bigp.fill_(1)
    bigd = torch.empty(nb, dtype=torch.uint8, device=DEV)
    report("pinned    单块 239 MiB", _timed(lambda: bigd.copy_(big, non_blocking=True)), nb)
    report("pageable  单块 239 MiB", _timed(lambda: bigd.copy_(bigp)), nb)

    # 重叠：side stream 搬砖 + 主 stream 跑 GEMM
    print("\n=== 传输与计算重叠（GEMM 4096³ bf16，~21 ms/次）===", flush=True)
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device=DEV)
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device=DEV)
    gemm_ms = _timed(lambda: [torch.mm(a, b) for _ in range(5)], rounds=2) / 5 * 1e3
    print(f"  纯 GEMM                        {gemm_ms:7.1f} ms/次", flush=True)

    side = torch.cuda.Stream()
    n_gemm = 10

    def compute_only():
        for _ in range(n_gemm):
            torch.mm(a, b)

    t_compute = _timed(compute_only, rounds=2)

    def overlapped():
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for s in pin:
                dst.copy_(s, non_blocking=True)
        for _ in range(n_gemm):
            torch.mm(a, b)
        torch.cuda.current_stream().wait_stream(side)

    t_over = _timed(overlapped, rounds=2)
    print(f"  计算单独 {t_compute*1e3:7.1f} ms  |  计算+239 MiB 传输 {t_over*1e3:7.1f} ms  "
          f"⇒ 传输被遮住 {max(0.0, (t_compute + _timed(pin_async) - t_over)) * 1e3:.1f} ms",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
