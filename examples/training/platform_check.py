"""平台守卫：训练这条路径只支持 Linux；Windows 明确拒绝，并说清到底卡在哪。

为什么单独一个文件：以前在 Windows 上跑 `train.py` 的失败方式是在**读完 76 GB pack
之前**什么都没发生，然后在 `loader_qwen4.py` 深处抛
`AttributeError: module 'os' has no attribute 'preadv'`—— 一个跟真正原因（平台不支持）
毫无关系的报错。所以把「能不能跑」提到入口，用能核对的话说清楚。

**本模块不 import torch**，这样上游 CI 用 `requirements.txt` 那个只有标准库的环境
也能直接跑它，作为合并门的第一道（见 `.github/workflows/training-test.yml`）。

探测结果分两类，别混：

  阻断项（blockers）—— 真的跑不了
  降级项（degrades） —— 能跑，但少了一个优化，别有人以为是 bug 去"修"

用法：
  python3 platform_check.py            # 打印支持矩阵 + 本机探测，阻断时退出码 2
  python3 platform_check.py --json     # 同上，输出机器可读

在 `train.py` 里是入口处一行 `platform_check.require()`。
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import sys

# 支持的平台。只有 linux —— 不是"Windows 还没测"，是有具体的调用不存在（见下）。
SUPPORTED = ("linux",)

# 降级项不阻断。写成常量是为了让人看清"这个是故意的"：Windows 上没有 madvise，
# 于是 GGUF 的映射页不还，多占那 ~15 GiB（见 BENCH-SFT-VS-MEZO.md §12.2）。
# 只有 pack 装载（preadv）是硬阻断。
PREADV_USERS = (
    "loader_qwen4.py:294  （_TieredPack._ensure：pack → 显存，双 stage 版）",
    "loader_qwen4.py:339  （_TieredPack._ensure：pack → 锁页内存）",
    "bench_pack_io.py:26  （pack 随机读天花板基准）",
)


def _torch_facts() -> dict:
    """torch / CUDA 的事实。torch 在这里才 import，所以本文件不依赖它。"""
    try:
        import torch
    except Exception as exc:                       # 没装 torch 也算一种事实，不是异常
        return {"torch": None, "torch_error": type(exc).__name__, "cuda": False}
    return {"torch": torch.__version__, "torch_error": None,
            "cuda": bool(torch.cuda.is_available())}


def probe() -> dict:
    """把「这台机器能不能跑」拆成几条能单独核对的事实。

    返回值是纯数据，`blockers()` 只吃这个 dict —— 于是测试可以直接造一个假的
    capabilities，不用去 monkeypatch 全局环境。
    """
    caps = {
        "platform": sys.platform,
        "has_preadv": hasattr(os, "preadv"),
        "has_madvise": hasattr(mmap.mmap, "madvise"),
    }
    caps.update(_torch_facts())
    return caps


def blockers(caps: dict) -> list[str]:
    """返回阻断原因。空列表 = 可以跑。每条都写成能自己核对的判据。"""
    out: list[str] = []

    plat = str(caps.get("platform") or "")
    if not plat.startswith(SUPPORTED):
        out.append(
            f"平台 {plat!r} 不在支持列表 {SUPPORTED} 里。"
            f"训练路径的 pack 装载用 os.preadv 做「一次系统调用读 N 个不连续块」，"
            f"而 Python 文档里 preadv 的可用范围是 Linux ≥2.6.30 / FreeBSD ≥6.0 / "
            f"OpenBSD ≥2.7 / AIX ≥7.1 —— 没有 Windows。"
            f"在 win32 上实测 hasattr(os,'preadv') 为 False。")

    # 这一条其实被上面那条蕴含了，但仍然单独查：万一哪天有人在 Windows 上补了
    # preadv 的替代实现，这里会先亮，比"某个深层函数抛 AttributeError"好定位。
    if not caps.get("has_preadv"):
        out.append(
            "os.preadv 不存在。用它的是：" + "；".join(PREADV_USERS) +
            "。这几处都没有回退分支，调用即 AttributeError。")

    if caps.get("torch") is None:
        err = caps.get("torch_error") or ""
        out.append(f"torch 没装（import 抛 {err or 'Exception'}）。"
                   f"训练路径必需；只有 collect_rollouts.py 不需要。")
    elif not caps.get("cuda"):
        out.append(f"torch {caps['torch']} 装了但 torch.cuda.is_available() 为 False。"
                   f"训练要走 CUDA —— CPU 上每步是分钟级，不是慢一点。")

    return out


def degrades(caps: dict) -> list[str]:
    """能跑，但少了个优化。列出来是为了防止有人当 bug 去修。"""
    out: list[str] = []
    if not caps.get("has_madvise"):
        out.append(
            "mmap.mmap.madvise 不存在 ⇒ GGUF 的映射页还不掉（loader_qwen4.py:126、889 有 "
            "hasattr 守卫，Windows 上是无操作）。代价是多占约 15 GiB 常驻，不是正确性"
            "问题；小显存档可能在基座加载阶段就先爆，见 BENCH-SFT-VS-MEZO.md §12.2。")
    return out


def require(caps: dict | None = None) -> dict:
    """有阻断就打印原因并以退出码 2 结束；否则返回 capabilities。

    退出码用 2 而不是 1：1 在训练脚本里被当作"训练本身失败"，守卫被拒应该是
    "根本没跑"，两者在 CI 里要能分开看。
    """
    caps = probe() if caps is None else caps
    bad = blockers(caps)
    if not bad:
        return caps
    w = sys.stderr.write
    w("=" * 78 + "\n")
    w("训练路径在本机跑不了 —— 平台不支持，不是配置问题。\n")
    w("=" * 78 + "\n")
    for i, b in enumerate(bad, 1):
        w(f"\n{i}. {b}\n")
    soft = degrades(caps)
    if soft:
        w("\n（以下是能跑但会降级的项，供参考）\n")
        for s in soft:
            w(f"  - {s}\n")
    w("\n支持矩阵：Linux ✅ / Windows ❌（本 PR 未做，见下） / macOS ❌（未测）\n")
    w("Windows 想支持，要动的是上面第 2 条：把 preadv 换成 os.pread 的循环\n")
    w("（或 np.memmap + readinto），并且重测锁页上限（本机 Linux 是 32 GiB，\n")
    w("Windows 侧数值不同，loader_qwen4.py 是按减半重试写的，不假设具体值）。\n")
    w("=" * 78 + "\n")
    raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description="打印训练路径的平台支持矩阵与本机探测结果。")
    ap.add_argument("--json", action="store_true", help="输出 JSON（供脚本判断能不能跑）")
    a = ap.parse_args()

    caps = probe()
    bad, soft = blockers(caps), degrades(caps)
    if a.json:
        print(json.dumps({"caps": caps, "blockers": bad, "degrades": soft},
                         indent=2, ensure_ascii=False))
        return 2 if bad else 0

    print("=" * 78)
    print("训练路径的平台支持矩阵（跟本机无关，是这份代码的声明）")
    print("=" * 78)
    print("  Linux    ✅ 支持 —— 有 os.preadv，锁页上限本机实测 32 GiB")
    print("  Windows  ❌ 不支持 —— 本 PR 未做，见 README「Platform support」")
    print("  macOS    ❌ 未测 —— 没有机器；有 preadv，但没验证过")
    print("-" * 78)
    print(f"本机：{caps.get('platform')}  →  {'❌ 被阻断' if bad else '✅ 可以跑'}")
    print("-" * 78)
    print("探测明细：")
    for k in ("has_preadv", "has_madvise", "torch", "cuda"):
        print(f"  {k:<11s} {caps.get(k)}")
    print("-" * 78)
    if bad:
        print("阻断项：")
        for i, b in enumerate(bad, 1):
            print(f"  {i}. {b}")
    else:
        print("阻断项：无 ✅")
    if soft:
        print("降级项：")
        for s in soft:
            print(f"  - {s}")
    print("=" * 78)
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
