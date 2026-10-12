r"""从 setup.py 写出的那份模型配置，派生一份**指向 strata-lp.exe** 的配置。

为什么要有它：`setup.py` 只认预编译引擎（它把 `engine\strata.exe` 写进配置的 `exe` 字段）。
要在同一个模型 pack 上用**带 logprobs 的引擎**，就得有一份 `exe` 指向 `strata-lp.exe` 的配置。
而这个配置**不该手写** —— pack 路径、MTP、KV、vision 等十几个字段 setup.py 会随机器和
设置变化，抄一遍就会漂。

做法：读 setup.py 刚写的那份配置（只改 `exe` 和 `log` 两个字段），原样写出一份
`strata-lp-<原名>.json`。

用法：
    python local\logprobs\make_lp_config.py --exe build-lp\local\logprobs\build\strata-lp.exe
    python local\logprobs\make_lp_config.py --exe ... --from strata-swift-iq3_xxs.json

成功时把派生配置的**路径**打印到 stdout（只有路径，方便 bat 接住）；
失败时打印原因到 stderr 并以 2 退出。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def candidates() -> list[Path]:
    """仓库里的模型配置：`strata-*.json`，且**不是**我们自己派生的、也不是备份。

    判据与 setup.py 的 `model_config()` 一致：有 `exe` 和 `args`。
    """
    out: list[Path] = []
    for p in sorted(ROOT.glob("strata-*.json")):
        name = p.name
        if ".bak" in name:                      # strata-x.json.bak
            continue
        if name.startswith("strata-lp-"):       # 上轮自己派生的
            continue
        try:
            cfg = json.loads(p.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(cfg, dict) and cfg.get("exe") and isinstance(cfg.get("args"), list):
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--exe", required=True, help="strata-lp engine path")
    ap.add_argument("--from", dest="src", default="", help="use this config instead of picking one")
    known, _ = ap.parse_known_args()

    exe = Path(known.exe)
    if not exe.is_absolute():
        exe = (ROOT / exe).resolve()
    if not exe.is_file():
        print(f"[make-lp-config] engine not found: {exe}", file=sys.stderr)
        return 2

    src = Path(known.src) if known.src else None
    if src is not None:
        if not src.is_absolute():
            src = (ROOT / src).resolve()
        if not src.is_file():
            print(f"[make-lp-config] config not found: {src}", file=sys.stderr)
            return 2
        picked = [src]
    else:
        picked = candidates()

    if not picked:
        print("[make-lp-config] no model config found "
              "(a strata-*.json with exe+args). Run setup.py once first.", file=sys.stderr)
        return 2

    # 取最新的那份：setup.py 刚跑过，它写的 mtime 最新。
    src = max(picked, key=lambda p: p.stat().st_mtime)
    cfg = json.loads(src.read_text(encoding="utf-8-sig"))

    stem = src.name[:-5] if src.name.endswith(".json") else src.name
    # strata-swift-iq3_xxs -> strata-lp-swift-iq3_xxs   (don't double the prefix)
    if stem.startswith("strata-"):
        stem = "strata-lp-" + stem[len("strata-"):]
    else:
        stem = "strata-lp-" + stem
    out = src.with_name(stem + ".json")

    cfg["exe"] = str(exe)
    # 日志换成另一个文件，别覆盖原版引擎的（两份服务可能都在排障）。
    old_log = str(cfg.get("log") or "")
    if old_log:
        cfg["log"] = str(Path(old_log).with_name(f"strata-lp-{Path(old_log).stem}.log"))

    out.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
