r"""setup.py（不启动）+ 启动服务，**默认用带 logprobs 的引擎**。

## 为什么有它

`setup.py` 只认预编译引擎 `engine\strata.exe`（它把这个路径写进模型配置的 `exe`）。
要在同一个 pack 上用 `strata-lp.exe`（带 per-request logprobs），必须：
  1. 让 setup.py 把该装的装好、配置写好，但**不要**由它启动（它只会启原版引擎）
  2. 派生一份 `exe` 指向 `strata-lp.exe` 的配置
  3. 用 `local/logprobs/server_logprobs.py` 启动

`strata-lp.exe` 没编出来时**自动退回原版引擎**，所以桌面启动永远起得来。

## 用法

    python local\logprobs\start_server.py --family swift --model IQ3_XXS ... --yes
    python local\logprobs\start_server.py --stock ...           # 强制原版
    python local\logprobs\start_server.py --logprobs ...         # 强制 logprobs，没编出来就报错
    python local\logprobs\start_server.py --setup ...            # 完全交给 setup.py（它自己收尾）

`--stock` / `--logprobs` / `--port` 是我们自己的开关，不会传给 setup.py（它不认识）。
`--setup` / `--check` / `--no-start` / `--calibrate` / `--update` / `--rollback-engine` 出现时
**不接管启动**，原样交给 setup.py。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
LP_DIR = ROOT / "local" / "logprobs"

# setup.py 自己就完成收尾的开关：见到它们就别接管启动。
HANDS_OFF = ("--setup", "--check", "--no-start", "--calibrate", "--update", "--rollback-engine")


def say(cmd: list) -> None:
    print("[start-server] $ " + " ".join(str(c) for c in cmd), flush=True)


def run(cmd: list, capture: bool = False):
    say(cmd)
    return subprocess.run([str(c) for c in cmd], cwd=str(ROOT), check=False,
                          capture_output=capture, text=capture)


def stock_config() -> str:
    """setup.py 刚写的那份配置（= 最新的、指向原版引擎的 strata-*.json）。"""
    best, best_mtime = None, -1.0
    for p in sorted(ROOT.glob("strata-*.json")):
        if ".bak" in p.name or p.name.startswith("strata-lp-"):
            continue
        try:
            cfg = json.loads(p.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(cfg, dict) and cfg.get("exe") and isinstance(cfg.get("args"), list):
            m = p.stat().st_mtime
            if m > best_mtime:
                best, best_mtime = p, m
    return str(best) if best else ""


def stock_cfg_exe(cfg: str) -> str:
    """The engine path from a model config, for one line of output."""
    try:
        return json.loads(Path(cfg).read_text(encoding="utf-8-sig")).get("exe", cfg)
    except (OSError, ValueError):
        return cfg


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--stock", action="store_true", help="force the stock engine (engine\\strata.exe)")
    ap.add_argument("--logprobs", action="store_true", help="force the logprobs engine, fail if not built")
    ap.add_argument("--port", type=int, default=8080)
    known, setup_args = ap.parse_known_args()

    lp_exe = ROOT / "build-lp" / "local" / "logprobs" / "build" / (
        "strata-lp.exe" if os.name == "nt" else "strata-lp")

    # ---- 1) setup.py: install / tune / write the config, but do NOT start it ----
    setup_cmd = [PY, str(ROOT / "setup.py")] + setup_args
    if not any(a in HANDS_OFF for a in setup_args):
        setup_cmd += ["--no-start"]
    rc = run(setup_cmd).returncode
    if rc != 0:
        print(f"[start-server] setup.py exited with {rc}", file=sys.stderr)
        return rc
    if any(a in HANDS_OFF for a in setup_args):
        return 0

    server = [PY, str(LP_DIR / "server_logprobs.py"), "--engine", "strata",
              "--port", str(known.port)]

    # ---- 2) no logprobs engine built ----
    if not lp_exe.is_file():
        if known.logprobs:
            print(f"[start-server] --logprobs was asked for, but it is not built:\n  {lp_exe}",
                  file=sys.stderr)
            print("                 build it with build-strata-lp.bat (Windows) or build-strata-lp.sh (Linux)",
                  file=sys.stderr)
            return 2
        cfg = stock_config()
        if not cfg:
            print("[start-server] no model config found (a strata-*.json with exe+args)", file=sys.stderr)
            return 2
        print(f"[start-server] {lp_exe.name} not built yet - starting the STOCK engine ({stock_cfg_exe(cfg)})")
        print("                 build it later, then re-run to get logprobs.")
        return run(server + ["--config", cfg]).returncode

    # ---- 3) derive a config whose exe points at strata-lp.exe ----
    out = run([PY, str(LP_DIR / "make_lp_config.py"), "--exe", str(lp_exe)], capture=True)
    lp_cfg = out.stdout.strip().splitlines()[-1].strip() if out.stdout.strip() else ""
    if out.returncode != 0 or not lp_cfg or not Path(lp_cfg).is_file():
        print(f"[start-server] could not derive the logprobs config: {out.stderr.strip()[:400]}",
              file=sys.stderr)
        return 2

    print(f"[start-server] logprobs engine : {lp_exe}")
    print(f"[start-server] config          : {lp_cfg}")
    return run(server + ["--config", lp_cfg]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
