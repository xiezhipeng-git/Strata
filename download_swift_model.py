#!/usr/bin/env python3
"""Download the Strata Swift 1.5 IQ3_XXS GGUF (model shards + vision encoder) from ModelScope.

Why this exists
--------------
Strata's setup.py only downloads models from HuggingFace, which is unreachable from this
PC (TLS/SSL handshake failure -> "502 Bad Gateway"). ModelScope (modelscope.cn) mirrors the
EXACT same GGUF files and is reachable on the domestic network, so we pull them locally and
then run Strata with --gguf-dir so it uses them instead of fetching from HuggingFace.

Behaviour
---------
- Resumable and safe to re-run: a file is skipped only if it exists AND its byte size is
  exactly the expected value (so a half-written 76 GB file is never mistaken for "done").
- Needs a Python that has the `modelscope` SDK (the system Python 3.13 has it); the Strata
  .venv does NOT, so this script is launched with `py -3.13` from START-HERE.bat.
"""
import os
import sys
from pathlib import Path

REPO = "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
REVISION = "master"

# Files Strata expects for Swift 1.5, IQ3_XXS, with images on (mmproj = vision encoder).
FILES = [
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
    "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf",
]

# Exact byte sizes from the ModelScope repo listing (used to detect partial downloads).
EXPECTED_SIZES = {
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf": 39785790560,
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf": 36180282560,
    "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf": 907543520,
}

# Default target: the STANDARD Strata data folder, OUTSIDE the repo (one level up from
# this script's folder). setup.py keeps its model data at <repo>\..\Strata-data, and at
# startup moves any in-repo Strata-data out to there. Downloading straight into the
# standard folder avoids that move and keeps --gguf-dir pointing at the real files.
# Override with the STRATA_MODELS_DIR environment variable (what START-HERE-modelscope.bat sets).
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "Strata-data" / "models" / "swift-IQ3_XXS"


def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}"
        f /= 1024
    return str(n)


def present(fname: str, dest: Path) -> bool:
    p = dest / fname
    if not p.exists():
        return False
    if EXPECTED_SIZES.get(fname) and p.stat().st_size != EXPECTED_SIZES[fname]:
        print(f"  PARTIAL  {fname} ({human(p.stat().st_size)} != {human(EXPECTED_SIZES[fname])}) -> will re-download")
        return False
    return True


def main() -> int:
    dest = Path(os.environ.get("STRATA_MODELS_DIR") or DEFAULT_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[modelscope] target dir: {dest}")

    status = {f: present(f, dest) for f in FILES}
    for f, ok in status.items():
        print(f"  {'OK     ' if ok else 'MISSING '} {f}" + (f"  ({human(EXPECTED_SIZES[f])})" if not ok else ""))
    if all(status.values()):
        print("[modelscope] all 3 files present and complete - nothing to download.")
        return 0

    try:
        from modelscope import snapshot_download
    except ImportError:
        print("[modelscope] ERROR: 'modelscope' SDK not found in this Python.")
        print("  Install it with:  py -3.13 -m pip install modelscope")
        print("  (the download must run with a Python that has modelscope, e.g. the system Python 3.13)")
        return 1

    print("[modelscope] downloading from ModelScope (resumable - safe to re-run) ...")
    snapshot_download(
        REPO,
        revision=REVISION,
        local_dir=str(dest),
        allow_patterns=["Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS*", "mmproj-Swift*"],
    )

    missing = [f for f in FILES if not present(f, dest)]
    if missing:
        print(f"[modelscope] WARNING: still missing/incomplete after download: {missing}")
        return 1
    print("[modelscope] download complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
