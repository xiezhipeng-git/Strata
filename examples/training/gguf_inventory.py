"""Dump the full GGUF inventory (kv + tensor names/types/dims/offsets) to JSON.

事实来源：qwen4exp 的 GGUF→HF 映射以此为准。shard 2 是 PLE 表（AMD_HIP.md 的 --ple-gguf）。

用法：python gguf_inventory.py <model.gguf> [out.json]
"""

from __future__ import annotations

import json
import struct
import sys


def read_header(path: str) -> tuple[dict, list[dict]]:
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", "not a GGUF file"
    ver = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    assert ver == 3, f"gguf v{ver} not handled"

    def rd_str() -> str:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def rd_val(t: int):
        if t in (0, 1, 7):
            return struct.unpack("<B", f.read(1))[0]
        if t in (2, 3):
            return struct.unpack("<H", f.read(2))[0]
        if t in (4, 5):
            return struct.unpack("<I", f.read(4))[0]
        if t == 6:
            return struct.unpack("<f", f.read(4))[0]
        if t == 8:
            return rd_str()
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [rd_val(et) for _ in range(n)]
        if t in (10, 11):
            return struct.unpack("<Q", f.read(8))[0]
        if t == 12:
            return struct.unpack("<d", f.read(8))[0]
        raise ValueError(f"gguf kv type {t}")

    kv = {}
    for _ in range(n_kv):
        k = rd_str()
        t = struct.unpack("<I", f.read(4))[0]
        kv[k] = rd_val(t)

    tensors = []
    for _ in range(n_tensors):
        name = rd_str()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]  # ne, dim0 fastest
        ttype = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        tensors.append({"name": name, "ne": dims, "type": ttype, "offset": off})
    return kv, tensors


def main() -> int:
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else path + ".inventory.json"
    kv, tensors = read_header(path)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"path": path, "kv": kv, "tensors": tensors}, f, ensure_ascii=False, indent=1)
    types: dict[int, int] = {}
    for t in tensors:
        types[t["type"]] = types.get(t["type"], 0) + 1
    print(f"{path}: {len(tensors)} tensors, {len(kv)} kv -> {out}")
    print("types:", dict(sorted(types.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
