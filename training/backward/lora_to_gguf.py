"""把训练产出的 LoRA state_dict 写成 llama.cpp 的 adapter GGUF，让引擎 `--lora` 能加载。

写什么由 `MAP` 决定，它是**引擎实际能应用的全部七个投影**（`src/kernels/lora_adapters.cpp`
的加载白名单，`src/core/verify.cpp` 的投机路径也按程序顺序逐个 apply）：

    HF `...self_attn.o_proj`        -> GGUF `blk.<l>.attn_output.weight`
    HF `...linear_attn.out_proj`    -> GGUF `blk.<l>.ssm_out.weight`
    HF `...self_attn.q_proj`        -> GGUF `blk.<l>.attn_q.weight`
    HF `...self_attn.k_proj`        -> GGUF `blk.<l>.attn_k.weight`
    HF `...self_attn.v_proj`        -> GGUF `blk.<l>.attn_v.weight`
    HF `...linear_attn.in_proj_qkv` -> GGUF `blk.<l>.attn_qkv.weight`
    HF `...linear_attn.in_proj_z`   -> GGUF `blk.<l>.attn_gate.weight`

两类投影的差别不是「多写几个」，而是**能学到什么**：`y + s·B(Ax)` 加在输出投影上只能改
「这一层输出什么」（退化成词表偏置），只有输入投影才能改「读什么」。只挂输出投影时，
模型无法学「注意到问法里的哪个字段」，只能挪偏置 —— 实测 48 挂点 0/15 对 156 挂点 5/15。

其余键（共享专家、`mtp.*`）仍然**必须丢掉**：引擎的加载器把「不支持的 tensor」当错误，
因为它宁可不加载，也不愿跑半个 adapter（跑半个得到的是既不像 base 也不像微调版的模型）。
所以训练侧的 `--targets` 要落在这七个后缀之内，才谈得上回灌引擎。

形状约定（GGUF 的 ne 是「每维元素数」，dim 0 变化最快）：

    lora_a  ne = [n_in, r]   偏移 (i_in, j_r) = j_r * n_in + i_in  == A 平铺（A 是 (r, n_in) 行主序）
    lora_b  ne = [r, n_out]  偏移 (k_r, j_out) = j_out * r + k     == B 平铺（B 是 (n_out, r) 行主序）

两个都不需要转置，只是 ne 的声明顺序和 numpy 的 shape 直觉相反。
"""

from __future__ import annotations

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))


import argparse
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]          # training/backward -> 仓库根
if not (REPO / "tools" / "gguf_writer.py").exists():
    # 早先这个脚本是从 .workbuddy 下的副本跑的（比仓库深一层），那条路径留给它
    REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
from gguf_writer import GGUFWriter, _Tensor  # noqa: E402

# module 后缀 -> llama.cpp 的 tensor 名（不含 blk.<l>. 前缀）
# Both output projections and input projections: the verify path applies every one of these names in program
# order (src/core/verify.cpp), so a whole-module adapter keeps speculative decoding.  An out-projection-only
# adapter can only shift what a layer emits; `attn_qkv` / `attn_q` are what let it read a different input.
MAP = {
    "self_attn.o_proj": "attn_output",
    "linear_attn.out_proj": "ssm_out",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
}


def write_adapter(sd, out, *, alpha: float = 32.0, arch: str = "qwen4exp",
                  keep=None, allow_empty: bool = False,
                  keep_zero_b: bool = False) -> dict:
    """state_dict -> 引擎能加载的 adapter GGUF。**训练循环里直接调它**（不落 .pt、不起子进程）。

    `keep_zero_b`：B 全零的模块照写（默认跳过，因为 B=0 等于没训）。用在「引擎启动要先有一个
    adapter 才能 `--lora`」这一处：起始点的零 B adapter 与 base 数值等价，训练开始后第一轮 swap
    就把它换掉。跳过它会让引擎无从启动。
    """
    keep = [k.strip() for k in (keep if keep is not None else MAP) if k.strip()]

    # module 名 -> {"A": tensor, "B": tensor}
    mods: dict[str, dict[str, torch.Tensor]] = {}
    skipped: list[str] = []
    for k, v in sd.items():
        if k.endswith(".A"):
            mods.setdefault(k[:-2], {})["A"] = v
        elif k.endswith(".B"):
            mods.setdefault(k[:-2], {})["B"] = v

    layers: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for name, ab in sorted(mods.items()):
        hit = next((m for m in keep if name.endswith(m)), None)
        if hit is None:
            skipped.append(name)
            continue
        if "A" not in ab or "B" not in ab:
            raise SystemExit(f"{name}: only one of A / B present")
        parts = name.split(".")
        li = None
        # `mtp.layers.0.self_attn.o_proj` 会解析出「第 0 层」——引擎的 MTP 块不在 blk.<l> 命名空间里，
        # 那样写会静默覆盖真正的第 0 层。MTP 只是起草层，本来就不该被适配。
        if parts and parts[0] == "mtp":
            skipped.append(name + " (the MTP draft layer: not adapted)")
            continue
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                li = int(parts[i + 1])
        if li is None:
            skipped.append(name)
            continue
        a = ab["A"].float().numpy()   # (r, n_in)
        b = ab["B"].float().numpy()   # (n_out, r)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise SystemExit(f"{name}: A {a.shape} / B {b.shape} do not agree on the rank")
        if float(np.abs(b).max()) == 0.0 and not keep_zero_b:
            skipped.append(name + " (B is zero: never trained)")
            continue
        if li in layers and MAP[hit] in layers[li]:
            raise SystemExit(f"layer {li} {MAP[hit]}: two modules map to one tensor")
        layers.setdefault(li, {})[MAP[hit]] = (a, b)

    n = sum(len(v) for v in layers.values())
    if n == 0 and not allow_empty:
        raise SystemExit(f"no adaptor tensor to write (kept {keep}); the state dict has {len(mods)} modules")

    w = GGUFWriter()
    w.add("general.type", "adapter", "string")
    w.add("general.architecture", arch, "string")
    w.add("adapter.type", "lora", "string")
    w.add("adapter.lora.alpha", float(alpha), "f32")

    total = 0
    for li in sorted(layers):
        for which, (a, b) in sorted(layers[li].items()):
            r, n_in = a.shape
            n_out = b.shape[0]
            base = f"blk.{li}.{which}.weight"
            # SEE the module docstring: the flat bytes are already in GGUF order, only the ne is declared reversed
            w.tensors.append(_Tensor(base + ".lora_a", [n_in, r], "F32",
                                     np.ascontiguousarray(a, dtype="<f4").tobytes()))
            w.tensors.append(_Tensor(base + ".lora_b", [r, n_out], "F32",
                                     np.ascontiguousarray(b, dtype="<f4").tobytes()))
            total += r * n_in + n_out * r

    path = w.write(out)
    ranks = sorted({a.shape[0] for v in layers.values() for a, _ in v.values()})
    return {"path": path, "layers": n, "tensors": len(w.tensors), "ranks": ranks,
            "mib": path.stat().st_size / 2**20, "skipped": skipped}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=32.0, help="训练时的 lora_alpha（s = alpha / rank）")
    ap.add_argument("--arch", default="qwen4exp",
                    help="写进 general.architecture，**必须等于基座 GGUF 的那个值**：引擎拿它和模型比对，"
                         "不一致直接拒载（'made for X, not Y'，宁可不加载也不跑半个 adapter）。"
                         "Swift-Qwen3.8-Flash-Next = qwen4exp；qwen3.6-35b-a3b = qwen35moe")
    ap.add_argument("--keep", default=",".join(MAP), help="要写出的 module 后缀，逗号分隔")
    ap.add_argument("--allow-empty", action="store_true", help="一个可映射的键都没有时也写出（默认报错）")
    ap.add_argument("--keep-zero-b", action="store_true",
                    help="B 全零的模块也写出：给引擎一个能启动的初始 adapter（数值等价于 base）")
    ap.add_argument("--zero-b", action="store_true",
                    help="先把 B 归零再写（隐含 --keep-zero-b）：从一份**训练产物**造出「同结构、"
                         "数值等于基座」的 seed。引擎 `--lora` 启动时加载它，之后热换训练产物 —— "
                         "热换要求每层投影集合不变（CUDA graph 捕获了表的地址），"
                         "拿一个只有输出投影的老 seed 去接 156 挂点的 adapter 会被拒："
                         "`layer 0 changed its projection set`")
    args = ap.parse_args()

    sd = torch.load(args.pt, map_location="cpu")
    if args.zero_b:
        n_zero = 0
        for k in list(sd):
            if k.endswith(".B"):
                sd[k] = torch.zeros_like(sd[k])
                n_zero += 1
        if not n_zero:
            raise SystemExit(f"--zero-b: {args.pt} 里没有 .B 张量，按错文件处理")
    r = write_adapter(sd, args.out,
                      alpha=args.alpha, arch=args.arch,
                      keep=args.keep.split(","), allow_empty=args.allow_empty,
                      keep_zero_b=args.keep_zero_b or args.zero_b)
    print(f"wrote {r['path']}: {r['layers']} layer(s) adapted, {r['tensors']} tensors, "
          f"rank {r['ranks']}, {r['mib']:.1f} MiB, alpha {args.alpha:g} "
          f"(s = alpha/rank = {args.alpha / r['ranks'][-1]:.3f} for the widest rank)")
    if r["skipped"]:
        print(f"dropped {len(r['skipped'])} module(s) the engine does not adapt:")
        for s in r["skipped"][:10]:
            print("  ", s)
        if len(r["skipped"]) > 10:
            print(f"   ... {len(r['skipped']) - 10} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
