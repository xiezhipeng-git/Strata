"""LoRA adapter 的磁盘格式与几何（GGUF 侧）。

引擎的 `LORA <file>` 换表要求张量集合与形状全程不变，所以 rank 在训练期间固定；
写盘走 tools/gguf_writer.py（与引擎读的那份格式同源）。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))
from gguf_writer import GGUFWriter, _Tensor  # noqa: E402

def read_geometry(gguf_path: str) -> dict:
    from gguf_reader import GGUFFile

    f = GGUFFile(gguf_path)
    arch = str(f.metadata.get("general.architecture") or "")
    need = ["block_count", "embedding_length", "attention.head_count", "attention.key_length",
            "full_attention_interval", "ssm.inner_size"]
    out = {}
    for k in need:
        v = f.metadata.get(f"{arch}.{k}", f.metadata.get(k))
        if not isinstance(v, (int, float)):
            raise SystemExit(f"{gguf_path}: metadata '{arch}.{k}' missing (need the base model's geometry)")
        out[k] = int(v)
    # 每层各目标矩阵的真实形状（ne0 = n_in，ne1 = n_out，与引擎 layout.cpp 的 want2 同一口径）。
    # 不从 metadata 推导（ssm_conv_channels 在引擎里是派生值，推错了就是静默的 adapter 形状错误），
    # 直接量基座文件 —— 量出来什么就是什么。基座可能是多分片 GGUF（本文件只见一部分层），
    # 所以按**家族**收集：同族（QSA / GDN）各层形状一致（引擎 layout.cpp 就按同一张 want2 表
    # 逐层校验），量到一族的一层即可展开到全模型。
    targets = ("attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate", "ssm_out",
               "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")
    fam_dims: dict[str, dict[str, tuple[int, int]]] = {}
    for t in f.tensors:
        if not t.name.startswith("blk.") or len(t.shape) != 2:
            continue
        parts = t.name.split(".")
        if len(parts) != 4 or parts[3] != "weight" or parts[2] not in targets:
            continue
        try:
            l = int(parts[1])
        except ValueError:
            continue
        # 家族归属：ssm_out/attn_output 自带家族信息；attn_q/k/v 只在 QSA 层；attn_qkv/gate 只在 GDN 层；
        # ffn_*_shexp 每层都有（shared expert），单独一族、逐层合并
        if parts[2] == "ssm_out":
            fam = "gdn"
        elif parts[2] == "attn_output" or parts[2] in ("attn_q", "attn_k", "attn_v"):
            fam = "qsa"
        elif parts[2].endswith("_shexp"):
            fam = "shexp"
        else:
            fam = "gdn"
        dim = (int(t.shape[0]), int(t.shape[1]))
        seen = fam_dims.setdefault(fam, {}).get(parts[2])
        if seen is not None and seen != dim:
            raise SystemExit(f"{gguf_path}: blk.{l}.{parts[2]}.weight ne={dim} 与之前量到的 {seen} 不一致"
                             "（同族各层形状应一致）")
        fam_dims.setdefault(fam, {})[parts[2]] = dim
    if fam_dims:
        # 两条底线自检：输出投影的 out 维、输入投影的 in 维必须等于 n_embd，
        # 对不上说明 GGUF 的 ne 口径和假设相反 —— 宁可现在死也不要写出形状错误的 adapter。
        n = out["embedding_length"]
        for fam, ds in fam_dims.items():
            for w, (ni, no) in ds.items():
                if w in ("attn_output", "ssm_out") and no != n:
                    raise SystemExit(f"{gguf_path}: {w} ne=({ni},{no})，但 n_embd={n}（ne 口径不符）")
                if w in ("attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate") and ni != n:
                    raise SystemExit(f"{gguf_path}: {w} ne=({ni},{no})，但 n_embd={n}（ne 口径不符）")
        interval = max(int(out["full_attention_interval"]), 2)
        missing_fams = [fam for fam in ("qsa", "gdn") if fam not in fam_dims]
        if missing_fams:
            raise SystemExit(f"{gguf_path}: 本分片没量到 {'/'.join(missing_fams)} 层的投影张量，"
                             "无法展开全模型布局（换任一含该族层的分片，或用 --gguf 指到另一片）")
        # ffn_*_shexp 每层都有，逐层并进该层的投影集合（有的模型没有 shared expert，那就只是缺席）。
        shexp = fam_dims.get("shexp", {})
        out["layer_dims"] = {
            l: {**(fam_dims["qsa"] if l % interval == interval - 1 else fam_dims["gdn"]), **shexp}
            for l in range(out["block_count"])
        }
    return out


_TARGETS_QSA = ("attn_q", "attn_k", "attn_v", "attn_output")
_TARGETS_GDN = ("attn_qkv", "attn_gate", "ssm_out")


_TARGETS_SHEXP = ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")


def adapter_layout(g: dict, projs: str = "out") -> dict[int, dict[str, tuple[int, int]]]:
    """layer -> {投影名 -> (n_in, n_out)}。qsa 判定与引擎 is_qsa_layer 同一条公式（layer % interval == interval-1）。

    `projs`："out"（默认，历史路径）= 每层只挂输出投影（ssm_out / attn_output）；
    "all" = 对齐 InferZO 的全模块口径（QSA: attn_q/k/v/output，GDN: attn_qkv/gate/ssm_out）；
    "ffn" = 每层 shared expert 的三件（gate/up/down，**专家 FFN 的入口**）；
    "all+ffn" = 上面两者相加。
    形状来源：真实模型走 `read_geometry` 量出的 `layer_dims`；合成 GEOM（测试）只支持 "out"，
    其派生公式与历史逐字相同。"""
    L, N = g["block_count"], g["embedding_length"]
    interval = max(int(g["full_attention_interval"]), 2)
    n_qsa_in = g["attention.head_count"] * g["attention.key_length"]
    n_gdn_in = g["ssm.inner_size"]
    layer_dims = g.get("layer_dims") or {}
    attn = projs in ("out", "all", "all+ffn")
    ffn = projs in ("ffn", "all+ffn")
    if projs not in ("out", "all", "ffn", "all+ffn"):
        raise SystemExit(f"--zo-projs {projs!r} 不认识（out / all / ffn / all+ffn）")
    if ffn and not layer_dims:
        raise SystemExit("--zo-projs ffn 需要真实模型的 layer_dims（read_geometry 量出）；"
                         "合成 GEOM 没有 shared expert")
    out = {}
    for l in range(L):
        qsa = l % interval == interval - 1
        fam = _TARGETS_QSA if qsa else _TARGETS_GDN
        want: tuple[str, ...] = ()
        if attn:
            want += (fam[3] if qsa else fam[2],) if projs == "out" else fam
        if ffn:
            want += _TARGETS_SHEXP
        if layer_dims:
            ds = layer_dims.get(l)
            if ds is None:
                raise SystemExit(f"base model: blk.{l} 没有可量形状的投影张量（qsa={qsa}）")
            missing = [w for w in want if w not in ds]
            if missing:
                raise SystemExit(f"base model: blk.{l} 缺少 {', '.join(missing)} 的张量（qsa={qsa}）")
            out[l] = {w: ds[w] for w in want}
        else:
            if projs != "out":
                raise SystemExit("--zo-projs all 需要真实模型的 layer_dims（read_geometry 量出），"
                                 "合成 GEOM 只支持 out")
            which = fam[3] if qsa else fam[2]
            n_in = n_qsa_in if qsa else n_gdn_in
            out[l] = {which: (n_in, N)}
    return out


def read_adapter_rank(gguf_path: str) -> int:
    """从 adapter GGUF 读 LoRA 的秩（`lora_a` 声明成 [n_in, rank]）。

    引擎在 `--lora` 启动时就把 shape 捕获进 CUDA graph，运行期 `LORA <path>` 只能换
    **同 shape** 的 adapter（`layer 0 changed shape (the tables' addresses are held by
    captured graphs)`）。探针要测训练后的 adapter，就必须用同 rank 的 B=0 适配器起引擎，
    而不是把 rank 写死成 1。
    """
    from gguf_reader import GGUFFile

    f = GGUFFile(gguf_path)
    for t in f.tensors:
        if t.name.endswith(".lora_a"):
            return int(t.shape[1])
    raise SystemExit(f"{gguf_path}: no lora_a tensor, cannot determine the rank")


def read_adapter_alpha(gguf_path: str) -> float:
    """同一个 adapter 的 `adapter.lora.alpha`（写 B=0 对照组时保持一致）。"""
    from gguf_reader import GGUFFile

    return float(GGUFFile(gguf_path).metadata.get("adapter.lora.alpha", 16.0))


# `write_adapter` 的存储精度。**为什么默认 F16 而不是 F32**（2026-11-18 离线实测，
# `.workbuddy/tmp/idrun/zobench/quant_err.py`，all/allmom 两份 rank-8 adapter，156 个投影）：
# 以 ΔW = s·B·A 的相对误差为准（logp 的一阶位移只由它决定，A/B 的元素误差会被 rank 平均）：
#   F32 0 · **F16 2.9e-4** · BF16 2.4e-3 · Q8 7.6e-3 · Q4 9.5e-2
# BF16 比 F16 差 8×（8 位尾数），adapter 的 A/B 在 1/sqrt(n_in)≈0.02 量级，正好卡在 bf16 的相对精度上，
# 所以"引擎只认 f32/f16/bf16"里对 adapter 唯一该用的是 F16。Q4 的 9.5% 与一步训练的步长同量级
# （rel 2–5%，|pj| 中位数 0.087）⇒ 4bit 会把差分吃掉，训练不可用；仅推理驻留才考虑。
ADAPTER_DTYPES = {"f32": "F32", "f16": "F16", "bf16": "BF16"}


def write_adapter(path: str, layout: dict, theta: dict, alpha: float, arch: str = "qwen4exp",
                  dtype: str = "f16"):
    """theta[layer][which] = (A (r, n_in), B (n_out, r))，写 llama.cpp 的 adapter GGUF。
    flat bytes 已在 GGUF 顺序（见 lora_to_gguf.py 的 docstring）：A 平铺、B 平铺，只把 ne 反着声明。

    `arch`：引擎按 `general.architecture` 拒收"给别的模型做的 adapter"（`load_lora_adapters`），
    所以要写模型自己的名字（`qwen4exp` / `qwen35moe`）。默认值就是这条线一直跑的那一个。

    `dtype`：`f32` / `f16` / `bf16`，默认 f16（见 ADAPTER_DTYPES 的实测表）。llama.cpp 的 adapter
    工具链认这三种，所以产物仍然可被外部工具合成（引擎自己**没有** merge 路径 —— `LORA <path>` 只换表）。
    量化只在运行时表那一侧（`--lora-quant`），不落在产物上：量化误差一旦写进文件，合成时会永久进权重。"""
    tag = ADAPTER_DTYPES.get(dtype.lower())
    if tag is None:
        raise SystemExit(f"adapter dtype {dtype!r} 不支持（{'/'.join(ADAPTER_DTYPES)}）")

    def bytes_of(x):
        if tag == "F32":
            return np.ascontiguousarray(x, dtype="<f4").tobytes()
        if tag == "F16":
            return np.ascontiguousarray(x, dtype="<f2").tobytes()
        f = np.ascontiguousarray(x, dtype="<f4").view(np.uint32)
        lsb = (f >> 16) & 1
        return ((f + 0x7FFF + lsb) & 0xFFFF0000).astype("<u4").tobytes()

    w = GGUFWriter()
    w.add("general.type", "adapter", "string")
    w.add("general.architecture", arch, "string")
    w.add("adapter.type", "lora", "string")
    w.add("adapter.lora.alpha", float(alpha), "f32")
    for l in sorted(layout):
        for which, (n_in, n_out) in layout[l].items():
            a, b = theta[l][which]
            base = f"blk.{l}.{which}.weight"
            w.tensors.append(_Tensor(base + ".lora_a", [n_in, a.shape[0]], tag, bytes_of(a)))
            w.tensors.append(_Tensor(base + ".lora_b", [b.shape[1], n_out], tag, bytes_of(b)))
    w.write(path)


def read_adapter(path: str, layout: dict) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """把 adapter GGUF 里的 A/B 读回成 θ（`write_adapter` 的逆）。

    **为什么必须有这条路径**（2026-11-16 实测）：`mezo.py --init-adapter` 原来只把文件交给
    引擎 `--lora`，进程内的 θ 仍是 `init_theta()` 的 B=0。于是 DPO 的 ref（θ₀ 缓存）落在
    SFT 的 θ 上，而 policy 侧每一步 `swap(θ±εb)` 写的是「B≈0 的骨架 + 噪声」，等于拿基座
    在训。日志上完全看不出来：pj 正常、‖θ‖ 正常涨、2000 步跑完 rc=0，只是 probe 从
    −51.69 起步（SFT 终点是 −21.15）、answer-ce 3.19（SFT 终点 0.30）。
    `training/coldstart.py` 的注释里早就记着这个坑（"只把 GGUF 交给引擎加载、不回读 A/B"），
    没有代码去补，就一直在。

    平铺顺序与 `write_adapter` 对称：A 的字节是 `(rank, n_in)` 行主序、声明成 `[n_in, rank]`，
    B 的字节是 `(n_out, rank)`、声明成 `[rank, n_out]` —— 所以两条都按 `shape[::-1]` reshape。
    """
    from gguf_reader import GGUFFile

    f = GGUFFile(path)
    tmap = {t.name: t for t in f.tensors}
    blob = pathlib.Path(path).read_bytes()
    base = f.data_start
    theta: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for l, projs in sorted(layout.items()):
        theta[l] = {}
        for which, (n_in, n_out) in projs.items():
            name = f"blk.{l}.{which}.weight"
            pair = []
            for suffix, want in ((".lora_a", (n_in, None)), (".lora_b", (None, n_out))):
                t = tmap.get(name + suffix)
                if t is None:
                    raise SystemExit(f"{path}: 缺少张量 {name + suffix}（与 layout 不匹配 —— 检查 --zo-projs："
                                     "这个 adapter 是不是按另一种投影集合写的）")
                if t.type_name not in ("F32", "F16", "BF16"):
                    raise SystemExit(f"{path}: {name + suffix} 是 {t.type_name}，本读取器只处理 F32/F16/BF16")
                if len(t.shape) != 2:
                    raise SystemExit(f"{path}: {name + suffix} 维度是 {t.shape}，期望二维")
                n = int(t.shape[0]) * int(t.shape[1])
                w = 4 if t.type_name == "F32" else 2
                raw = blob[base + t.offset: base + t.offset + w * n]
                if len(raw) != w * n:
                    raise SystemExit(f"{path}: {name + suffix} 数据被截断")
                if t.type_name == "F32":
                    vals = np.frombuffer(raw, dtype="<f4")
                elif t.type_name == "F16":
                    vals = np.frombuffer(raw, dtype="<f2").astype(np.float32)
                else:   # BF16：高 16 位
                    vals = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
                pair.append(vals.reshape(t.shape[1], t.shape[0]).astype(np.float32, copy=True))
            a, b = pair
            # a 是 (rank, n_in)、b 是 (n_out, rank)：A 的最后维、B 的第一维对着几何，
            # 另一维必须同 rank，两处任一对不上都说明这个 adapter 不属于这个模型。
            if a.shape[1] != n_in or b.shape[0] != n_out or a.shape[0] != b.shape[1]:
                raise SystemExit(f"{path}: {name} 形状 A{a.shape} B{b.shape} 与几何 "
                                 f"(n_in {n_in}, n_out {n_out}, 同 rank) 不符")
            theta[l][which] = (a, b)
    return theta


def init_theta(layout: dict, rank: int, seed: int) -> dict:
    """A ~ N(0, 1/sqrt(n_in))（kaiming 的 LoRA 惯例），B = 0：B=0 的 adapter 与 base 逐字相同，
    所以训练前的任何读法都是 base 的读法（ab_identity 的 zero 臂就是这个含义）。"""
    rng = np.random.default_rng(seed)
    theta = {}
    for l, projs in layout.items():
        theta[l] = {}
        for which, (n_in, n_out) in projs.items():
            a = rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(rank, n_in)).astype(np.float32)
            b = np.zeros((n_out, rank), dtype=np.float32)
            theta[l][which] = (a, b)
    return theta


