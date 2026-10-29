"""验证 fla GDN 的 autotune config 会改变数值（不需要加载模型，几十秒）。

背景：T=15 的 CE 跨进程仍差 0.0018 nats，逐模块指纹定位到 layer 30 的
`linear_attn.norm`，即 GDN 内部。fla 的 `chunk_fwd_kernel_o` 用 @triton.autotune，候选
config 的 **BK/BV 块大小**不同（128/128、64/64、32/32）——块大小改变 tl.dot 沿 K 的切分
就是改变累加顺序。autotune 按实测耗时选优，未命中时进程间可能选中不同 config。

本脚本把同一份输入的每种 config 各跑一次，比较输出是否逐位相同。

用法：python3 _mb_gdn_det.py
"""
from __future__ import annotations

import torch

torch.manual_seed(0)
dev = "cuda"

from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: E402
from fla.ops.common import chunk_o, chunk_delta_h  # noqa: E402

H, K, V = 48, 128, 128          # 与 qwen4exp 的 GDN 头数/头维同阶
T = 15
BT = 16

q = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, T, H, V, device=dev, dtype=torch.bfloat16)
g = -torch.rand(1, T, H, device=dev, dtype=torch.float32)
beta = torch.rand(1, T, H, device=dev, dtype=torch.float32)


def run():
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta, head_first=False)
    torch.cuda.synchronize()
    return o


def find_tuners(module: str):
    """fla 的装饰顺序是 heuristics -> autotune -> jit，模块属性是 Heuristics，
    Autotuner 在它的 .fn 链上。"""
    import importlib

    m = importlib.import_module(module)
    out = []
    for attr in dir(m):
        obj = getattr(m, attr)
        for _ in range(4):
            if isinstance(getattr(obj, "configs", None), list):
                out.append((f"{module.split('.')[-1]}.{attr}", obj))
                break
            nxt = getattr(obj, "fn", None)
            if nxt is None or nxt is obj:
                break
            obj = nxt
    return out


print("候选 config（chunk_o / chunk_delta_h）：")
targets = []
for modname in ("fla.ops.common.chunk_o", "fla.ops.common.chunk_delta_h"):
    for label, tuner in find_tuners(modname):
        cfgs = list(tuner.configs)
        print(f"  {label}: {len(cfgs)} 个 -> {[c.kwargs for c in cfgs]}")
        if len(cfgs) > 1:
            targets.append((label, tuner, cfgs))

# 让 autotune 正常选一次
ref = run().float()
import hashlib  # noqa: E402
print(f"\nautotune 默认输出 digest "
      f"{hashlib.blake2b(ref.cpu().numpy().tobytes(), digest_size=6).hexdigest()}")

# 逐个把 config 钉死，看输出是否相同
for label, tuner, cfgs in targets:
    orig = list(tuner.configs)
    outs = []
    for c in cfgs:
        tuner.configs = [c]
        tuner.cache.clear()
        outs.append(run().float().cpu())
    tuner.configs = orig
    tuner.cache.clear()
    base = outs[0]
    worst = max(float((o - base).abs().max()) for o in outs[1:])
    eq = all(torch.equal(o, base) for o in outs[1:])
    print(f"{label}: {len(cfgs)} 个 config  max|Δ|={worst:.3e}  "
          f"{'逐位相同' if eq else '*** 数值不同 ***'}")
