"""前向确定性诊断：找出「同输入两次前向结果不同」的第一处来源。

背景（2026-10-29）：dbg_forward.py 同版代码连跑两次，CE(' Paris') 0.474 vs 0.409、
CE('，') 1.192 vs 1.082 —— run-to-run 波动 0.065~0.11 nats，比任何要比较的改动差值
都大一个量级。不先钉死确定性，后面所有「没算错」的验证都是空的。

本脚本做两件事：
1) 逐层比对两次前向的残差流，报出**第一个分叉的层**和分叉幅度 —— 区分是「某一层
   的 op 非确定」还是「整条链路的 bf16 舍入被放大」；
2) 在 use_deterministic_algorithms(True, warn_only=True) 下跑一次，让 PyTorch 自己
   点名哪些 op 没有确定性实现（index_add_ / scatter_add_ / 某些 cuBLAS 路径等）。

用法：
    python3 _det_check.py --gguf a.gguf b.gguf --pack /root/experts_nf4.bin [--tokens 256]
"""
from __future__ import annotations

import argparse
import warnings

import torch

from loader_qwen4 import load_model

GGUF = [
    "/mnt/d/Admin/Documents/GitHub/Strata-data/models/swift-IQ3_XXS/"
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    "/mnt/d/Admin/Documents/GitHub/Strata-data/models/swift-IQ3_XXS/"
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
]
PACK = "/root/experts_nf4.bin"
TOK_DIR = "/root/models/qwen36-35b-a3b"


def diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, int]:
    """返回 (max|Δ|, 分叉元素个数)。"""
    d = (a.float() - b.float()).abs()
    return float(d.max()), int((d > 0).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", nargs="+", default=GGUF)
    ap.add_argument("--pack", default=PACK)
    ap.add_argument("--tokenizer", default=TOK_DIR)
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()

    model, _ = load_model(args.gguf, args.pack)
    model.eval()
    model.config.use_cache = False
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    body = (getattr(model, "model", None) or model)
    inner = getattr(body, "language_model", body)

    text = ("The capital of France is Paris. 1 2 3 4 5 6 7 8 9 10. "
            "我是人工智能助手，很高兴见到你。MoE 路由决定每个 token 命中哪些专家。") * 8
    ids = tok(text, return_tensors="pt").input_ids[:, : args.tokens].cuda()
    T = ids.shape[1]
    print(f"input: {T} tokens", flush=True)

    layers = inner.layers
    print(f"layers: {len(layers)}", flush=True)

    # ---- 1) 逐层比对 ----
    caps: list[list[torch.Tensor]] = []

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        caps[-1].append(h.detach().clone())

    handles = [ly.register_forward_hook(hook) for ly in layers]

    def run() -> torch.Tensor:
        caps.append([])
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits
        torch.cuda.synchronize()
        return logits.float()

    lg1 = run()
    lg2 = run()
    for h in handles:
        h.remove()

    print(f"\n=== 两次前向（同进程、同输入）===")
    print(f"logits  max|Δ| = {diff(lg1, lg2)[0]:.6e}   "
          f"分叉元素 {diff(lg1, lg2)[1]}/{lg1.numel()}")
    print(f"argmax 一致 = {bool((lg1.argmax(-1) == lg2.argmax(-1)).all())}")

    a, b = caps[0], caps[1]
    assert len(a) == len(b) == len(layers), (len(a), len(b), len(layers))
    first = None
    print(f"\n{'层':>4s} {'max|Δ|':>12s} {'分叉元素':>10s}  {'相对幅度':>10s}")
    for i, (x, y) in enumerate(zip(a, b)):
        mx, n = diff(x, y)
        amp = mx / (float(x.float().abs().max()) + 1e-9)
        if n and first is None:
            first = i
        if i < 6 or n and i < (first or 0) + 6 or i >= len(a) - 3:
            print(f"{i:4d} {mx:12.3e} {n:10d}  {amp:10.3e}")
    print(f"\n第一个分叉层: {first if first is not None else '无（逐层完全一致）'}")

    # ---- 2) 让 PyTorch 点名非确定 op ----
    print("\n=== use_deterministic_algorithms(True, warn_only=True) ===", flush=True)
    msgs: dict[str, int] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            with torch.no_grad():
                model(input_ids=ids, use_cache=False).logits
            torch.cuda.synchronize()
        finally:
            torch.use_deterministic_algorithms(False)
    for w in caught:
        m = str(w.message).splitlines()[0][:150]
        msgs[m] = msgs.get(m, 0) + 1
    if msgs:
        for m, c in sorted(msgs.items(), key=lambda kv: -kv[1]):
            print(f"  x{c:<5d} {m}")
    else:
        print("  （没有警告：所有 op 都有确定性实现 ⇒ 差异来自 cuBLAS/Triton 内部的累加顺序）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
