"""跨进程逐模块指纹：定位「哪个 module 的输出在两个进程里不同」。

已知（2026-10-29）：
  - 同进程内 3 次前向逐位一致（_det_short.py digest SAME）；
  - 跨进程三个用例全部不同；
  - 1069 个参数/缓冲的指纹跨进程**完全一致**（_det_weights.py）。
⇒ 权重没问题，输入没变，op 在同进程内确定。剩下的解释只有一个：某个 kernel 读了
  未初始化显存 / 依赖分配器给到的内存内容（跨进程内容不同，同进程内被复用所以稳定）。

本脚本给每个子模块的 forward 输出算指纹并按调用顺序记录，两个进程各存一份 JSON，
再 diff 出**最早分叉的模块**。

用法：
    python3 _det_layers.py --out /root/l_1.json
    python3 _det_layers.py --out /root/l_2.json
    python3 _det_layers.py --diff /root/l_1.json /root/l_2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json

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


def first_tensor(o, depth=0):
    if isinstance(o, torch.Tensor):
        return o
    if depth > 3:
        return None
    if isinstance(o, (tuple, list)):
        for x in o:
            t = first_tensor(x, depth + 1)
            if t is not None:
                return t
    if isinstance(o, dict):
        for x in o.values():
            t = first_tensor(x, depth + 1)
            if t is not None:
                return t
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--diff", nargs=2, default=None)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--gguf", nargs="+", default=GGUF)
    ap.add_argument("--pack", default=PACK)
    ap.add_argument("--tokenizer", default=TOK_DIR)
    args = ap.parse_args()

    if args.diff:
        a, b = json.load(open(args.diff[0])), json.load(open(args.diff[1]))
        seq_a, seq_b, da, db = a["seq"], b["seq"], a["dig"], b["dig"]
        if len(seq_a) != len(seq_b):
            print(f"调用序列长度不同: {len(seq_a)} vs {len(seq_b)}")
        n = min(len(seq_a), len(seq_b))
        bad = []
        for i in range(n):
            k = seq_a[i]
            if k != seq_b[i]:
                print(f"[{i}] 调用顺序不同: {k} vs {seq_b[i]}")
                break
            if da.get(k) != db.get(k):
                bad.append((i, k))
        print(f"共 {n} 次 module 调用，指纹不同 {len(bad)} 次")
        # 第一个分叉 + 它的直接子模块分叉情况（区分「自己算错」还是「子模块传上来的」）
        for i, k in bad[:12]:
            print(f"  [{i}] {k}")
        if bad:
            i0, k0 = bad[0]
            print(f"\n最早分叉: 调用 #{i0} {k0}")
            kids = [kk for _, kk in bad if kk.startswith(k0 + ".")]
            print(f"  该模块自身输出不同；其子模块分叉 {len(kids)} 个"
                  + (f"，例如 {kids[:3]}" if kids else " ⇒ **自己算的**（与子模块无关）"))
        return 0

    model, _ = load_model(args.gguf, args.pack)
    model.eval()
    model.config.use_cache = False
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    body = (getattr(model, "model", None) or model)
    inner = getattr(body, "language_model", body)

    text = ("The capital of France is Paris. 1 2 3 4 5 6 7 8 9 10. "
            "我是人工智能助手，很高兴见到你。MoE 路由决定每个 token 命中哪些专家。") * 4
    ids = tok(text, return_tensors="pt").input_ids[:, : args.tokens].cuda()

    seq: list[str] = []
    dig: dict[str, list[str]] = {}
    depth_of: dict[str, int] = {}

    def mk(name):
        def hook(mod, inp, out):
            t = first_tensor(out)
            seq.append(name)
            if t is None:
                dig.setdefault(name, []).append("-")
                return
            b = t.detach().to("cpu", non_blocking=False).contiguous()
            dig.setdefault(name, []).append(hashlib.blake2b(
                b.view(torch.uint8).reshape(-1).numpy().tobytes(), digest_size=6).hexdigest())
        return hook

    handles = []
    for name, mod in model.named_modules():
        if not name:
            continue
        depth_of[name] = name.count(".")
        handles.append(mod.register_forward_hook(mk(name)))

    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()
    for h in handles:
        h.remove()

    json.dump({"seq": seq, "dig": dig, "depth": depth_of}, open(args.out, "w"))
    print(f"{len(seq)} 次调用 / {len(dig)} 个模块 -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
