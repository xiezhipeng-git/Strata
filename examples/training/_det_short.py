"""short-T 确定性复现：dbg_forward.py 的同一批用例，同进程内重复两次。

_det_check.py 在 T=256 上两次前向逐位一致（48 层全 0），但 dbg_forward.py 跨进程
两次运行 CE 差 0.065~0.11 nats。差别有两个维度：T（256 vs 8~30）和进程。
本脚本固定 T，先排除「同进程内就不可复现」；再配合两次进程调用排除「跨进程」。

用法：python3 _det_short.py [--reps 2]
"""
from __future__ import annotations

import argparse
import hashlib

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

CASES = [
    ("The capital of France is", " Paris"),
    ("1 2 3 4 5 6 7", " 8"),
    ("我是人工智能助手", "，"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--gguf", nargs="+", default=GGUF)
    ap.add_argument("--pack", default=PACK)
    ap.add_argument("--tokenizer", default=TOK_DIR)
    args = ap.parse_args()

    model, _ = load_model(args.gguf, args.pack)
    model.eval()
    model.config.use_cache = False
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    body = (getattr(model, "model", None) or model)
    inner = getattr(body, "language_model", body)
    head = model.lm_head

    @torch.no_grad()
    def ce_of(prefix: str, cont: str) -> tuple[float, torch.Tensor]:
        p = tok(prefix, add_special_tokens=False).input_ids
        c = tok(cont, add_special_tokens=False).input_ids
        ids = torch.tensor([p + c], device="cuda")
        h = inner(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
        labels = ids[0, 1:]
        logits = head(h[-len(c):]).float()
        lp = torch.log_softmax(logits, dim=-1)
        picked = lp.gather(-1, labels[-len(c):, None]).squeeze(-1)
        return float(-picked.mean()), logits

    print(f"reps={args.reps}", flush=True)
    for prefix, cont in CASES:
        vals, hashes = [], []
        for _ in range(args.reps):
            ce, logits = ce_of(prefix, cont)
            vals.append(ce)
            hashes.append(hashlib.sha256(
                logits.contiguous().cpu().numpy().tobytes()).hexdigest()[:16])
            ntok = len(tok(prefix, add_special_tokens=False).input_ids
                       + tok(cont, add_special_tokens=False).input_ids)
        same = len(set(hashes)) == 1
        print(f"CE({cont!r}) = " + " / ".join(f"{v:.4f}" for v in vals)
              + f"   T={ntok}  digest {'SAME' if same else 'DIFF'} {hashes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
