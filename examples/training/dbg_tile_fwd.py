"""同一进程内对比 GDN key-head 两种配对约定的功能性判据（不依赖引擎）。

  interleave (HF transformers 5.17)：值头 j 配键头 j//3
  tile       (Strata 内核)          ：值头 j 配键头 j%16

判据：已知续写的 CE。真实语义下 CE 必须明显更低；错误的配对会让 46/48 值头
拿到别的键头的 q/k，输出近乎噪声。
"""

import torch

from loader_qwen4 import load_model
import transformers.models.qwen4_exp.modeling_qwen4_exp as M

GGUF = [
    "/mnt/d/Admin/Documents/GitHub/Strata-data/models/swift-IQ3_XXS/"
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    "/mnt/d/Admin/Documents/GitHub/Strata-data/models/swift-IQ3_XXS/"
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
]
PACK = "/root/experts_nf4.bin"
TOK_DIR = "/root/models/qwen36-35b-a3b"
HK, HV = 16, 48

model, _ = load_model(GGUF, PACK)
model.eval()
model.config.use_cache = False
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(TOK_DIR)
body, head = (getattr(model, "model", None) or model), model.lm_head
inner = getattr(body, "language_model", body)

CASES = [
    ("The capital of France is", " Paris"),
    ("1 2 3 4 5 6 7", " 8"),
    ("我是人工智能助手", "，"),
]


@torch.no_grad()
def next_top5(prefix: str):
    ids = tok(prefix, return_tensors="pt").input_ids.cuda()
    h = inner(input_ids=ids, use_cache=False).last_hidden_state[0, -1]
    lp = torch.log_softmax(head(h).float(), dim=-1)
    top = lp.topk(5)
    return [repr(tok.decode([int(i)])) for i in top.indices], [float(v) for v in top.values]


@torch.no_grad()
def ce_of(prefix: str, continuation: str):
    p_ids = tok(prefix, add_special_tokens=False).input_ids
    c_ids = tok(continuation, add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + c_ids], device="cuda")
    h = inner(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
    labels = ids[0, 1:]
    lp_all = torch.log_softmax(head(h[-len(c_ids):]).float(), dim=-1)
    picked = lp_all.gather(-1, labels[-len(c_ids):, None]).squeeze(-1)
    return float(-picked.mean())


def report(tag: str):
    print(f"\n===== {tag} =====", flush=True)
    for prefix, cont in CASES:
        toks, vals = next_top5(prefix)
        print(f"{prefix!r} -> top5: " + ", ".join(f"{t}({v:.2f})" for t, v in zip(toks, vals)), flush=True)
    tot = 0.0
    for prefix, cont in CASES:
        ce = ce_of(prefix, cont)
        tot += ce
        print(f"CE({cont!r} | {prefix!r}) = {ce:.3f}   (uniform=12.42)", flush=True)
    print(f"mean CE = {tot / len(CASES):.3f}", flush=True)


report("interleave (HF, j//3)")

# ---- 切到 tile (j % 16) ----
_orig_chunk = M.torch_chunk_gated_delta_rule
_orig_rec = M.torch_recurrent_gated_delta_rule


def _tile(t):
    idx = torch.arange(HV, device=t.device)
    return t[..., (idx % HK) * (HV // HK), :]


def _patched_chunk(q, k, v, *a, **kw):
    return _orig_chunk(_tile(q), _tile(k), v, *a, **kw)


def _patched_rec(q, k, v, *a, **kw):
    return _orig_rec(_tile(q), _tile(k), v, *a, **kw)


M.torch_chunk_gated_delta_rule = _patched_chunk
M.torch_recurrent_gated_delta_rule = _patched_rec

report("tile (Strata engine, j%16)")
