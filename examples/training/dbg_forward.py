"""前向健康检查：argmax 续写必须是人话，已知续写的 CE 必须远低于均匀分布。

背景（2026-10-29）：v10 pretrain loss 13.7~14.3、MeZO 冒烟身份样本 CE 13.5 —— 两条
独立数据路径都接近甚至差于均匀分布（ln 248320 ≈ 12.4）。引擎侧同一 GGUF 输出正常，
权重本身没问题 ⇒ 怀疑 PyTorch HF 前向路径有 bug。本脚本把假设空间一分为二：
  - argmax 续写是乱码 / CE≈12+ ⇒ 前向坏了（去查 loader/PLE/GDN/路由）；
  - 续写正常但 chat 模板下 CE 高 ⇒ 问题在模板/分词对齐。
不训不扰动，纯 no_grad 前向。
"""

import sys

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

model, _ = load_model(GGUF, PACK)
model.eval()
model.config.use_cache = False
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(TOK_DIR)
body, head = (getattr(model, "model", None) or model), model.lm_head
inner = getattr(body, "language_model", body)

print(f"vocab: tokenizer {len(tok)}  head out {head.out_features}", flush=True)

CASES = [
    # (前缀, 期望续写)。不用 chat 模板，纯文本续写，排除模板因素。
    ("The capital of France is", " Paris"),
    ("1 2 3 4 5 6 7", " 8"),
    ("我是人工智能助手", "，"),
]


@torch.no_grad()
def next_top5(prefix: str):
    ids = tok(prefix, return_tensors="pt").input_ids.cuda()
    h = inner(input_ids=ids, use_cache=False).last_hidden_state[0, -1]
    logits = head(h).float()
    lp = torch.log_softmax(logits, dim=-1)
    top = lp.topk(5)
    toks = [repr(tok.decode([int(i)])) for i in top.indices]
    ce_known = None
    return toks, [float(v) for v in top.values], ce_known


@torch.no_grad()
def ce_of(prefix: str, continuation: str):
    p_ids = tok(prefix, add_special_tokens=False).input_ids
    c_ids = tok(continuation, add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + c_ids], device="cuda")
    h = inner(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
    labels = ids[0, 1:]
    lp_all = torch.log_softmax(head(h[-len(c_ids):]).float(), dim=-1)
    picked = lp_all.gather(-1, labels[-len(c_ids):, None]).squeeze(-1)
    return float(-picked.mean()), [tok.decode([int(t)]) for t in c_ids]


for prefix in CASES:
    toks, vals, _ = next_top5(prefix[0])
    print(f"{prefix[0]!r} -> top5: "
          + ", ".join(f"{t}({v:.2f})" for t, v in zip(toks, vals)), flush=True)

for prefix, cont in CASES:
    ce, toks = ce_of(prefix, cont)
    print(f"CE({cont!r} | {prefix!r}) = {ce:.3f} per token  (uniform=12.42)  "
          f"tokens={toks}", flush=True)
