"""判别实验：PT 侧把 GDN 的 key-head 映射从 repeat_interleave(j//3) 换成引擎的 tile(j%16)，
看与引擎 oracle logits 的 pearson 是否显著升高。

- 若 tile 下 pearson 明显 > interleave 下 → 引擎 tile 语义才是对的（PT/loader 需适配，引擎不动）
- 若 tile 下 pearson 更低 → 引擎 `% HK` 是真 bug（改引擎）
"""

import sys

import numpy as np
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
IDS = [760, 6511, 314, 9338, 369]
HK, HV = 16, 48

TILE = "--tile" in sys.argv

model, _ = load_model(GGUF, PACK)
model.eval()

if TILE:
    import transformers.models.qwen4_exp.modeling_qwen4_exp as M

    orig = M.torch_chunk_gated_delta_rule

    def tile_qk(t):
        # t: (B, T, HV, D)，interleave 约定 t[..., j, :] = key head j//3
        idx = torch.arange(HV, device=t.device)
        src = (idx % HK) * (HV // HK)   # key head h 在 interleave 下的代表行 3h
        return t[..., src, :]

    def patched(q, k, v, *a, **kw):
        return orig(tile_qk(q), tile_qk(k), v, *a, **kw)

    M.torch_chunk_gated_delta_rule = patched
    print("MODE: tile (engine semantics)", flush=True)
else:
    print("MODE: interleave (HF semantics)", flush=True)

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(TOK_DIR)
ids = torch.tensor([IDS], device="cuda")
with torch.no_grad():
    logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()

oracle = torch.from_numpy(np.fromfile("/mnt/d/Admin/Documents/GitHub/Strata/oracle_full.bin", dtype=np.float32))
pt = logits.cpu()


def topk(v, k=5):
    t = v.topk(k)
    return [(int(i), round(float(x), 2)) for x, i in zip(t.values, t.indices)]


print("oracle top5:", topk(oracle), flush=True)
print("pytorc top5:", topk(pt), flush=True)
c = float(torch.corrcoef(torch.stack([pt, oracle]))[0, 1])
print(f"pearson corr(pt, oracle) = {c:.4f}", flush=True)
