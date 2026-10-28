"""125B load smoke: build skeleton, stream weights from GGUF/NF4 pack, one forward."""
import sys
import torch

from loader_qwen4 import load_model

GGUF = [
    "/mnt/d/Admin/Documents/GitHub/Strata-data/models/swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    "/mnt/d/Admin/Documents/GitHub/Strata-data/models/swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
]
PACK = "/root/experts_nf4.bin"

model, tok = load_model(GGUF, PACK)
model = model.cuda().eval()
torch.cuda.empty_cache()
print(f"loaded; cuda allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
      f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)

ids = torch.tensor([[151644, 9707, 91, 8308, 11, 1049, 91, 4890, 151645]],
                   dtype=torch.long, device="cuda")   # 任意小 id 序列,仅查前向健康
with torch.no_grad():
    out = model(ids)
logits = out.logits[0, -1]
top5 = logits.topk(5)
print("logits:", logits.shape, f"mean {logits.mean():.3f} std {logits.std():.3f} "
      f"absmax {logits.abs().max():.1f}", flush=True)
print("top5 ids:", top5.indices.tolist(), "values:", [round(float(v), 2) for v in top5.values], flush=True)
print("nan:", bool(torch.isnan(logits).any()), flush=True)
