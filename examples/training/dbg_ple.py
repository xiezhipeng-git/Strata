"""Debug PLE shim forward step by step."""
import torch

from loader_qwen4 import load_model

GGUF = [
    "/mnt/d/Admin/Documents/GitHub/Strata-data/models/swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    "/mnt/d/Admin/Documents/GitHub/Strata-data/models/swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
]
model, _ = load_model(GGUF, "/root/experts_nf4.bin")
model = model.cuda().eval()
ple = model.model.layers[1].ple.ple_embedding
print("attrs:", ple.ngram_size, ple.heads_per_ngram, ple.context_len, ple.eos_token_id,
      "layer_idx", ple.layer_idx, flush=True)
print("multipliers:", ple.layer_multipliers.tolist(), flush=True)

ids = torch.tensor([[151644, 9707, 91, 8308, 11]], dtype=torch.long, device="cuda")
with torch.no_grad():
    history = torch.cat([ids.new_full((1, ple.context_len), ple.eos_token_id), ids], dim=-1)
    shifted = [ple._shift_right_ignore_eos(history, s) for s in range(ple.ngram_size)]
    for s, t in enumerate(shifted):
        print(f"shifted[{s}]: dtype {t.dtype} min {t.min().item()} max {t.max().item()}", flush=True)
    m0 = shifted[0] * ple.layer_multipliers[0]
    print("t0*m0:", m0.dtype, m0.min().item(), m0.max().item(), flush=True)
    mixed = torch.bitwise_xor(m0, shifted[1] * ple.layer_multipliers[1])
    print("mixed n2:", mixed.dtype, mixed.min().item(), mixed.max().item(), flush=True)
    hvs = ple.ngram_heads_vocab_sizes[0:8]
    offs = ple.ngram_heads_offsets[0:8]
    print("hvs:", hvs.tolist(), flush=True)
    print("offs:", offs.tolist(), flush=True)
    nid = torch.remainder(mixed.unsqueeze(-1), hvs.view(1, 1, -1))
    print("after remainder:", nid.dtype, nid.min().item(), nid.max().item(), flush=True)
    nid = nid + offs.view(1, 1, -1)
    print("final ngram ids:", nid.min().item(), nid.max().item(), flush=True)
