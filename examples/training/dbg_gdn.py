"""最小复现：fla chunk_gated_delta_rule 在长 seq 下炸 "device not ready"。

从短到长扫 T，定位失效边界 + 判断是不是 T 非整块（64）倍数引起。
"""

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

dev = "cuda"
B, H, K, V = 1, 8, 128, 128

for T in (64, 128, 256, 512, 1024, 2047, 2048, 4096):
    try:
        q = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16)
        k = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16)
        v = torch.randn(B, T, H, V, device=dev, dtype=torch.bfloat16)
        g = torch.randn(B, T, H, device=dev, dtype=torch.float32).sigmoid().log()
        beta = torch.rand(B, T, H, device=dev, dtype=torch.bfloat16)
        o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta,
                                      output_final_state=False,
                                      use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
        print(f"T={T:5d}  OK  out {tuple(o.shape)}", flush=True)
        del q, k, v, g, beta, o
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"T={T:5d}  FAIL  {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()
