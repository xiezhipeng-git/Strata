"""加载正确性的门禁：引擎 logprobs vs PyTorch(GGUF 源) 逐 token 对照。

引擎（Windows 侧，125B IQ3_XXS）和 PyTorch 学生读的是同一份量化权重——
反量化是确定性的，所以两边 base 模型的 logprob 应当几乎一致（注意力 kernel
差异除外）。对不上 = 加载/映射/反量化有错，一律修到对上为止才准训练。

用法：
  1. Windows 侧把引擎跑起来（Strata CN 快捷方式，端口见 strata-swift-iq3_xxs.json）；
  2. python verify_q4.py --engine http://127.0.0.1:8080 \
       --gguf <shard1> <shard2> --pack experts_nf4.bin
"""

from __future__ import annotations

import argparse
import sys

import torch

import strata_client


PROMPTS = ["你是谁？", "Who are you?", "请用一句话介绍你自己。"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://127.0.0.1:8080")
    ap.add_argument("--engine-model", default=None)
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--pack", default="experts_nf4.bin")
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()

    model_id = args.engine_model
    if model_id is None:
        import urllib.request
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(args.engine.rstrip("/") + "/v1/models", timeout=30) as r:
            model_id = str(json.loads(r.read())["data"][0]["id"])

    print("collecting engine rollouts ...", flush=True)
    rollouts = []
    for q in PROMPTS:
        out = strata_client.chat(args.engine, [{"role": "user", "content": q}], model=model_id,
                                 max_tokens=args.max_new, temperature=1.0, seed=42)
        assert out["tokens_exact"], "engine did not return exact token ids/logprobs"
        rollouts.append(out)
        print(f"  {q!r}: {len(out['token_ids'])} tokens", flush=True)

    print("loading student from GGUF ...", flush=True)
    import json  # noqa: F401
    from loader_qwen4 import load_model

    model, meta = load_model(args.gguf, args.pack)
    model.eval()
    tok_prompt_ids = [r["prompt_ids"] for r in rollouts]

    worst, agree = 0.0, 0
    total = 0
    with torch.no_grad():
        for r in rollouts:
            ids = torch.tensor([r["prompt_ids"] + r["token_ids"]], device="cuda")
            # 逐位置 teacher forcing：引擎的 logprob 是它自己采样时对选中 token 的值
            for step in range(len(r["token_ids"])):
                ctx = ids[:, : len(r["prompt_ids"]) + step]
                logits = model(input_ids=ctx).logits[0, -1].float()
                lp = torch.log_softmax(logits, -1)
                tok_id = r["token_ids"][step]
                mine = float(lp[tok_id])
                eng = r["token_logprobs"][step]
                worst = max(worst, abs(mine - eng))
                agree += int(mine >= lp.max() - 1e-9 and eng >= 0 or True)  # top-1 一致性单独算
                total += 1
                print(f"  tok {step}: engine {eng:.4f}  torch {mine:.4f}  diff {mine - eng:+.4f}")
    print(f"\nmax |Δlogprob| = {worst:.4f} over {total} tokens")
    gate = 0.05
    print("PASS" if worst < gate else f"FAIL (> {gate} nats)", flush=True)
    return 0 if worst < gate else 1


if __name__ == "__main__":
    sys.exit(main())
