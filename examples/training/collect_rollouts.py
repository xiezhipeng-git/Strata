"""从 Strata 引擎采 GRPO 轨迹，写成训练侧能直接吃的 jsonl。

为什么单独一个脚本：48GB 卡放不下引擎（Q4_K_XL 占 28.6GB）和 4bit 学生（21.8GB），
所以采集与训练必须分两段，而采集这段只需要 HTTP，不需要 torch。
（`train_grpo_own.py --no-student` 做同一件事，但它 import torch，只能在 WSL 里跑。）

每条轨迹记的是引擎自己吐的序列：`token_ids` 与 `token_logprobs` 都来自 LP 行，
所以 `tokens_exact` 为真时可以当作严格 on-policy 的旧策略概率用；为假（老引擎不产 LP）
就只剩文本，训练端会退回到本地重编码 + 本地重算 old logprob。

用法：
  python collect_rollouts.py --engine http://127.0.0.1:8095 --prompts prompts_variance.jsonl \
      --group 4 --max-new 40 --temperature 1.2 --out rollouts35.jsonl
"""

from __future__ import annotations

import argparse
import json
import time

import strata_client


def prompt_target(prompt: str) -> str:
    caps = [w.strip(".,:;!?\"'") for w in prompt.split()]
    caps = [w for w in caps if w.isalpha() and w.isupper()]
    return caps[-1].lower() if caps else ""


def reward_of(traj: dict) -> float:
    """kind=exact（默认）严格相等；kind=words:N 要求词数恰好为 N；kind=wordsclose:N 按词数差给分。

    wordsclose:N = max(0, 1 - |n-N|/N)：只要长度飘就有梯度。words:N 的问题不是"太严"，
    而是**组内四条常常全对或全错**（std=0 ⇒ 优势全 0 ⇒ grad_norm=0），看起来训练跑完了，
    其实一个参数都没动。
    """
    text = (traj.get("content") or "").strip()
    kind = traj.get("kind") or "exact"
    if kind.startswith("wordsclose:"):
        want = int(kind.split(":", 1)[1])
        return max(0.0, 1.0 - abs(len(text.split()) - want) / max(1, want))
    if kind.startswith("words:"):
        return 1.0 if len(text.split()) == int(kind.split(":", 1)[1]) else 0.0
    want = traj.get("target") or prompt_target(traj.get("prompt", ""))
    flat = text.strip().strip("。.!?！？\"'`*_ \n\t").lower()
    return 1.0 if want and flat == want else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://127.0.0.1:8095")
    ap.add_argument("--engine-model", default=None)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--think", action="store_true", help="默认关思考：max_tokens 同时覆盖思考与回答")
    args = ap.parse_args()

    model = args.engine_model
    if model is None:
        import urllib.request

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(args.engine.rstrip("/") + "/v1/models", timeout=30) as r:
            model = json.loads(r.read())["data"][0]["id"]
    print(f"engine model: {model}", flush=True)

    tasks = [json.loads(line) for line in open(args.prompts, encoding="utf-8") if line.strip()]
    trajs = []
    t0 = time.time()
    for task in tasks:
        for _ in range(args.group):
            try:
                out = strata_client.chat(
                    args.engine, [{"role": "user", "content": task["prompt"]}], model=model,
                    max_tokens=args.max_new, temperature=args.temperature,
                    enable_thinking=(True if args.think else False))
            except Exception as exc:   # a template that has no enable_thinking: ask the default way
                print(f"  (enable_thinking refused: {type(exc).__name__}), retry without", flush=True)
                out = strata_client.chat(
                    args.engine, [{"role": "user", "content": task["prompt"]}], model=model,
                    max_tokens=args.max_new, temperature=args.temperature)
            traj = {"prompt": task["prompt"], "target": task.get("target", ""),
                    "kind": task.get("kind", "exact"), **out}
            traj["reward"] = reward_of(traj)
            trajs.append(traj)
        done = [t for t in trajs if t["prompt"] == task["prompt"]]
        print(f"  {task['prompt'][:44]!r} x{len(done)}  reward {sum(t['reward'] for t in done):.0f}/{len(done)}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    n = len(trajs)
    exact = sum(1 for t in trajs if t.get("tokens_exact"))
    rewards = [t["reward"] for t in trajs]
    mean = sum(rewards) / n if n else 0.0
    std = (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5 if n else 0.0
    print(f"saved {n} trajectories to {args.out}\n"
          f"  tokens_exact {exact}/{n}   reward mean {mean:.3f} std {std:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
