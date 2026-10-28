"""MeZO 训练器（feat/zo-mezo 分支）：只靠前向的零阶优化，与反传训练器（sft.py）并存对比。

为什么单独开分支 —— 两条路线的分工（2026-10-29 定）：
  sft.py       PyTorch autograd 反传：每步 1 前向 + 1 反传（+ckpt 重算），梯度精确，
               但需要保留计算图/重算激活，T=2048 全命中时 --ckpt 不可避免。
  sft_mezo.py  MeZO (Malladi et al., NeurIPS 2023, princeton-nlp/MeZO)：每步 2 前向、
               0 反传。全程 no_grad —— 内存占用=推理占用，无 ckpt、无计算图、无
               packed 字节滞留。这正是"从推理侧 logprobs 直接优化 LoRA"的严格形态。

算法（对称差分 + 种子再生）：
    pj = (L(θ+εz) − L(θ−εz)) / (2ε)          一次 z 采样的方向导数估计
    m  = momentum·m + mean_q(pj_q·z_q)        z 由 seed 重生成，不存储
    θ  ← θ − lr · m
    前向顺序：扰动 +εz → 前向 → 扰动 −2εz（即 −εz）→ 前向 → 扰动 +εz 复原 → 更新。

小规模验证（test_zo.py，全绿）：估计器与 autograd 真梯度满足 cos(ĝ,∇L)=|cos(z,∇L)|
（二次损失恒等式，逐位成立）；K 平均收敛 cos≈√(k/(d+k))；ZO-SGD 在保守参数下
loss 1.61→1.10。重要实测教训（写进默认值）：
  - 零阶噪声比信号大 ~√d 倍，动量会放大噪声：lr 1e-2 或 lr 1e-3+momentum0.9 在
    d=32768 玩具问题上发散 ⇒ 默认 momentum=0，lr 从 1e-6 起。
  - bf16 参数吃不动 1e-6 量级更新（bf16 相对精度 ~0.4%）⇒ LoRA A/B 保存在 fp32。

与引擎的关系（对接缝，本文件不实现）：优化器核心只需要一个"给定点集上的目标
token logprob"的前向。将来引擎加一个"运行时叠加 LoRA delta（ε·z 由 seed 现算，
只叠加命中专家/线性层）+ 返回目标 logprob"的接口，本文件的循环体原样换成
engine forward，内存/显存/调度全部交还引擎 —— 那才是这条路线的完全体。

用法示例：
  python3 sft_mezo.py --gguf <shards...> --pack /root/experts_nf4.bin \
      --tokenizer ~/models/qwen36-35b-a3b --steps 20 --lr 1e-6 \
      --targets o_proj,out_proj --out /tmp/lora_mezo.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import types

import torch

from ce import token_logprobs
from lora import DEFAULT_TARGETS, LoRALinear, freeze_all, inject_lora, trainable_summary
from sft import builtin_data, encode, name_logprob, text_model


def zo_cast_fp32(model):
    """LoRA A/B 转 fp32 并给模块绑 dtype 安全的 forward。

    两个原因：
      1) ZO 更新步长 lr~1e-6，bf16 相对精度 ~2^-8，小幅更新会整体舍入丢失；
      2) fp32 A/B × bf16 x 的 F.linear 会 dtype 报错，需显式换边。
    扰动 z 也用 fp32 生成（ε=1e-3 在 bf16 下只有 ~3 位有效数字）。
    """
    n = 0

    def fwd(self, x):
        out = self.base(x)
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(x.to(self.A.dtype), self.A), self.B)
        return out + delta.to(out.dtype) * self.scale

    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.A.data = m.A.data.float()
            m.B.data = m.B.data.float()
            m.forward = types.MethodType(fwd, m)
            n += 1
    return n


def perturb(params, seed, scaling, eps):
    """θ ±= scaling·ε·z，z 逐参数由 seed 重生成（与 MeZO 官方实现同构）。"""
    for p in params:
        torch.manual_seed(seed)
        z = torch.normal(0, 1, size=p.shape, device=p.device, dtype=p.dtype)
        p.data.add_(z, alpha=scaling * eps)


@torch.no_grad()
def sample_loss(body, head, ids, plen, reduction: str) -> float:
    """目标 token 的损失（eval 态、no_grad、不落任何计算图）。"""
    h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
    labels = ids[0, 1:]
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[plen - 1:] = True
    lp = token_logprobs(h, head, labels, chunk=128)
    picked = lp[mask]
    return float(-picked.sum() if reduction == "sum" else -picked.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="HF 权重目录（不用 GGUF 时）")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--gguf", nargs="+", default=[], help="GGUF 分片；专家需 NF4 pack")
    ap.add_argument("--pack", default="")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-6,
                    help="ZO-SGD 步长。MeZO 论文 30B 级别用 1e-6~1e-5")
    ap.add_argument("--eps", type=float, default=1e-3, help="扰动半径 ε（论文默认 1e-3）")
    ap.add_argument("--momentum", type=float, default=0.0,
                    help="默认 0：零阶噪声 ~√d 倍于信号，动量放大噪声（test_zo 实测发散区）")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="对动量缓冲的解耦衰减（MeZO 同款；momentum=0 时无意义）")
    ap.add_argument("--queries", type=int, default=1,
                    help="每步独立 z 采样次数 K，平均降方差，代价线性（每 K 多 2 次前向）")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", default="o_proj,out_proj",
                    help="默认引擎可回灌的挂载点（与 sft.py --targets 语义一致）")
    ap.add_argument("--loss-reduction", choices=("mean", "sum"), default="mean",
                    help="proj_grad 用哪个口径。mean 口径下 lr 可直接沿用论文量级；"
                         "sum 会把梯度放大 ~目标 token 数倍")
    ap.add_argument("--out", default="lora_mezo.pt")
    ap.add_argument("--word", default="Strata")
    ap.add_argument("--data", default="")
    ap.add_argument("--pretrain-data", default="")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--probe", default="你是谁")
    args = ap.parse_args()
    data = ([(r["prompt"], r["answer"])
             for r in (json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip())]
            if args.data else builtin_data(args.word))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    if args.gguf:
        if not args.pack:
            raise SystemExit("--gguf needs --pack")
        from loader_qwen4 import load_model

        model, _meta = load_model(args.gguf, args.pack)
        print(f"loaded from GGUF {time.time() - t0:.0f}s  "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    else:
        if not args.model:
            raise SystemExit("--model (HF dir) or --gguf is required")
        from loader import build_skeleton, materialize, stream_load

        model, experts = build_skeleton(args.model)
        materialize(model, experts, device, torch.bfloat16)
        stream_load(model, experts, args.model, device)
        print(f"loaded {time.time() - t0:.0f}s", flush=True)
    torch.cuda.empty_cache()

    # MeZO 全程 eval + no_grad：没有反向，也就没有 dropout/ckpt 的任何必要。
    # 不要 model.train() —— 那是 sft.py 为 HF checkpoint 分支开的服务。
    model.eval()
    model.config.use_cache = False
    freeze_all(model)
    targets = tuple(t for t in args.targets.split(",") if t)
    n = inject_lora(model, targets=targets, r=args.rank, alpha=args.alpha)
    n_cast = zo_cast_fp32(model)
    s = trainable_summary(model)
    print(f"lora: {n} modules (fp32 cast {n_cast}), trainable {s['trainable']:,} "
          f"({s['pct']:.6f}%)", flush=True)

    body, head = text_model(model), model.lm_head
    lp0 = name_logprob(body, head, tok, args.probe, args.word)
    print(f"before: logp('{args.word}' | {args.probe!r}) = {lp0:.4f} "
          f"(p={math.exp(lp0):.8f})", flush=True)

    if args.pretrain_data:
        import numpy as np

        blocks = np.load(args.pretrain_data)
        assert blocks.shape[1] == args.seqlen
        enc = [(list(map(int, blocks[i])), 1) for i in range(len(blocks))]
        print(f"pretrain data: {len(enc)} blocks × {args.seqlen} tokens", flush=True)
    else:
        enc = [encode(tok, q, a) for q, a in data]   # [(ids, plen), ...]

    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "没有可训练参数"
    print(f"zo dimension d = {sum(p.numel() for p in params):,}（单步估计 cos ~ 1/√d；"
          f"收敛靠 SGD 轨迹平均，步数需求见 test_zo.py 注释）", flush=True)
    exp_avgs = [torch.zeros_like(p) for p in params]
    rng = random.Random(0)          # 数据洗牌专用，与扰动 seed 互不污染

    order, cursor = [], 0
    for step in range(1, args.steps + 1):
        t0 = time.time()
        if cursor >= len(order):
            order = rng.sample(range(len(enc)), len(enc))
            cursor = 0
        full, plen = enc[order[cursor]]
        cursor += 1
        ids = torch.tensor([full], device=device)
        ntoks = len(full) - (plen - 1)          # answer 段长度（plen=1 时整块）

        # K 个独立 z 采样：每个 2 次前向，pj 取平均
        qs, loss_pos, loss_neg = [], 0.0, 0.0
        for _ in range(args.queries):
            seed = rng.randrange(1 << 31)
            perturb(params, seed, +1, args.eps)
            lp_ = sample_loss(body, head, ids, plen, args.loss_reduction)
            perturb(params, seed, -2, args.eps)
            lm_ = sample_loss(body, head, ids, plen, args.loss_reduction)
            perturb(params, seed, +1, args.eps)  # 复原 θ
            qs.append((seed, (lp_ - lm_) / (2 * args.eps)))
            loss_pos += lp_
            loss_neg += lm_
        pj = sum(p for _, p in qs) / args.queries

        # m = momentum·m + mean_q(pj_q·z_q)；θ -= lr·m（z 由 seed 重生成）
        for p, m in zip(params, exp_avgs):
            signal = torch.zeros_like(p)
            for seed, pj_q in qs:
                torch.manual_seed(seed)
                z = torch.normal(0, 1, size=p.shape, device=p.device, dtype=p.dtype)
                signal.add_(z, alpha=pj_q)
            if args.weight_decay > 0:
                m.mul_(args.weight_decay)
            m.mul_(args.momentum).add_(signal, alpha=1.0 / args.queries)
            p.data.add_(m, alpha=-args.lr)

        dt = time.time() - t0
        print(f"step {step}: loss+ {loss_pos / args.queries:.4f}  "
              f"loss- {loss_neg / args.queries:.4f}  proj_grad {pj:.4e}  "
              f"loss_tokens {ntoks}  tok/s {ntoks / dt:.1f}  "
              f"{dt:.1f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
              flush=True)

    lp1 = name_logprob(body, head, tok, args.probe, args.word)
    print(f"after:  logp('{args.word}' | {args.probe!r}) = {lp1:.4f} (p={math.exp(lp1):.8f})   "
          f"Δ={lp1 - lp0:+.4f} nats", flush=True)

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
          if k.endswith(".A") or k.endswith(".B")}
    torch.save(sd, args.out)
    nz = sum(1 for v in sd.values() if float(v.abs().max()) > 0)
    print(f"saved {args.out}: {len(sd)} tensors, non-zero {nz}, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
