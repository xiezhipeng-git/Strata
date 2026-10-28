# 训练路线对比：反传 sft.py vs MeZO sft_mezo.py（2026-10-29 实测）

模型：Qwen3.8-Flash-Next 125B / IQ3_XXS，4090 48GB + 64GB 内存，LoRA r=16 α=32、
12 个挂载点（744 张量，可训练 1.15e8 量级）、内置身份数据（10 条）、每步 1 条样本、
20 步。三条路线各跑独立进程，外部采样 RSS（进程树）与 nvidia-smi。

复现：
```
python3 bench_cmp.py --gguf <shards...> --pack /root/experts_nf4.bin \
    --tokenizer /root/models/qwen36-35b-a3b --steps 20 --mezo-lr 1e-5
```

## 结果

| | 推理基线 (no_grad) | 反传 sft.py | MeZO sft_mezo.py |
|---|---|---|---|
| s / step | 3.1 | 5.4 | 8.7 |
| tok/s（loss token） | 2.6 | 1.5 | 1.3 |
| loss 首→末 | — | 4.57 → **0.42** | 4.71 → 2.84 |
| 探针 Δ logp('Strata'\|'你是谁') | — | **+19.60 nats** | −0.43 nats |
| 显存峰值（训练段） | 25.6 GiB | 35.2 GiB | **25.8 GiB** |
| 显存峰值（加载段） | 37.4 GiB | 37.1 GiB | 37.3 GiB |
| RSS（训练段） | 45.7 GiB | 46.2 GiB | 45.8 GiB |
| RSS（加载段，mmap 峰值） | 53.5 GiB | 53.4 GiB | 53.4 GiB |

口径：显存是 `torch.cuda.max_memory_allocated()`，两个训练脚本都在训练循环前
`reset_peak_memory_stats()`，所以"训练段"不含加载期临时缓冲；RSS 是进程树常驻内存，
加载段包含 GGUF mmap + 反量化的 page cache 峰值。

## 结论

1. **显存：MeZO 的核心卖点成立**。训练段 25.8 GiB ≈ 纯推理 25.6 GiB（+0.2），
   反传 35.2 GiB（比推理 +9.6）。MeZO 省 **9.4 GiB** —— 正好是"激活 + 梯度 +
   优化器状态"那一块。
2. **速度：MeZO 反而慢 1.6×**（8.7 vs 5.4 s/step）。原因拆开算：一次前向 3.1s，
   反传只花 2.3s（0.74× 前向，因为 LoRA + fused NF4 kernel 要保留的激活很少），
   而 MeZO 每步要 2 次完整前向 6.2s + 扰动/更新约 2.5s。**在这个配置下"反传比
   第二次前向便宜"，MeZO 的速度优势不成立**。
3. **收敛：20 步内 MeZO 推不动**。反传 loss 4.57→0.42、探针 +19.6 nats；
   MeZO 在 lr=1e-6（Δ +0.82）、1e-5（Δ −0.43）都基本不动，lr=1e-3 直接发散
   （loss 4.5→11.7，Δ +3.74 但分布已坏）。proj_grad 在 ±100 间跳，方向信噪比
   ~1/√d 的体现 —— 论文的量级（上千步）才够。
4. **CPU 内存三条路一样**（训练段 ~46 GiB），瓶颈是 GGUF 常驻 + NF4 pack，
   与优化器无关。加载段 53.4 GiB 是全程最高点（mmap 反量化）。

## 选型建议

- 追求"训得动、步数少" → 反传。20 步就有 19.6 nats 的行为改变。
- 追求"显存贴着推理走"（比如想在同一个进程里一边服务一边训）→ MeZO，但要接受
  1.6× 的时间代价和明显更多的步数。
- MeZO 若要用，下一步应先解决两件事：① 扰动/更新的 2.5s 开销（z 现在每步按
  744 个张量重生成，可以缓存或用更粗的块扰动）；② lr 的自适应（proj_grad 量级
  在 1e0~1e2 跳，固定 lr 必然要么不动要么发散）。
