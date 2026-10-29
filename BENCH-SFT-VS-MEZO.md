# 训练路线对比：反传 sft.py vs MeZO sft_mezo.py

模型：Qwen3.8-Flash-Next 125B / IQ3_XXS（专家预打包成 NF4 pack，59.8 GiB / 24576 块 /
2.49 MiB 每块），单卡 4090 48GB + 宿主 64GB，LoRA r=16 α=32、12 类挂载点、
每步 1 条样本。三个变体各跑独立进程，外部采样进程树 RSS 与 nvidia-smi。

> 2026-10-29 修订：旧版只有短序列一档，且当时的 loader 与数据顺序都没有固定。
> 现在给了两档序列长度、同 seed 同批块，并把「92.63 GiB 从哪来」单列一节。

## 0. 先钉口径：三个 tok/s 不是一回事

这几个数经常被混着比，先拆开（同一台机器、同一张 4090）。

**更正**：早先这里写过「192 tok/s 是 35B-A3B、160 是另一个模型」，那条是错的。
`strata-main-lp.log`（125B，888 个请求）的 decode 中位数就是 **172.8 tok/s**、p90 194.5、
max 243 —— 125B 自己就跑到 160~190。35B 的 `strata-qwen36-lora.log` 是 192.4，两个模型
在 decode 上几乎同值，**所以 decode tok/s 根本不能用来判别模型**。

| 数字 | 谁打的 | 模型 | 阶段 |
|---|---|---|---|
| **172.8 tok/s**（p50，888 请求；用户说的 160） | `strata-main-lp.log` | **125B IQ3_XXS** | **decode** |
| **982 tok/s** | `build-lp\strata.exe` 今天实测 2047 token | 同一个 125B pack | prefill |
| **62~70 tok/s** | `bench_infer.py`（WSL2） | 同一个 125B pack | prefill |
| **2.0 tok/s**（506.9 ms/token） | `bench_decode.py`（WSL2） | 同一个 125B pack | **decode** |

同一模型、同一份 pack，两档各自比：

- **prefill**：982 → 62~70，差 **14~16×**（训练是 prefill 形状，这是训练该看的数）
- **decode**：172.8 → 2.0，差 **86×**

decode 那档差得更多，原因是结构而不是 LoRA：引擎 decode 走 `--expert-cache` 常驻 + MTP
推测解码（`drafts accepted 131 of 181`，一次前向出约 1.7 个 token），单 token 基本不搬专家；
训练路径每个 token 都要把该层未常驻的专家从 host 搬一遍（本例 VRAM 常驻 6579/24576 块，
约 73% 要现搬）。这个差距的来源是实现而不是平台（见 §6）。

### 配置口径（顺带查清：当前快捷方式已不在 125B 上）

`launcher/start_server.py:241` 的 `stock_config()` 按 **mtime 最新**选 `strata-*.json`。
各配置 mtime：

| 配置 | mtime | 指向 |
|---|---|---|
| `strata-qwen36-devlora.json` | **10-28 11:21（最新）** | `build-q6k` + **Qwen3.6-35B-A3B-UD-Q4_K_XL** + LoRA |
| `strata-qwen36-lora0.json` / `-lora.json` / `-train.json` | 10-28 07:14~10:17 | 同上 35B |
| `strata-swift-iq3_xxs.json` | 10-20 01:25 | `build-lp` + **125B IQ3_XXS** |
| `strata-main-lp.json` | 10-20 01:23 | `build-lp` + **125B IQ3_XXS** |

所以 10-28 做 dev-lora 那批实验时写下的 35B 配置把 mtime 顶到了最前，**现在双击
`Strata CN.lnk` 起的是 35B，不是 125B**。125B 的两份配置内容自 10-20 起没变
（`strata-main-lp.log` 的启动命令行与 `strata-main-lp.json` 逐参数一致）。
改回来只需让 125B 那份成为最新：`touch strata-main-lp.json`（配置被 `.gitignore:37`
的 `/strata-*.json` 忽略，git 查不到历史，只能靠 mtime/内容判断）。

## 1. 长序列 T=2048（真实训练形状）

`/root/pretrain_blocks.npy`：22269 块 × 2048 token（uint32），每步 1 块，`--seed 0`
固定洗牌，四个变体走**完全相同的块序列**。

| 变体 | s/step（中位 / best） | tok/s | 显存·训练段 | **额外显存** | RSS·训练段 | 额外 RSS | 探针 Δ |
|---|---|---|---|---|---|---|---|
| 推理基线（no_grad） | 45.9 / 31.4 | 44.6 | 26.30 GiB | — | 49.09 GiB | — | — |
| **反传 + ckpt** | **84.3 / 75.1** | 24.3 | 30.20 GiB | **+3.90** | 50.01 GiB | +0.92 | +0.74 |
| **MeZO**（queries=1） | **62.8 / 52.0** | 32.6 | 26.70 GiB | **+0.40** | 49.14 GiB | +0.05 | −0.49 |
| 反传·不开 ckpt | 跑不动 | — | — | — | — | — | — |

「跑不动」是 OOM，PyTorch 已分配 **92.63 GiB**（48 GiB 卡），原因见 §4，不是物理下限。

配对可验证：两个变体确实是同一批块，同一步的 loss 一致 ——
`sft 0.2600 / 0.1492 / 1.0169 / 1.4260` ↔ `mezo loss+ 0.2617 / 0.1473 / 1.0075 / 1.4422`。
逐步配对比值 1.19 / 1.44 / 1.29 / 1.34，均值 **1.32×**。

口径：显存是 `torch.cuda.max_memory_allocated()`，各脚本都在训练循环前
`reset_peak_memory_stats()`，所以「训练段」不含加载期临时缓冲。RSS 是进程树常驻集。
只有 4~6 步，专家缓存仍在变热（推理那一档第 1 步 45s、第 6 步已降到 29.4s），
所以绝对值偏保守，**看同一步内的配对比值更可靠**。

## 2. 短序列（内置身份数据，T≈17，12 步）

| 变体 | s/step（中位 / best） | 显存·训练段 | 额外显存 | RSS·训练段 | 探针 Δ |
|---|---|---|---|---|---|
| 推理基线 | 2.6 / 2.4 | 25.80 GiB | — | 45.42 GiB | — |
| **反传·不开 ckpt** | **5.4 / 4.3** | 35.50 GiB | **+9.70** | 46.08 GiB | **+22.65** |
| 反传 + ckpt | 9.1 / 7.9 | 26.40 GiB | +0.60 | 46.07 GiB | +22.45 |
| MeZO | 7.5 / 6.2 | 26.20 GiB | +0.40 | 45.69 GiB | −0.17 |

（`sft` 不开 ckpt 的 5.4 s/step 与 2026-10-29 早先那次完全一致，说明短序列没有回归。）

## 3. 结论：排序随序列长度翻转，开关是 ckpt

| | 短序列（不 ckpt） | 长序列 T=2048 |
|---|---|---|
| 谁快 | **反传 5.4 vs MeZO 7.5（1.39×）** | **MeZO 62.8 vs 反传 84.3（1.34×）** |
| 谁省显存 | MeZO（+0.40 vs +9.70，省 9.3 GiB） | MeZO（+0.40 vs +3.90，省 3.5 GiB） |

原因是一个前向的单价：

- 短序列不需要 ckpt。反传 = 1 前向 + 反传（0.74× 前向，QDoRA/fused NF4 下要留的激活
  很少）≈ 1.74 个前向当量；MeZO = 2 个前向 + 小量扰动/更新。**反传更便宜**，于是赢。
- 长序列必须 ckpt（否则 OOM，§4）。ckpt 让反传 = 1 前向（存输入）+ 1 次**重算前向** +
  反传 ≈ 2.7 个前向当量，而 MeZO 还是 2 个。**MeZO 更便宜**，于是反超。

所以「MeZO 一定慢」和「反传一定快」都只在特定序列长度下成立。MeZO 的显存优势
（贴住推理基线，+0.4 GiB）在两档都成立。

## 4. 92.63 GiB 从哪来 —— 这是设计缺陷，不是物理下限

`nf4_fused.py:122`：

```python
class FusedNF4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, absmax, code):
        ctx.save_for_backward(packed, absmax, code)      # ← 这里
```

在真正的 QLoRA 里这行几乎免费：量化权重是**常驻参数**，`save_for_backward` 只多存一个
引用。但在这里，专家块是从 pack 流式搬进一块**会被轮转复用的 staging 缓冲**里的；一旦
进了计算图，PyTorch 就必须让那一块一直活着。T=2048 时几乎每个专家都被命中一次：

```
24576 块 × 2.49 MiB = 59.8 GiB   ← 正好是整个专家 pack
实测增量 92.63 − 26.30 = 66.3 GiB  ← pack + 激活
```

**结论：反传图里把整套流式专家权重装了进去。** 这与「比 QLoRA 更省」的设计目标相反 ——
QLoRA 省显存靠的是「权重只以量化形式常驻，反传时现场 dequant」，而现在这个实现是
「权重搬一次就留一份」。

正确做法（下一步要做的）：**反传不要权重，也不要激活全存**。专家权重是冻结的、可按
(layer, expert) 寻址重取的，所以

1. 前向只在 ctx 里存 **专家 id + 路由**（几 KB），不存 packed/absmax；
2. 反传按 id **重新流取**该专家的块，跑 `_nf4_bwd_kernel` 出 dx。

这样权重永不进图，显存只剩激活；同时**不再需要 ckpt**（省掉那次重算前向）——
按上表的口径，长序列反传会从 84.3 s 降到 ~1.74 个前向当量，即**反超 MeZO**。

顺带说优化器：LoRA 可训练参数约 1.15e8，AdamW fp32 的 m+v ≈ **1.4 GB**，占那 66 GiB 的
约 2%。换成 8-bit 状态能省 ~1 GB —— 不是这次爆显存的原因，但等 66 GiB 消掉之后它就
占到可观比例了，届时值得一起做。

## 5. 本轮修掉的两个测量缺陷

1. **整块 H2D 与反传别名冲突（回归）**。`loader_qwen4.py` 的「整块一次 `copy_` + 显存侧
   四段 view」返回的是共享 staging 缓冲上的视图，slot 会轮转覆盖。纯前向无所谓，但反传时
   fused 函数刚把 packed/absmax 存进 ctx，缓冲就被下一组覆盖 →
   `variables needed for gradient computation has been modified by an inplace operation
   ... [ByteTensor [2560, 320]] is at version N`。旧的 per-expert `.to(dev)` 每块都是独立
   新张量，所以从没踩过。现在该快路径只在 `not torch.is_grad_enabled()` 下启用。
2. **数据顺序没固定**。`sft.py` 用的是无 seed 的 `randperm`，不同 run 命中的专家数不同
   （同配置 loss 在 0.06~1.44 之间跳），s/step 的差异里混着大量数据噪声。现在
   `sft.py` / `sft_mezo.py` / `bench_infer.py` / `bench_cmp.py` 统一 `--seed`，
   四条变体走同一批块。

## 6. 平台：WSL 是不是拖慢了？（结论：不是主因）

- 传输：WSL 里 43.8 GiB `pin_memory` 直接 `AcceleratorError`，退回 pageable；微基准
  pageable 整块直连 **~20 GB/s**，引擎在 Windows 上的 probe 是 **24.1 GB/s** —— 差 1.2×。
- 计算：同一张卡、同一套驱动（通过 WSL 直通），kernel 层面没有平台差异。而计算项本身
  就占大头（拟合出 7.38 ms/token vs 引擎约 1.0 ms/token）。
- ⇒ 从 Windows 能拿回的最多是传输那 1.2×，而当前差距是 14~16×。**主因是实现了
  dequant→bf16→GEMM（每专家只分到 ~80 行，小 GEMM 跑不满 GPU），引擎是量化内核直接算。**

平台真正值得统一的是**内存预算**，不是速度。`.wslconfig` 里已经写死：

```
memory=56GB
# 代价：WSL 顶到 56GB 时 Windows 只剩 ~7GB
#       —— 桌面推理引擎（需 60GB）不能与训练同跑。
```

宿主只有 63.7 GB，而「WSL 训练 56 GB」与「Windows 引擎 60 GB」互斥。搬到 Windows 原生
能省掉 WSL 这一层，但代价是：Windows 侧 `.venv` 目前**没有 torch**（只有 numpy），要新装
cu13 wheel；`sft.py` 里 `resource.getrusage` / `/proc/self/statm` 是 Linux 专用，得改；
而且 44 GiB 的 pinned 内存在 Windows 上同样很可能失败。

## 7. 复现

```bash
# 长序列（同 seed 同批块）
python3 bench_cmp.py --gguf <shards...> --pack /root/experts_nf4.bin \
    --tokenizer /root/models/qwen36-35b-a3b \
    --pretrain-data /root/pretrain_blocks.npy --seqlen 2048 \
    --steps 4 --seed 0 --out /root/cmp_seed.json

# 短序列
python3 bench_cmp.py --gguf <shards...> --pack /root/experts_nf4.bin \
    --tokenizer /root/models/qwen36-35b-a3b --steps 12 --seed 0 --out /root/cmp_short.json
```

`--only infer,sft,sft+ckpt,mezo` 可只跑其中几个。
