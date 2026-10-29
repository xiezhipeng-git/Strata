# 训练路线对比：反传 sft.py vs MeZO sft_mezo.py

模型：Qwen3.8-Flash-Next 125B / IQ3_XXS（专家预打包成 NF4 pack，59.8 GiB / 24576 块 /
2.49 MiB 每块），单卡 4090 48GB + 宿主 64GB，LoRA r=16 α=32、12 类挂载点、
每步 1 条样本。三个变体各跑独立进程，外部采样进程树 RSS 与 nvidia-smi。

> 2026-10-29 修订：旧版只有短序列一档，且当时的 loader 与数据顺序都没有固定。
> 现在给了两档序列长度、同 seed 同批块，并把「92.63 GiB 从哪来」单列一节。
>
> **2026-10-29 第二次修订（commit `b9af996a` + `45ce80c9`）：前向 17.9 → 9.1 s/步、反传 84.3 → 24.3 s/步。**
> 瓶颈不在 MeZO 与反传的选择上，而是 transformers 的 QSA 稀疏索引器逐 query 跑 Python
> 循环，每步制造 2.4e4 次 DtoH 同步（§7）。§1 的整张表已替换、§7 重写；旧数值保留在
> 下方「上一版」的引用里以便对照。

> **2026-10-29 第三次修订（收尾）：§1 那张表是 6 步口径，被「首轮变慢」拖累了。**
> 14 步实测：前 7 步每步比稳态慢 4~14 s，第 8 步之后稳定在 **18~20 s/步 = 103~120 tok/s**。
> 一次性成本约 40 s，真跑会摊掉。成因未定位，但已排除四项（§7.7）。

## 0. 先钉口径：三个 tok/s 不是一回事

这几个数经常被混着比，先拆开（同一台机器、同一张 4090）。

**更正**：早先这里写过「192 tok/s 是 35B-A3B、160 是另一个模型」，那条是错的。
`strata-main-lp.log`（125B，888 个请求）的 decode 中位数就是 **172.8 tok/s**、p90 194.5、
max 243 —— 125B 自己就跑到 160~190。35B 的 `strata-qwen36-lora.log` 是 192.4，两个模型
在 decode 上几乎同值，**所以 decode tok/s 根本不能用来判别模型**。

| 数字 | 谁打的 | 模型 | 阶段 |
|---|---|---|---|
| **172.8 tok/s**（p50，888 请求；用户说的 160） | `strata-main-lp.log` | **125B IQ3_XXS** | **decode** |
| **1482 / 3824 / 4681**（p50 / p90 / max，n=888） | `strata-main-lp.log` | 同一个 125B pack | **prefill** |
| **982 tok/s** | `build-lp\strata.exe` 实测 2047 token | 同一个 125B pack | prefill，**短 prompt，固定开销未摊开** |
| **62~70 tok/s** | `bench_infer.py`（WSL2，修复前） | 同一个 125B pack | prefill（2047 token 一次前向） |
| **224.5 tok/s**（6 步中位 9.1 s；稳态 **465**，见 §1.1） | `bench_cmp.py` / `_mb_lora.py`（WSL2，修复后） | 同一个 125B pack | prefill |
| **2.0 tok/s**（506.9 ms/token） | `bench_decode.py`（WSL2） | 同一个 125B pack | **decode** |

同一模型、同一份 pack，两档各自比：

- **prefill**（训练是 prefill 形状，这是训练该看的数）：引擎 **1482（p50）~3824（p90）** → 训练
  稳态 **465**，差 **3.2~8.2×**（6 步口径 225 是保守下界，差 6.6~17×；修复前 16~40×）
- **decode**：引擎 172.8 → 训练 2.0，差 **86×**（decode 路径没有吃到这一轮的修复，见 §6）

注意 2047 token 那个 982 不能当基准：prefill 有固定启动开销，短 prompt 摊不开，日志里 888 个真实
请求的 p50 就是 1482、p90 3824、max 4681。拿 982 比会把差距缩小一半。

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

| 变体 | s/step（中位 / best） | tok/s（中位 / best） | 显存·训练段 | **额外显存** | RSS·训练段 | 额外 RSS | 探针 Δ |
|---|---|---|---|---|---|---|---|
| 推理基线（no_grad） | **9.1 / 6.4** | **224.5 / 318.8** | 34.30 GiB | — | 46.02 GiB | — | — |
| **反传 + ckpt** | **24.3 / 20.1** | **84.4 / 102.0** | 38.20 GiB | **+3.90** | 46.86 GiB | +0.84 | +0.96 |
| **MeZO**（queries=1） | **28.3 / 15.0** | **72.4 / 137.0** | 34.70 GiB | **+0.40** | 46.09 GiB | +0.07 | −0.23 |
| 反传·不开 ckpt | 仍跑不动 | — | — | — | — | — | — |

配置：`STRATA_VRAM_PACK_GB=24`、`STRATA_PIN_GB=30`（见 §7.5），6 步同 seed 同批块。
单步抖动大（infer 6.4~19.2 s、mezo 15.0~64.0 s），中位与 best 都列出来是**必须的**：
两者在这个抖动下落在同一档里，见下面的配对讨论。

> 上一版同一张表（修复前，`STRATA_VRAM_PACK_GB=16`、43.8 GiB 全 pageable）：
> 推理 45.9 / 31.4 → 44.6 tok/s；反传+ckpt 84.3 / 75.1 → 24.3 tok/s；MeZO 62.8 / 52.0 → 32.6。
> **前向 5.0×、反传 3.5×、MeZO 2.2×。**

「跑不动」是 OOM，原因见 §4，不是物理下限：上一版实测 PyTorch 已分配 **92.63 GiB**
（48 GiB 卡，ctx 钉住 staging 缓冲）。现在 grad 下每专家独占一块显存（§5 第二轮第 4 条），
权重侧的下限变成「24,576 块全部活到反传结束」= **59.8 GiB** 再加激活 —— 仍然远超 48 GiB，
所以 ckpt 还是必须开，只是账从 92.63 降到 ~60+，不到 66 GiB 的清账方式仍是 §4/§7.6 ③。

配对可验证：两个变体确实是同一批块，同一步的 loss 对得上 ——
`sft 0.2596 / 0.1474 / 1.0052 / 1.4279 / 0.9622 / 0.8980` ↔
`mezo loss+ 0.2626 / 0.1460 / 1.0066 / 1.4393 / 0.9974 / 0.9176`。

**谁快：中位说反传赢（24.3 vs 28.3），best 说 MeZO 赢（15.0 vs 20.1），两者落在同一档
噪声里。** 这个抖动不是算法造成的（见 §7.5 的显存超额订阅），所以更适合拿 best 比：
20.1 / 15.0 = **1.34×**，正好等于「ckpt 下反传 2.7 个前向 vs MeZO 2 个前向」=
2.7 / 2 = 1.35× 的模型预测。也就是说，**反传之所以在长序列上落后，唯一的变量仍然是
ckpt**；把 ckpt 去掉（§7.6 ③）它才真正反超。

口径：显存是 `torch.cuda.max_memory_allocated()`，各脚本都在训练循环前
`reset_peak_memory_stats()`，所以「训练段」不含加载期临时缓冲。RSS 是进程树常驻集。
**中位数跳过第 1 步**（专家缓存还没热：推理档第 1 步 19.2 s、第 2 步 6.4 s），只统计
2~6 步；`best` 取全部步的最小值。因为只有 4~6 步、绝对值偏保守，**看同一步内的配对比值
更可靠**。

### 1.1 6 步太短：稳态是 18~20 s/步（103~120 tok/s）

同配置再跑 14 步（`sft.py --steps 14 --ckpt`，seed 0），前 7 步之后曲线就平了：

```
step  1   29.1s          step  8   18.3s
step  2   19.8s          step  9   18.0s
step  3   28.6s          step 10   18.9s
step  4   26.4s          step 11   19.1s
step  5   24.4s          step 12   19.6s
step  6   24.9s          step 13   19.8s
step  7   32.7s          step 14   17.1s
```

| 口径 | s/步 | tok/s |
|---|---|---|
| 14 步总计（含首轮一次性成本） | 22.6（316.7 s / 14） | **90.5** |
| 第 2~14 步中位 | 19.8 | 103.4 |
| **第 8~14 步中位（稳态）** | **18.9** | **108.3** |
| 最差一步（step 7） | 32.7 | 62.6 |

所以 §1 表里 24.3 s / 84.4 tok/s 是**保守下界**：6 步窗口正好落在首轮里。稳态 **108 tok/s**，
比 80 的目标高一档。同类现象在 `_mb_lora.py` 里更干净地复现了（同一块数据连跑三遍）：

```
第 1 遍 11.97 s → 第 2 遍 4.63 s → 第 3 遍 4.53 s   （H2D 三遍都是 28.85 GiB）
```

也就是说前向的稳态单价是 **4.4 s / 465 tok/s**（换块 8 个样本中位），而 18.9 / 4.4 = **4.3×**，
比 ckpt 模型预测的 2.7× 多出约 1.6 个前向当量（≈7 s）。**这 7 s 已经拆开了，见 §7.8**：
不是显存分配、也不是同步点，而是「反传本身 ≈ 2 个前向」这个实测值 —— 原模型里的 0.74×
是个错估计。

## 2. 短序列（内置身份数据，T≈17，12 步）——**修复前口径，未重测**

| 变体 | s/step（中位 / best） | 显存·训练段 | 额外显存 | RSS·训练段 | 探针 Δ |
|---|---|---|---|---|---|
| 推理基线 | 2.6 / 2.4 | 25.80 GiB | — | 45.42 GiB | — |
| **反传·不开 ckpt** | **5.4 / 4.3** | 35.50 GiB | **+9.70** | 46.08 GiB | **+22.65** |
| 反传 + ckpt | 9.1 / 7.9 | 26.40 GiB | +0.60 | 46.07 GiB | +22.45 |
| MeZO | 7.5 / 6.2 | 26.20 GiB | +0.40 | 45.69 GiB | −0.17 |

这套数字是 QSA 修复之前测的，**没有重测**，原因：T≈17 时 QSA 循环只有 17 次迭代
（2047 → 17），这轮修复对它几乎没有影响；而 T=2048 才是真实训练形状。§3 的
「排序翻转」结论不依赖这些绝对值。

## 3. 结论：排序随序列长度翻转，开关是 ckpt

| | 短序列（不 ckpt） | 长序列 T=2048 |
|---|---|---|
| 谁快 | **反传 5.4 vs MeZO 7.5（1.39×）**（修复前口径，§2） | **同一档噪声里**：中位 24.3 vs 28.3（反传赢）、best 20.1 vs 15.0（MeZO 赢），比值 1.34× ≈ 2.7/2，见 §1 |
| 谁省显存 | MeZO（+0.40 vs +9.70，省 9.3 GiB） | MeZO（+0.40 vs +3.90，省 3.5 GiB） |

原因是一个前向的单价：

- 短序列不需要 ckpt。反传 = 1 前向 + 反传（0.74× 前向，QDoRA/fused NF4 下要留的激活
  很少）≈ 1.74 个前向当量；MeZO = 2 个前向 + 小量扰动/更新。**反传更便宜**，于是赢。
- 长序列必须 ckpt（否则 OOM，§4）。ckpt 让反传 = 1 前向（存输入）+ 1 次**重算前向** +
  反传 ≈ 2.7 个前向当量，而 MeZO 还是 2 个。**MeZO 更便宜**，于是反超。

所以「MeZO 一定慢」和「反传一定快」都只在特定序列长度下成立。MeZO 的显存优势
（贴住推理基线，+0.4 GiB）在两档都成立。

QSA / PLE / 锁页这三项修的是**两条路线共用的那一次前向**（MeZO 走两次前向、反传走
一次前向 + 一次 ckpt 重算），所以它们同比例受益，**不改变上表的排序**：前向 45.9 → 9.1 s
之后，反传 84.3 → 24.3（3.5×）、MeZO 62.8 → 28.3（2.2×）—— 反传收益更大，是因为它多吃
一次前向（ckpt 重算），前向单价降了两次。稳态口径（§1.1）同理：前向 465 tok/s 时反传
108 tok/s，比值仍是那个 4.3×。

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
按更新后的前向单价算，15.8 s ≈ 1.74 个前向当量（2047/15.8 ≈ 130 tok/s），即**反超 MeZO**。

> 补充（`45ce80c9`）：现在 grad 下每专家是**独占**显存块，不再与 staging slot 别名，
> 于是「一块留一份」的下限就是 59.8 GiB 全在图上 —— 机制换了，结论没变，这条仍然是
> 去掉 ckpt 的唯一正路。

顺带说优化器：LoRA 可训练参数约 1.15e8，AdamW fp32 的 m+v ≈ **1.4 GB**，占那 66 GiB 的
约 2%。换成 8-bit 状态能省 ~1 GB —— 不是这次爆显存的原因，但等 66 GiB 消掉之后它就
占到可观比例了，届时值得一起做。

## 5. 两轮修掉的缺陷

### 第一轮

第 3 条对结论影响最大：它说明「同配置两次跑的 loss 差 0.06~1.44」不全是数据顺序造成的，
而是一半权重被读错。

1. **整块 H2D 与反传别名冲突（回归）**。`loader_qwen4.py` 的「整块一次 `copy_` + 显存侧
   四段 view」返回的是共享 staging 缓冲上的视图，slot 会轮转覆盖。纯前向无所谓，但反传时
   fused 函数刚把 packed/absmax 存进 ctx，缓冲就被下一组覆盖 →
   `variables needed for gradient computation has been modified by an inplace operation
   ... [ByteTensor [2560, 320]] is at version N`。旧的 per-expert `.to(dev)` 每块都是独立
   新张量，所以从没踩过。当时该快路径只在 `not torch.is_grad_enabled()` 下启用
   （第二轮第 4 条用「每专家独立显存块」把 grad 也接回这条路径）。
2. **数据顺序没固定**。`sft.py` 用的是无 seed 的 `randperm`，不同 run 命中的专家数不同，
   s/step 的差异里混着大量数据噪声。现在
   `sft.py` / `sft_mezo.py` / `bench_infer.py` / `bench_cmp.py` 统一 `--seed`，
   四条变体走同一批块。
3. **显存那半 pack 被静默损坏（最重要，commit `df0b90ca`）**。这一条推翻了我上一版写在
   这里的结论「这条前向本身不可复现，loss 差 <0.1 nats 是噪声」——**那个漂移是 bug，不是
   模型噪声**。`_TieredPack._ensure` 用**单个** pinned stage 缓冲逐 chunk `preadv`，而
   `copy_(stage, non_blocking=True)`（源是 pinned ⇒ 真异步 DMA）还在读它，唯一的
   `synchronize()` 在整轮循环之后。RAM 那半是 preadv 直接进 `cpu_buf`（同步）所以完好，
   **只有驻显存那部分被覆写**：同一份文件两次加载对比，layer 0 的 512 个专家里
   **150 个字节不同**，而全走 RAM staging 的 layer 40 一个字节不差。
   判据链条（全部是指纹，不是推断）：

   | 检查 | 结果 |
   |---|---|
   | 同进程连续 3 次前向 | 逐位一致（digest SAME） |
   | 两个进程同一输入 | 三个用例 CE 全不同（0.5179/0.4094、0.0331/0.0254、1.1918/0.9934） |
   | 1069 个参数/缓冲跨进程 | **0 差异**（稠密权重没问题） |
   | 逐模块输出跨进程 | 到 `layers.0.mlp.gate`（路由 logits）为止**逐位一致**，第一处不同是 layer 0 第一个 `experts.act_fn`；命中专家数 116 vs 116 相同 |
   | 专家块指纹 | layer 0（512 全 resident）**150 个不同**，layer 40（512 全走 RAM）**0 个不同** |

   「同进程稳定、跨进程变」正是加载期状态决定的特征，不是非确定 kernel（那会在进程内也散）；
   `use_deterministic_algorithms(True, warn_only=True)` 也没有任何警告。
   修法：两个 pinned stage + 每 stage 一个 Event（覆写前等自己上一次 DMA），保留 preadv/DMA
   重叠。验证：前 512 块「文件字节 vs 显存」**MATCH**（修复前 MISMATCH）；5123 个张量指纹
   跨进程差异 **598 → 0**；跨进程 CE 两个用例现已相同。

   ⇒ **本文件里所有 loss/CE 都产自「前 13 层专家被损坏」的基座**（6579 块常驻 ≈ 12.8 层），
   计时不受影响（工作量没变），但 sft vs MeZO 的配对必须重测。

### 第二轮（commit `b9af996a` + `45ce80c9`，`_mb_split` 口径前向 17.9 → 5.0 s、`bench_cmp` infer 档 45.9 → 9.1 s，同配置训练 84.3 → 24.3 s；稳态 465 / 108 tok/s，见 §1.1）

1. **QSA 索引器逐 query 循环**（§7.3）—— 稠密路径 12.73 → 0.46 s，掩码逐位一致。
2. **PLE 表逐行反量化**（§7.4）—— 0.65 → 0.07 s。
3. **部分锁页**（§7.5）—— 43.8 GiB 里锁 30 GiB；pageable 源的同步阻塞（每步约 4 s CPU
   无法派发）消失。任何一片 pin 失败就减半重试，不假设上限（本机 32 GiB，Windows 侧不同）。
4. **grad 模式也走整块 H2D**。上一轮为了躲别名冲突把这条快路径 gate 在 `not is_grad_enabled()`，
   反传因此退回「线程池 pinned 拷贝 + 每专家 4 次 `.to()` + 4 次 view」。现在 grad 下**每个
   专家分配独立显存块**（多一次 `torch.empty`，缓存分配器约 2 µs），不再与轮转 slot 别名，
   于是同样只做 1 次整块 `copy_` + 4 次 view。代价按层计：一层 512 × 2.49 MiB ≈ 1.27 GiB，
   该层反传完即释放 —— 只有开 ckpt 才付得起（不开 ckpt 权重本来就全进图）。
   效果：sft+ckpt 39.4 → 24.3 s（中位）。数值上 step 1 的 loss 与 grad_norm 与改动前
   逐位相同（0.2596 / 0.2793），之后因反传累积顺序不同有 ~0.2% 漂移（与第一轮第 3 条
   同一个未收敛的非确定性源）。

## 6. 平台：WSL 是不是拖慢了？（结论：不是主因）

- 传输：WSL 里一次 43.8 GiB `pin_memory` 直接 `AcceleratorError`，退回全 pageable；微基准
  pageable 整块直连 **~20 GB/s**，引擎在 Windows 上的 probe 是 **24.1 GB/s** —— 差 1.2×。
  分片锁页可行（单次上限 32 GiB，现在锁 30 GiB，§7.5），pinned 块级 H2D 21.7 GiB/s。
- 计算：同一张卡、同一套驱动（通过 WSL 直通），kernel 层面没有平台差异。
- ⇒ 从 Windows 能拿回的最多是传输那 1.2×，而当前差距是 **3.2~8.2×**（稳态口径）。剩下这部分里，
  已经有一块被证明**纯粹是实现问题**（QSA 逐 query 循环，§7.3，吃掉了 74% 的墙钟），
  剩下的主因是 dequant→bf16→GEMM（每专家只分到 ~80 行，小 GEMM 跑不满 GPU）+ 24,576 次
  逐专家启动，引擎是量化内核直接算。

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

## 7. 这一轮：前向 17.9 s → 9.1 s，钱花在哪

### 7.1 kernel tile 调参是死路（上一版结论，仍然成立）

单专家形状 `gu(1280×2560)` + `dn(2560×640)`，T=64（= T2048 × top_k 8 / 256 experts）：

| tile (BT,BO,BK) | ms | TFLOPS | 权重带宽 |
|---|---|---|---|
| **默认 (16,16,auto)** | **0.081** | 5.20 | 21.6 GB/s |
| (16,32,auto) | 0.086 | 4.90 | 20.3 |
| (64,32,64) | 0.084 | 4.97 | 20.6 |
| (16,64)/(32,32)/(64,64) | `OutOfResources` | — | — |
| 对照 dequant+cuBLAS | 0.212 | 1.98 | 解码 0.180 + GEMM 0.032 |

`dn` 的最优是 (32,32) 0.050 ms（默认 0.055，仅 +9%）。默认已经最优，加大 tile 直接爆
共享内存。5 TFLOPS 只有峰值的 3%，21 GB/s 远低于 L2 带宽 ⇒ **既不是算力也不是带宽瓶颈**，
是 decode 的 LUT gather（`tl.load(code_ptr + nib)`，每权重元素一次 gather）与循环开销。

顺带确认融合 kernel 本身是对的：对 `gu` 比 dequant+cuBLAS 快 2.6×，对 `dn` 快 6×。

### 7.2 先量「谁在等谁」：GPU 忙 5.3 s，墙钟 17.4 s

上一版在这里写「剩下 ~9.6 s 未测量」。补上之后结论完全不同：

```
T=2048 一次前向：CUDA 总时间 5.30 s / 墙钟 17.4 s ⇒ GPU 空转 74%
               2,410,575 个事件，其中 cudaLaunchKernel 984,312 次
               24,661 次 Memcpy DtoH + 24,763 次 cudaStreamSynchronize
```

按 op 聚合的 CUDA 时间都很小（最大 `_nf4_fwd_kernel` 1.29 s、H2D 合计 2.20 s），
说明成本在「启动次数 × 每次开销」上，不在某个昂贵的 kernel 上。而 2.4e4 次 DtoH 同步
是最刺眼的信号：**每 2,047 次同步对应一层**（2,047 = T-1）。

### 7.3 真凶：QSA 稀疏索引器的逐 query Python 循环

`transformers/models/qwen4_exp/modeling_qwen4_exp.py:729`：

```python
for batch_idx in range(batch_size):
    for query_idx in range(seq_length):
        local_visible_indices = torch.nonzero(visible_token_indices[b, 0, q], ...)
        num_complete_blocks = local_visible_indices.shape[-1] // self.compress_ratio   # DtoH
        ...
        pooled_keys = self.k_layernorm(key_groups.view(...).float().mean(dim=1).to(dtype))
        block_key_states = apply_rotary_pos_emb(pooled_keys.unsqueeze(1), cos[...], sin[...])
        scores = torch.relu(q[b, q].float() @ block_key_states.float().T).sum(-1) / sqrt(D)
        selected_block_indices = scores.topk(...).indices
        ...
        selected_token_indices[b, query_idx, : selected_tokens.numel()] = selected_tokens   # DtoH
```

三个问题叠在一起：

1. `pooled_keys / k_layernorm / rope` 只依赖 **block**，与 query 无关，却每个 query 重算
   一遍 —— T=2048 就是 2,047 倍冗余；
2. `.shape[-1]` 与 `.numel()` 都是 **DtoH 同步**，每次同步都会排空前面的 kernel；
3. 12 个 QSA 层 × 2,047 query = 24,564 次同步/步，正好等于实测的 24,661 / 24,763。

纯 causal 掩码下 `visible(q) = [0..q]`，于是 `block_token_indices` 对所有 query 相同：
block 池化一次、一次 `einsum` 出 `(S, n_block)` 分数、一次 `topk`、gather 拼掩码。

校验（`STRATA_QSA_CHECK=1`，与逐 query 原实现逐元素对比）：**掩码差异 0/4,190,209**，
同一块数据 loss 前后都是 2.0236034393。只在「无 past_key_values、kv 长度 = 序列长、
掩码恰为下三角全 True」时启用，其它情况退回原实现。

### 7.4 拆开两条路径（stub 掉专家 forward）

`_mb_split.py` 把 `StreamingNF4Experts.forward` 换成 `zeros_like`，同一份权重跑两遍：

| | 墙钟 | PLE |
|---|---|---|
| A 完整前向 | **17.89 s** | 0.65 s |
| B 专家整段 stub | **12.73 s** | 0.58 s |
| A − B 专家路径 | **5.17 s（29%）** | |
| B 里非 PLE 的稠密/注意力 | 12.15 s（71%） | |

也就是说，上一版一直盯着专家路径（占了 29%），真正的大头是稠密路径的 12.15 s —— 全在
QSA 那个循环里。修完之后同一脚本变成 **专家 4.58 s（91%）/ 稠密 0.46 s（9%）**。

顺带另一处：`MmapNgramEmbedding._gather_rows` 对每个 unique 行做一次 mmap 切片 +
`gguf.dequantize`（T=2048 是 2,047 行，每行 **319 µs**）。这段是纯 CPU，CUDA profiler 里
完全看不见。改成「mmap 上取 `(total_rows, row_bytes)` 零拷贝视图 → numpy fancy-index 一次
搬进连续缓冲 → 只调一次 dequantize」，0.65 s → **0.07 s**，数值路径不变。

### 7.5 传输：能锁 32 GiB，锁上之后每块 112 µs 而不是 222 µs

`_mb_pin.py`（本机实测）：

| 源 | 2.49 MiB/块 | 239 MiB/块 |
|---|---|---|
| pageable, blocking | 10.96 GiB/s，222 µs/块 | 14.57 GiB/s |
| pageable, `non_blocking=True` | 12.57 GiB/s，193 µs/块 | — |
| **pinned, `non_blocking=True`** | **21.72 GiB/s，112 µs/块** | 21.60 GiB/s |

关键点：**pageable 源的 `copy_(non_blocking=True)` 并不异步** —— 它阻塞调用线程直到搬完。
43.8 GiB / 17,997 块 ⇒ 每步约 4 s 里 CPU 完全无法派发后续 kernel，GPU 同时空转。
单次可 pin 上限实测 **32 GiB**（33 GiB 直接 `AcceleratorError`），所以 `_TieredPack` 改成
「pinned 前缀 + pageable 余量」两段，失败按减半重试。

显存那半也要给够：`STRATA_VRAM_PACK_GB=30` 时 `peak_reserved` 到过 **50.1 GiB**，超过
卡的 47.99 GiB —— WSL 下 CUDA 会向宿主内存超额订阅，一旦触发就是 PCIe 换页，单步能从
6.4 s 抖到 19.2 s。最终取 **24 GiB 常驻 + 30 GiB 锁页**：allocated 34.3 GiB（infer）/
38.2 GiB（ckpt 反传），RAM 侧 30 + 6 + ~10 = 46/54 GiB。

### 7.6 剩余的账与下一档

前向的**稳态**单价是 4.4 s（§1.1；§1 表里的 9.1 s 是中位口径被首轮拖累，见 §7.7）。
`_mb_split.py` 那套（no_grad、无 LoRA）测到 5.04 s，其中专家 4.58 s、稠密 0.46 s ——
修完之后**稠密路径已经不是问题**。上一版这里猜「比 5.04 多出来的 4 s 来自 LoRA / train
模式」，§7.7 已证伪（LoRA +0.12 s、train 模式 +0.02 s）：那 4 s 就是首轮变慢。

专家那 4.58 s 里：融合 GEMM 约 1.3 s、H2D 约 1.6 s、逐专家 gather/scatter/elementwise
约 1.2 s，其余是每块传输与 kernel 之间的空隙。单项都不大，共同点是 **24,576 次逐专家
迭代**（配 7~8 个 kernel）。下一档要动的是：

| # | 动作 | 依据 | 预期 |
|---|---|---|---|
| ① | 把 silu*up 融进 gu GEMM、把 top-k 权重融进 dn GEMM 的 epilogue | 每专家 7~8 个 kernel，机器码级只省 2~3 个 | 专家段 -10~20% |
| ② | 按组做 grouped GEMM（一组 32 个专家的 block 在 staging 里本就连续） | 把 24,576 次启动降到 ~800 次 | 专家段数量级下降 |
| ③ | ctx 只存专家 id、backward 按 id 重流 | 现在 grad 下整层独占 1.27 GiB 显存，仍必须开 ckpt | 去掉 ckpt ⇒ 省掉那次重算前向（实测 +4.4 s/步） |
| ④ | B>1 | 实测 B=8 是 B=1 的 1.32× | 白拿 +32% |
| ⑤ | ~~拆 18.9 s 与 4.3× 前向之间的差额~~ | **已拆完，见 §7.8** | 结论：反传 ≈ 2× 前向，3.2~4.2× 是结构性的 |

③ 按稳态算：18.9 − 4.4（省掉的重算前向）≈ **14.5 s/步 ≈ 141 tok/s**，顺带省 3.9 GiB。
⑤ 的结论（§7.8）把 ① 和 ② 从「锦上添花」变成了**唯一的大杠杆**：每一步的 op 数会同时
乘进前向和反传（反传按倍率放大），所以省一个 op 等于省三份时间。

### 7.7 「首轮变慢」：一次性约 40 s，四项已排除，机制未定位

现象（`_mb_lora.py`，同一进程，同一块数据连跑三遍）：

```
第 1 遍 11.97 s   H2D 28.85 GiB
第 2 遍  4.63 s   H2D 28.85 GiB
第 3 遍  4.53 s   H2D 28.85 GiB
```

**传输字节数三遍完全相同，时间差 2.6 倍** ⇒ 不是数据、不是带宽、不是专家命中数。
在整条训练链上，它表现为「前 7 步每步慢 4~14 s，第 8 步起突然稳定」，两个独立进程都
精确停在第 8 步（累计读过约 2 倍 pack 的量），像是某个「读满一遍才热」的缓存。

排除掉的四项（都是实测，不是推断）：

| 嫌疑 | 做法 | 结果 |
|---|---|---|
| LoRA 挂载 | 同一进程 eval/train x 有无 LoRA 四组 | LoRA **+0.12 s**、train 模式 **+0.02 s** |
| 显存分配器碎片 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 逐步曲线**逐点不变** |
| host 页首次建立映射 | 启动时把 35.8 GiB 全部 DMA 一遍（2.2 s） | 前 7 步**照旧慢** |
| 显存容量压力 | pack 从 24 GiB 降到 20 GiB | 慢步模式**一模一样**，只是整体慢 10% |

也排除了宿主内存回收：训练期采 `/proc/vmstat`，`pgscan_direct` 恒为 0、`pgmajfault` 在
慢步里没有尖峰，`MemAvailable` 稳定在 10.6 GiB（紧张但不触发直接回收）。

还没做的：抓一个慢步 vs 快步的 CUDA profiler 对比（GPU 忙/墙钟缺口、`cudaMalloc` 次数、
Memcpy 次数）。**在那之前不要再靠换配置试** —— 这四项都是换配置，全无效。

实用结论：这是**一次性启动成本**，量级 40 s，真跑（几百步）会被摊平；基准脚本因为只跑
4~6 步，才会被它主导。所以本文件的长序列数字要按「§1.1 稳态」那一栏读。

### 7.8 反传步的账拆开了：3.2× 前向是结构性的，不是 bug

新增 `_mb_train.py`（前向 / 整步各自一对 CUDA event，反传由差值反推）。T=2048，同配置：

| 成分 | 改前 | 改后（本轮两处） |
|---|---|---|
| 前向（no_grad，ckpt 的第一次） | 4.46 s | 4.36 s |
| ckpt 段（重算 + 反传） | 19.26 s | 18.23 s |
| 整步墙钟 | 23.72 s | **22.59 s**（91 tok/s） |
| 整步 / 前向 | 4.31× | 4.18× |

CUDA-only profiler 给一步的形状（改前）：`_nf4_fwd_kernel` 2.55 s / 75,516 次、
HtoD 2.17 s(pinned) + 0.49 s(pageable) / 22,168 次、`_nf4_bwd_kernel` 1.84 s / 37,758 次，
合计 CUDA 9.75 s，而这一步约 19 s ⇒ **GPU 忙一半，另一半在等 CPU 派发**；事件数 131 万。
75,516 ≈ 24,576 个专家 × 2 个 GEMM × 1.54（部分层命中不满），也就是**重算确实把两个 GEMM
又跑了一遍**。

那 22.6 s 到底怎么解释？用 `_mb_expert_grad.py` 把「每专家循环」单独拿出来量（同形状、
权重换成 bf16 + cuBLAS，只量调度与 autograd）：

| 模式 | µs/expert | 相对 no_grad |
|---|---|---|
| no_grad | 98.9 | 1.00× |
| 建图（前向带 grad） | 112.8 | **1.18×** |
| 建图 + 反传 | 330.0 | **3.20×** |

⇒ **反传本身就是 2 个前向，不是 0.74 个。** 建图只贵 18%，所以问题不在显存分配、不在
同步点、也不在「重算前向带着 grad 跑」。ckpt 只是在这之上再加一次前向（1.18×）：
3.20 + 1.18 = 4.38×，与实测 4.18× 对得上 —— **这笔账闭合了**。

原因很朴素：这一层是「24,576 个微小算子」而不是「几十个大 kernel」，反传给前向的每个算子
各加 1~2 个反向节点（gather/scatter/silu/mul 都有 backward），dispatch 成本翻倍。所以 op 数
会**同时**乘进前向和反传。

本轮实际做掉的两处（都在 `loader_qwen4.py`）：

1. **grad 下整层一次分配**：原来每专家 `torch.empty(2.49 MiB)`，一层 512 次、一步 24,576 次
   （重算再来一遍）。改成整层 1.27 GiB 一次申请、逐块切片 —— 分配器调用 24,576 → 48 次/步。
   切片互不重叠，仍不会与槽位轮转别名（那才是当初 version counter 报错的成因）。
2. **去掉 grad + 整块路径上无效的槽位同步**：那条路径不写 staging 槽位，`slot_sync` 只剩
   「等 GPU 跑完上一次 record」这个副作用 —— 每 32 个专家一次**阻塞同步**（一层 16 次、
   一步 768 次），把 CPU 派发和 GPU 执行硬串起来。现在 `slot_in_use` 只在真会覆写槽位时才等。

合计 23.72 → 22.59 s（-5%），**数值不变**：`_mb_train` 的 5 步 loss 轨迹
（1.97 / 1.88 / 1.79 / 1.70 / 1.60）与改动前逐位可比。

⇒ 剩下的唯一大杠杆是 §7.6 的 ① + ②：把每专家的 8 个算子压到 2~3 个（融合 epilogue、
分组 GEMM）。按上表，每省一个算子同时省掉它在反传里的 1~2 个节点和一次 dispatch ——
实测倍率 **3.2×**，不是 1×。

### 7.9 显存 ↔ 速度的取舍线（同一套代码，只改 `STRATA_VRAM_PACK_GB`）

`_mb_train.py`，T=2048，4 步稳态，`STRATA_PIN_GB=30` 不变：

| pack 常驻显存 | torch 峰值 | 前向 | 整步墙钟 | tok/s | 相对 24 GiB |
|---|---|---|---|---|---|
| 24 GiB | 38.2 GiB | 4.36 s | **22.59 s** | **91** | — |
| 20 GiB | 34.2 GiB | 5.12 s | 23.59 s | 87 | -4.4% |
| 16 GiB | 30.2 GiB | 5.52 s | 25.98 s | 79 | -13% |

读法是「每少 4 GiB 常驻，前向多 0.6~0.8 s」—— 因为少驻留一块就得多搬一次 H2D，而搬运
是这条路径里唯一按字节线性计时的事。所以：

- 想给别的东西腾显存，**能腾到 20 GiB 还不掉出 80 tok/s**；16 GiB 就掉到 79 了。
- 反过来想更快，把 pack 常驻调**大**并不可行：单次 `pin_memory` 上限 32 GiB、
  峰值一旦越过 47.99 GiB 就会触发 WSL 向宿主内存换页（§7.5），40 GiB 那条配置实测抖动到
  无法计时。24 GiB 是这台机器上的拐点。

## 8. 复现

```bash
# 长序列（同 seed 同批块）；24 GiB 常驻 + 30 GiB 锁页是本机 48G 卡 + 54G WSL 的甜点
export STRATA_VRAM_PACK_GB=24 STRATA_PIN_GB=30
python3 bench_cmp.py --gguf <shards...> --pack /root/experts_nf4.bin \
    --tokenizer /root/models/qwen36-35b-a3b \
    --pretrain-data /root/pretrain_blocks.npy --seqlen 2048 \
    --steps 6 --seed 0 --only infer,sft+ckpt,mezo --out /root/cmp.json

# 短序列
python3 bench_cmp.py --gguf <shards...> --pack /root/experts_nf4.bin \
    --tokenizer /root/models/qwen36-35b-a3b --steps 12 --seed 0 --out /root/cmp_short.json
```

`--only infer,sft,sft+ckpt,mezo` 可只跑其中几个。诊断脚本：

| 脚本 | 回答什么 |
|---|---|
| `_mb_pin.py` | 单次能 pin 多大、pageable/pinned 的块级 H2D 带宽 |
| `_mb_phase.py` | 专家段事件计时（不在层内同步，保住跨层流水） |
| `_mb_split.py` | stub 掉专家 forward，把专家路径与稠密路径的墙钟拆开 |
| `_prof_step.py` | CUDA-only profiler：op 级时间 + 启动次数 + GPU 忙/墙钟缺口 |
| `_mb_lora.py` | eval/train x 有无 LoRA 四组；`--blocks --passes 3` 复现「首轮变慢」（§7.7） |
| `_mb_train.py` | 一步训练拆成「前向 / ckpt 段（重算+反传）」两段事件，外加 CUDA-only profiler 步（§7.8） |
| `_mb_expert_grad.py` | 同形状的每专家循环只量调度与 autograd：no_grad / 建图 / 建图+反传（§7.8） |
| `_det_*.py` | 指纹法定位数值差异（见 §5 第一轮第 3 条） |

## 9. 怎么跑（默认入口）

`train.py` 是唯一的默认入口：模型默认取**桌面快捷方式会起的那份**（`launcher/launch-choice.json`
里钉住的 `strata-*.json`，见 §10），数据默认「改变身份」，算法默认反传 SFT。

```bash
python3 train.py                                  # 身份数据 + 反传 + 钉住的模型
python3 train.py --dry-run                        # 只打印最终命令，不加载模型
python3 train.py --list                           # 算法 / 目标函数 / 数据源
python3 train.py --algo mezo --steps 20           # 同数据，零阶（显存贴住推理基线）
python3 train.py --data jsonl:/root/my.jsonl      # 换数据只换这一行
```

Windows 上双击 `Strata Train (WSL2)`（= `START-TRAIN.bat`）：训练只能在 WSL2 跑
（Triton / fla / `kernels` 都没有 Windows 构建），推理则在 Windows（引擎是 Windows 二进制）。
两边用同一份权重与同一套路径，唯一交界是导出的 LoRA GGUF。

**「只靠前向做 RL」跑通过的证据**（T=2048 之外的小样本，只为验链路）：
`--algo mezo --objective grpo --rollouts roll_smoke.jsonl --steps 2` →

```
rollouts: 4 条 / 2 组，有效组 2（std=0 的组优势全 0，不会动参数）；old logprob 来自 local，重算 4 条
step 1: obj+ 0.0548  obj- 0.0378  proj_grad 8.4674e+00  ratio 1.0506  adv0 0.00  peak 33.8 GiB  3.6s
step 2: obj+ 0.0800  obj- 0.0434  proj_grad 1.8303e+01  ratio 1.1271  adv0 0.00  peak 33.8 GiB  3.2s
saved /root/lora_grpo_smoke.pt: 96 tensors, non-zero 96
```

`proj_grad` 非零、96 个 LoRA 张量全部非零 ⇒ 零阶更新确实走到了参数上；`ratio` 从 1.05
漂到 1.13 是两次更新之后的应有表现（`adv0 = 0` 表示没有优势为 0 的死样本）。
显存 33.8 GiB，比反传那档低 4.4 GiB —— 全程 no_grad。

## 10. 模型默认值从哪来，以及它为什么以前会错

`launcher/start_server.py:stock_config()` 原来按 **mtime 取最新的 `strata-*.json`**。为任何
别的模型写一次配置（35B 适配器开发、另一个量化、benchmark family）都会让那份成为新的
默认值 —— 双击 `Strata CN.lnk` 起的是 35B，而日志里才看得出来。

现在改成**显式钉住**：`launcher/launch-choice.json` 的 `"config"` 指向 `strata-main-lp.json`
（125B IQ3_XXS），`--use <file>` 改它并记住；mtime 只在没有钉子时兜底，并在启动时说明。
启动日志多两行，把「用哪份配置、哪个权重文件」直接打出来：

```
[start-server] 已钉住的模型：strata-main-lp.json（记住了 —— 要换就用 --use <文件>）
[start-server] 权重：D:\...\Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
```

`train.py` 复用同一份逻辑（直接 import `start_server.py` 的 `stock_config()`/`read_cfg()`），
所以「训练用的模型」和「正在跑的模型」不会分叉。

## 11. 剩下的三条杠杆（都**没**实施，附判据与预期）

稳态基线：**22.59 s/步 = 90.7 tok/s**（T=2048、batch=1；前向 4.36 s + ckpt 段 18.23 s）。
每步 48 层 × 512 专家 = **24,576 次专家访问**，单次约 330 µs（微基准：no_grad 98.9 µs、
建图 112.8 µs、建图+反传 330.0 µs）。折算 **≈3 TFLOPS**，即 4090 bf16 峰值的 **2%**
⇒ 当前不是算力瓶颈，是「小 tile + 逐次 launch」的填充率瓶颈。这就是下面三条的依据。

| # | 杠杆 | 改什么 | 预期 | 工作量 | 风险 |
|---|---|---|---|---|---|
| ① | 融合 epilogue | `_nf4_fwd_kernel` 里把 `silu(gate)*up` 直接写进 gu GEMM 的 epilogue；dn GEMM 的 epilogue 乘 `grp` 后直接累加 | 每专家 9 个 dispatch op → 6~7 个 ⇒ **18.2~20.8 s（98~112 tok/s）** | 中（两个 kernel，逐值需回归） | 低（fp32 epilogue 比现在的 bf16 中间结果更准） |
| ② | **分组 GEMM** | 一层 512 个专家合成一次 segmented GEMM（按 `bounds` 切段），反传同样分组 | 24,576 次 launch → 96 次；tile 填充率从「32×1280 单专家 ≈ 10 CTA」变成 5,120 CTA 并发 ⇒ **预计 8~12 s/步（170~250 tok/s）** | 高（Triton grouped GEMM + 分段 reduction + backward） | 中（累加顺序变，必须给 CE 门禁） |
| ③ | 去掉 ckpt | ctx 只存专家 id，backward 按 id 重流权重；权重永不进图 | 省掉整段重算 ⇒ **18.2 s/步（113 tok/s）** | 中高（自定义 autograd，需与 ckpt 路径逐值对照） | 中（配错会静默算错梯度） |

**②是唯一改变算术强度的一条**，①②③里只有它能把「2% 峰值」往上抬；①③只是把
「2% 峰值 × 更少的 op / 更少的遍数」乘小。三者可叠加，但 ② 与 ① 在同一个 kernel 里，
先做 ② 会把 ① 重写一遍，所以顺序建议 **③ → ① → ②**（③ 与 kernel 无关，先拿 +24% 的无风险收益）。

**为什么先不动 ②**：它要重写前向与反传两个 Triton kernel 并重排 `index_add_` 的累加顺序，
必须有「同 seed 同数据下 loss/CE 相对 ckpt 路径的偏差」这道门；而这台机器每验证一次要独占
GPU ~40 分钟（模型加载 ~100 s + 40 步）。开工前需要一次明确的长跑窗口。

**显存侧还没有动的**：24/20/16 GiB 三档只压了显存（见 §7.9）。RAM 侧（pinned 30 GiB +
pageable + RSS 46 GiB）尚未做过任何取舍实验 —— 这一项与速度无冲突，是先能拿的。

## 12. 显存与内存：能压到多小（2026-10-30，**标准入口** `train.py`）

§7.9 那张表是诊断脚本（`_mb_train.py`）的口径。换成真实入口后多出一笔开销，而它正好
是「能压到多小」的决定项。

### 12.1 硬约束：pack 必须整个待在「显存 + RAM」

pack 59.77 GiB（24,576 × 2.49 MiB），当前实现零读盘（§7.5 论证过读盘的代价：
9~18 tok/s、GPU util 1%）。WSL 的 RAM 上限是 58 GiB（`.wslconfig memory=60GB`），于是

    显存 ≥ pack − (RAM 上限 − 进程其它开销)

24 GiB 档 RSS 53.8 GiB ⇒ 其它开销约 18.2 GiB。按这个数算下界该是 20 GiB —— **实测不是**：
16 GiB 档跑通了。差别在那 18.2 GiB 里最大的一块：**两次加载一共读过的 ~15 GiB GGUF 映射页**，
它们是干净的 file-backed 页，**可以还**。

### 12.2 把映射页还给内核（本轮唯一的代码改动）

`GgufShard.drop()` / `_release_mmap_pages()`：每把一个张量搬上显存就 `madvise(MADV_DONTNEED)`
它占的那段页，全部加载完再整体还一次。唯一还会读它们的是 PLE（`MmapNgramEmbedding` 本来
就按行读、表不驻留），代价是每步几 MB。

不做的后果不是「慢」而是「跑不动」：16 GiB 档在基座加载阶段 `available` 归零、10 分钟
走不出加载（实测日志：`tiered pack ... loaded 300s` 之后再无输出）。做完 300~344 s 加载完。

### 12.3 实测表（T=2048，3 步）

| pack 常驻显存 | 训练段峰值 `max_memory_allocated` | RSS 峰值 | 步时（末步） | tok/s | 结论 |
|---|---|---|---|---|---|
| 24 GiB | 34.5 GiB | 53.8 GiB | **20.5 s** | **99.9** | 基准 |
| 16 GiB | 26.5 GiB | **57.3 GiB** | 29.0 s | 70.6 | 省 8 GiB 显存，慢 41% |
| 12 GiB | — | 需 ~61 GiB | 换页卡死 | — | **不可行** |

（12 / 8 / 6 / 4 四档是同一批实验里主动停掉的：12 GiB 档 10 分钟没走出加载、`available`
归零；8/6/4 的 RAM 需求分别是 53.8/55.8/57.8 GiB + 其它开销，只会更糟。）

读法：**显存每省 8 GiB，速度掉 41%，RSS 同时涨到贴着 58 GiB 的顶**。16 GiB 是这个实现的
下界，不是「再压一点会慢」—— 再压就是 RAM 装不下 pack。想要更小只有放弃零读盘、按层读盘
（§7.5：9~18 tok/s，慢一个数量级）。

16 GiB 档的步时波动比 24 GiB 大（44.3 / 21.1 / 29.0 vs 22.1 / 15.7 / 20.5）：RAM 贴顶时
内核回收页的抖动会落到步上。稳态（末步）取 29.0 s。

### 12.4 怎么用

```bash
python3 train.py --algo mezo                 # 默认 16 GiB 常驻（$STRATA_VRAM_PACK_GB）
python3 train.py --vram 24                   # 更快，多占 8 GiB 显存
python3 train.py --vram 16 --pin 20          # 锁页也调小：RSS 再松几 GiB
```

`--vram` / `--pin` 只是把两个环境变量包成参数（`train.py` 同进程 import，设 env 即生效），
表头会回显实际档位，日志自证。

## 13. 标准 GRPO：进程内采样 + 组内 batch（2026-10-30）

**问题**：GRPO 的标准形态是 on-policy —— 每一轮用**当前策略**采一组轨迹，组内归一化优势，
再更新一步。原来的实现是离线回放：`--rollouts` 一份 jsonl，`--steps 20` 就把同一批组循环
用 10 遍（`groups[(step-1) % len(groups)]`）。`old` 一旦固定，第 2 步起 `ratio` 必然偏离 1
—— 那已经不是 GRPO，是「带重要性权重的离线策略梯度」。

**为什么不能拉引擎来采样**：引擎要 ~40 GiB 常驻（expert cache 39.97 GiB，§7.5），
训练侧 34.5 GiB，48 GiB 卡装不下两个。轮转的话（起引擎采一批 → 关引擎 → 训）
每轮多付 100 s 引擎加载 + 300 s 训练加载，摊到 20 步上是 20 s/步的固定成本。

**做法**：让学生自己采。MeZO 只需要**前向**，采样也是**前向** —— 同一条路径，
**显存不涨**。三个让它可以接受的设计：

1. **组内 batch 采样**（`sample_group`）：K 条共享同一个 prompt 前缀，一次 decode 处理 K 条，
   每 token 的专家搬运摊到 K 条上。
2. **`old` 就地记下**：采样时的 logits 顺手算 logprob，`ratio` 起点严格 = 1；
   既不用起引擎，也不用「再跑一遍重算 old」。
3. **组内 batch 前向**（`grpo_loss(batched=True)`）：K 条拼一个 batch，只 pad 回答段
   （同组 prompt 等长，不需要左 padding），专家只搬一遍。

### 13.1 实测（125B，T=2048 的 pack，`--vram 16`）

```bash
python3 train.py --algo mezo --objective grpo \
    --prompts grpo_prompts.jsonl --group 4 --max-new 16 --steps 3 --lr 1e-6 --vram 16
```

| step | sample | 整步 | 峰值显存 | ratio | uniq | rew | proj_grad |
|---|---|---|---|---|---|---|---|
| 1 | 30.6 s | 48.9 s | 26.2 GiB | 0.9955 | 4/4 | 0.50 | −1.49 |
| 2 | 10.7 s | 21.4 s | 26.2 GiB | 0.9974 | 4/4 | 0.30 | −3.63 |
| 3 | 9.7 s | 20.3 s | 26.2 GiB | 1.0100 | 3/4 | 0.00 | **0** |

读法：

- **`ratio` 从 1 附近起步**（0.9955 / 0.9974 / 1.0100）—— 这才是 on-policy 的样子。
  离线回放第 2 步起就已经飘到 1.1 以上。
- **第 1 步采样 30.6 s、之后 10 s**：专家缓存没热时 decode 慢（和反传路径首步慢是同一个原因）。
  采样占整步的一半，训 3 步以上就摊薄了。
- **第 3 步 `rew 0.00` ⇒ 全组奖励为 0 ⇒ `adv0 1.00` ⇒ `proj_grad 0`**：那一步什么都没学。
  这是 GRPO 的冷启动（组内无方差），**日志必须让它可见**，否则「跑完了」看着像「学会了」。
  正确顺序是先用 SFT/MeZO-ce 把行为抬起来，再上 GRPO（本目录 §3 的老结论不变）。
- 96 个 LoRA 张量全非零，峰值 **26.2 GiB** —— 比反传 SFT 的 38.2 GiB 少 12 GiB。

### 13.2 代价与边界

- 采样是**学生自己**做的，所以速度受训练路径的 decode 限制（每 token 要把该层专家搬一遍）。
  引擎有 expert-cache + MTP，同一模型能到 172.8 tok/s；训练路径只有 ~2 tok/s（4 条 batch 后）。
  **这就是「少显存」的价钱**：拿引擎的速度换掉 40 GiB 的常驻。
- `--resample-every N`：N>1 用同一批轨迹多训几步（`ratio` 会开始飘），是显存与速度之外的
  第三个旋钮；默认 1，即标准 on-policy。
- `--batch-group off`：逐条前向，仅用于对照（正确性由 `test_train.Sampling` 与
  `GRPOLoss.test_batched_matches_serial_when_lengths_differ` 守着）。

### 13.3 组内 batch 值多少（K=4，4 条 × 16 token，同一进程内各测 3 次取均值）

| `grpo_loss` | 耗时 | 相对 |
|---|---|---|
| `batched=True` | **1.89 s** | 1.00× |
| `batched=False`（逐条） | 4.25 s | 2.25× |

**2.25× 而不是 4×**，因为 batch 后的单次前向确实比单条慢（pad 段 + 更大的 tile），
但省掉的 3 次专家搬运是主导项。这个倍数随 K 走：K=8 时省 7 次，K=4 时省 3 次。

同一次加载里还量到：`sample_group` 4 条 × 16 token = **29.5 s**（≈2.2 tok/s 墙钟，
4 条 batch 摊过），`uniq 4/4`（真有方差，不是一组同文）。

## 14. 并行：对接引擎的批槽，不是自己实现一套

先纠正一个容易记错的事实：**上游的并行推理就是批槽，而且已经合并进主线了**（`docs/BATCHING.md`，
引擎部分 PR #559，opt-in 是 #465）。`--batch N` / `--slots N` 在 `src/program/generate.cpp` 里是现役
代码，本地构建也能跑到它（`--batch: 4 slot sessions on CUDA0`、`batch windows of up to 4 sequences`）。
上游**正在处理**的是批槽的短板，不是批槽本身：

| 上游 | 状态 | 说什么 |
|---|---|---|
| #4121 | open | `--batch-groups` 的流水线组里带 MTP 草稿（`STRATA_PIPED_MTP=N`，opt-in） |
| #4199 | open | dsv41 的批服务，16 槽下聚合 2× |
| #3757 / #3783 | open | Metal 后端的批槽 |
| #2526 | open | 槽复用前保住已解码的批槽会话 |
| #3382 | open | `--batch` 下的 prefill CPU 分担（opt-in，短读 −21% ~ −31%） |

所以训练侧做的是**对接**，不是实现：采集脚本不再自己定一个并发数，而是**问引擎**——读
`/v1/status` 的 `concurrency`（就是配置里 `"parallel": N` 声明的批槽数），按它发这么多条在飞。
`--parallel 0`（默认）= 自动；给数字 = 覆盖。引擎没开批槽时自动是 1，串行发，也不排队。

```
python collect_rollouts.py --engine http://127.0.0.1:8095 --prompts prompts.jsonl     --group 4 --parallel 0 --out rollouts.jsonl
```

组结构本身适合它：同组 K 条共享 prompt 前缀、彼此无依赖，是天然的一批并发请求；而采样方是
已经常驻的引擎，所以这一段并行**不额外要显存**。

**实测（125B，一张 4090，`--batch 4`，28 条请求 × 最长 256 token，交替跑 串行/auto/并行/串行）**

| 在飞请求数 | 墙钟 | 生成 token | 聚合 tok/s |
|---|---:|---:|---:|
| 1（串行） | 8.4 s | 960 | 118.4 |
| auto（引擎说 4） | 9.3 s | 1208 | 134.6 |
| 4 | 8.8 s | 1042 | 122.4 |
| 1（串行，再跑一遍） | 7.6 s | 885 | 120.8 |

结论：**没有可测的吞吐收益**。并行确实让请求重叠了（逐请求耗时之和 26.5 s vs 墙钟 6.7 s，
重叠 3.8×），但聚合吞吐落在串行两次的区间里（118.4 / 120.8），差异是噪声。原因是日志里
明摆着的两条：批路径**没有草稿**（要 `--batch-mtp`，opt-in，且只有两级测过），而且引擎自己
在警告「只有 133–324 MiB 显存空闲，卡这么满时驱动会把显存换页到内存」——批槽要的 verify
窗口缓冲区根本没地方放。

**两条不能当并行用的东西**（都踩过）：

- **批槽不等于训练并行**：学生要 26–34 GiB，引擎常驻 ~40 GiB，一张 48GB 卡放不下两个，
  所以采集与训练必须分两段（采集 → 关引擎 → `--rollouts` 训）。
- **`--batch` 下拿不到 token id**：引擎的 LP 行格式是 `LP <logprob> [id:logprob ...]`，
  而 `id:logprob` 只在 K>0 或给了 allowed 时才打印（`emit_lp_line`，k<=0 且 allowed 为空就
  只有 logprob）。采集脚本请求的是 `top_logprobs: 0`，于是 `tokens_exact=False`、只拿到文本，
  训练端退回本地重编码 + 本地重算 old logprob。要 ids 就把 `top_logprobs` 提到 1——但那时打印的是
  top-1 的 id，采样走温度时不保证等于采样到的 id，**这条路今天不通**，得等上游把带 slot 的 LP 行做出来。

**想要「不碰引擎就能并行」，目前只有多副本**（多个引擎进程，`serve/replicas.py`）；那是几份
常驻，不是一份变并行。

## 15. WSL2 与 Windows：同源引擎实测对比

一直到这次之前，两边都没法比——`FetchContent` 去 GitHub 克隆 `llama.cpp` 一直卡在
`_deps/strata_llamacpp-src` 只有 11 MB（完整要 600 MB+）。绕过去的办法不用联网：
`-DSTRATA_GGML_DIR=<已有的完整 llama.cpp>`。上游那份在
`Strata-train/build-train/_deps/strata_llamacpp-src`，正好是 `third_party/ggml/VERSION.txt`
pin 的 `3cf03257f219`，633 MB，3,636 个文件。

于是两边用**同一份源码、同一组 CMake 选项、同一张卡、同一组推理参数**各构建一次：

```
-G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR=<pin 的 llama.cpp> -DCMAKE_CUDA_ARCHITECTURES=89 --target strata
```

两边都是 153/153 步，引擎都自报 `engine 0.1.44.1`、同一 arch 89、expert cache 也落在同一档
（Windows 24,473 slots / 39.80 GiB，WSL 24,440 slots / 39.75 GiB）——配置确实对齐了，剩下的差
就是平台。

推理参数（两侧逐字相同，只换了路径）：`--pack --native --ple-gguf --expert-profile
--mtp --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --max-context 131072
--kv int8 --kv-resident 32768 --vram-reserve-mib 700 --pool-workers 15`。
每次跑四条：预热 64 token、400 token ×2、3107 token prompt 生成 16 token。

**结果（各三轮，取中位；括号是区间）**

| 指标 | Windows | WSL2 | WSL/Win |
|---|---:|---:|---:|
| 模型加载（到 READY） | **21.0 s**（20.5–22.0） | 133.5 s（93.2–134.5） | **6.4× 慢** |
| 解码 400 token，第 1 次 | 157.0 tok/s（155.8–157.7） | 149.4 tok/s（132.6–153.7） | 0.95× |
| 解码 400 token，第 2 次 | 148.9 tok/s（146.6–152.6） | 151.4 tok/s（140.4–151.5） | 1.02× |
| 长 prompt 3107 token | **3048.8 tok/s**（3016–3149） | 1852.9 tok/s（1174–2649） | **0.61×** |

**两条结论**：

- **解码速度两边一样**（差 ±5%，落在串行两次的噪声里）。推理主循环是 GPU 上的，
  WDDM 与 `/dev/nvidia` 在这条路径上没造成可测差别。
- **加载和 prefill 是两回事**：加载慢 6.4×、prefill 慢 1.6×，两段都主要是**读盘**，
  不是 CUDA。

**加载差距的原因（试过一条，否掉了）**：引擎在 Windows 上打
`expert arena read unbuffered`，在 Linux 上打 `expert arena read through the file cache
(buffered (not Windows))`——同一个函数按平台分的支。76 GB 文件、可用内存只剩 12.1 GiB 时，
buffered 路径要跟 RAM 副本抢内存、反复换页。于是用 `STRATA_UNBUFFERED_LOAD=1` 强制走
unbuffered 做了 A/B（交替三轮，抵消缓存漂移）：

| `STRATA_UNBUFFERED_LOAD` | 加载中位 |
|---|---:|
| 0（buffered，默认） | 152.4 s |
| 1（强制 unbuffered） | 155.7 s |

**没有改善**，所以这个开关不是原因。更可能是 WSL 的 ext4 建在 VHDX 上（多一层）、
以及 58 GiB 的可用内存装不下 76 GB 文件。这两条**还没验证到能下结论**，
先别当成结论用。

**因此给训练流程的一条实际建议**：训练只能 WSL，而每次启动都要多付 1.5–2 分钟加载
（95–135 s vs 21 s）。这对「跑 40 步、6 分钟训完」的短任务很致命——
加载占了整个 wall clock 的一半以上。步数少时，宁愿一次多训几步，
也别反复重启引擎。

复现脚本与两侧配置在 `.workbuddy/wslcmp/`
（`bench_cmp.py`、`cfg-win.json`、`cfg-wsl.json`、`build-cmp-win.bat`、`build-cmp-wsl.sh`）。
