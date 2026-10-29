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
比 ckpt 模型预测的 2.7× 多出约 1.6 个前向当量（≈7 s）。**这 7 s 还没拆** —— 它和 §7.6 ⑤
是同一件事，只是稳态前向变便宜之后才显出来。

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
| ③ | ctx 只存专家 id、backward 按 id 重流 | 现在 grad 下每专家独占一块显存（1.27 GiB/层），仍必须开 ckpt | 去掉 ckpt ⇒ 省掉那次重算前向 |
| ④ | B>1 | 实测 B=8 是 B=1 的 1.32× | 白拿 +32% |
| ⑤ | 拆 18.9 s 与 4.3× 前向之间的差额 | ckpt 模型给 2.7×，实测 4.3× | 还有 ~1.6 个前向当量（≈7 s）没归因 |

③ 按稳态算：18.9 − 4.4（省掉的重算前向）≈ **14.5 s/步 ≈ 141 tok/s**，顺带省 3.9 GiB；
⑤ 是这条账里唯一还没拆的部分 —— 它可能就藏在 ③ 里（重算前向带着 grad 跑，比 no_grad
的前向贵），所以做 ③ 之前先量 ⑤ 更省事。

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
| `_det_*.py` | 指纹法定位数值差异（见 §5 第一轮第 3 条） |
