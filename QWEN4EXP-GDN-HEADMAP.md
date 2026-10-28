# Qwen3.8-Flash-Next：GDN 的 key-head 配对约定（tile，不是 repeat_interleave）

2026-10-29 实测。结论：**Strata 引擎是对的，HF transformers 5.17 的 `qwen4_exp`
实现与这份权重不符**。训练侧必须与引擎一致，否则 LoRA 训在另一条函数上。

## 约定差

GDN 的 48 个值头（HV=48）配 16 个键头（HK=16），两种配对：

| | 值头 j 用的键头 | 出处 |
|---|---|---|
| tile | `j % HK`（= j % 16），值头分组 {j, j+16, j+32} | Strata 引擎 `kernels.cu:489` 等 16 处 `qh = head % HK` |
| interleave | `j // (HV/HK)`（= j // 3），值头分组 {3h, 3h+1, 3h+2} | HF `modeling_qwen4_exp.py` `repeat_interleave(HV//HK, dim=-2)`；上游 llama.cpp `qwen3next.cpp` 同 |

两者**不是行置换关系**（值头分组方式不同），所以改 GGUF 的 q/k 行序补救不了 ——
只能改配对逻辑。

## 判据（同一份 GGUF、同一进程内切换，dbg_tile_fwd.py / dbg_tile_test.py）

| | tile (`j%16`) | interleave (`j//3`) |
|---|---|---|
| CE(' Paris' \| 'The capital of France is') | **0.474** | 1.834 |
| CE(' 8' \| '1 2 3 4 5 6 7') | **0.030** | 0.988 |
| CE('，' \| '我是人工智能助手') | **1.056** | 10.652 |
| mean CE | **0.520** | 4.491 |
| vs 引擎 logits pearson（248320 维） | **0.9841** | 0.44 |

tile 下模型几乎确定地预测 " 8"（CE 0.03），中文标点续写也回到 1.06；interleave 下
中文那句 10.65 已接近均匀分布（ln 248320 = 12.42）。CE 判据只用了 PyTorch + 权重，
不依赖引擎。

排除项：config 传错（`ssm.group_count=16`/`ssm.time_step_rank=48`，且
`2*16*128+48*128=10240` 与 q|k|v 宽度自洽）；GGUF 值头被置换（τ 假设 4 变体扫描，
cos 均 0.47 << 1）；token 对齐、conv 语义、GDN 递推数学（此前逐项验过）。

## 修复

`examples/training/loader_qwen4.py` 的 `install_gdn_tile_mapping()`：patch 模块级
`torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule`，把入参 q/k
（`(B,T,HV,D)`，interleave 约定）按 `(arange(HV) % HK) * (HV//HK)` 重排行；v 不动
（值头索引两边一致）。`STRATA_GDN_TILE=0` 可关闭做 A/B。

同一批修复还含另外两项（都是前向乱码的根因）：
1. `Qwen4ExpTextRMSNorm` 的 weight 是**增量**约定（有效 scale = 1 + w），GGUF 存完整
   gamma ⇒ 8 类后缀装载时 −1。
2. GDN 输出门：引擎用 `sigmoid(z)`（`kernels.cu:528`），config 默认回落 `silu` ⇒
   `output_gate_type="sigmoid"`（q4_map_check.py）。

## 引擎侧

**不需要改**。16 处 `% HK` 保持原样（kernels.cu ×9、gdn_chunk.cu ×3、
gdn_rec_parity.cu ×2、gdn_chunk_parity.cpp、q36ref.cu）。

## 可选：给 transformers 上游报 issue

标题建议：`Qwen4Exp GatedDeltaNet expands q/k with repeat_interleave, but
Qwen3.8-Flash-Next weights need tile (v_head % num_k_heads)`

要点：
- 现象：官方 `repeat_interleave(HV//HK)` 下 Qwen3.8-Flash-Next 的续写近乎随机
  （CE('，') = 10.65，均匀 12.42）；改成 `j % HK` 后降到 1.06，CE(' 8') 从 0.99 降到 0.03。
- 复现：加载官方权重，把 GDN forward 里 q/k 的 head 展开换成
  `k[..., (arange(HV) % HK) * (HV//HK), :]`，对比两条前缀的 CE。
- 备注：本结论在 IQ3_XXS 反量化权重上取得，量化噪声不足以解释 10.65 vs 1.06 的差；
  若有官方 bf16 权重建议直接复核一遍。
- 不确定处：Qwen3-Next（qwen3_next）是否也是 tile —— 未测，不应在 issue 里外推。
