# local/ —— 本机实验用的改动，**不碰共享源码**

> ## ⏸ 已搁置（2026-10-12）
>
> 决定**不在 Strata 这条路上做** —— 改用**本地 vLLM 27B**：
> vLLM 的 OpenAI 接口**开箱就给 `logprobs` / `top_logprobs`**，连 **tool call 的参数**都给
> （实测 `play` 的 `action` 值 `'3'` 有 logprob `-0.9033`），所以**不需要动任何引擎**。
> 而 Strata 那条路要么依赖 `STRATA_LOGPOS`（进程级、写文件、只在 prefill、**还会关掉并行
> prefill** ⇒ 每轮重新预填充，性能代价大），要么就得打下面的引擎补丁再重编译。
>
> 另外：`arc_agi3` 的 VISTA harness 默认后端是 `inprocess`（自写的 `openai_transport.py`，
> 直接 POST `/v1/chat/completions`）—— **本地和 Kaggle 是同一条代码路径**，
> 换服务端即可，所以本地用 vLLM 验证出来的东西能直接搬到 Kaggle。
>
> **这些文件保留着**（独立文件、零副作用，`src/` 已 revert 干净），
> 哪天要在 Strata 上做分布类实验可以直接用。

> ## ✅ 2026-10-12 晚更新：这活儿又启动了，但改成「完全不动原文件」
>
> 用户要 logprobs 来驱动 AnyJev。已按「**新文件 + 生成副本**」重做，见
> **[`logprobs/`](logprobs/README.md)**：
> 引擎侧不再就地改 `src/program/generate.cpp`（原文件 sha256 不变），而是生成补丁副本编
> 新二进制 `strata-lp`；server 侧与 AnyJev 后端都是新增文件。
> 25/25 自检过；**只差编译** —— 本机没有 cmake/MSVC，编译要在有完整工具链的机器上做。
> 顺带量了收益预期：Strata 基线上 play 请求 0 次被拒、40% token 烧在空想回合 ⇒
> 除非用 AnyJev 的 stopping 校准去砍空想，否则重建引擎换 logprobs 收益接近 0。
>
> 本目录下的 `apply_engine_logprobs.py` / `server_logprobs.py` 是**就地版**（会写
> `src/`），已被 `logprobs/` 取代，留着只为对照。

这个目录放「本机要用、但不想污染上游」的改动。每个改动都是**独立文件**，可随时删掉或还原。

~~目前只有一个：**让 native(CUDA) 引擎的 Strata 支持 OpenAI 的 `logprobs` / `top_logprobs`**。~~
现在是两个：**就地版**（`apply_engine_logprobs.py`，已搁置）和**生成版**（`logprobs/`，当前推荐）。

> ⚠️ **Kaggle 那条线用不到它** —— Kaggle 上跑的是 SGLang/vLLM，不经过 Strata。
> 这是本地「看模型分布」的诊断能力。

---

## 为什么需要

`/v1/chat/completions` 的 `logprobs` 目前只有 **SYCL(Intel)** 端口支持：

| 端口 | 引擎吐 `LP` 行 | Server 层转成 OpenAI `logprobs` |
|---|---|---|
| SYCL（Intel） | ✅ `sycl/src/program/generate.cpp:9244` | ✅ `sycl/serve/server_intel.py::install_logprobs()` |
| **native（CUDA，本机在用）** | ❌ 只有 `T <id>` | ❌ `serve/server.py` 里搜不到 `logprob` |

所以分两层补齐，两层都是独立文件。

---

## 两个文件

### 1. `apply_engine_logprobs.py` —— 引擎侧（要重编译）

给 `src/program/generate.cpp` 加 **5 处锚点替换**：

| # | 位置 | 内容 |
|---|---|---|
| 1 | 顶部匿名 namespace | `local_emit_lp()` —— 从一个窗口行的 head logits 算 logprob + top-K |
| 2 | GEN 参数声明处 | `int req_logprobs = -1;` |
| 3 | GEN 参数解析处 | `else if (key == "logprobs") …` |
| 4 | speculative decode 的 `T` 行后 | `local_emit_lp(ver, i, outp[i], req_logprobs);` |
| 5 | plain decode 的 `T` 行后 | `local_emit_lp(ver, i, outv[i], req_logprobs);` |

**改动为什么小**：native 引擎**已经有** `Verifier::copy_logits(int t, float* host)`
（`include/strata/core/verify.hpp:86`，`src/core/verify.cpp:2878`）—— 和 SYCL 用的是同一个接口，
而且 `src/program/generate.cpp:10696/11819` 已经在用它 dump logits 到文件。
差的只是「解析 `logprobs=K`」和「在两个 `T` 行后打出 `LP` 行」。

```bash
python local/apply_engine_logprobs.py --check     # 只看状态（不写盘）
python local/apply_engine_logprobs.py --apply     # 打补丁（幂等，留 .localorig 备份）
python local/apply_engine_logprobs.py --revert    # 还原
# 然后重编译引擎（见 docs/INSTALL.md）
```

**设计要点**：
- 锚点对不上就**硬失败并报出是哪一处**，绝不静默改错 —— 上游改了那几行你能立刻知道。
- `--apply` 幂等；首次会把原文件**字节副本**存到 `local/generate.cpp.localorig`
  （放 `local/` 而不是 `src/` 旁边，免得 `src/` 里多出一个看着像「源码被改过」的文件）。
- 源码里每处都插了 `// local/apply_engine_logprobs.py` 标记，方便 `grep` 和 `--revert`。
- 打完再 `--revert`，`git diff src/program/generate.cpp` 应为**空**（逐字节还原）。

### 2. `server_logprobs.py` —— Server 侧（纯 Python，不用重编译）

`sycl/serve/server_intel.py::install_logprobs()` 的 CUDA 版：从外部 monkey-patch
`serve/server.py` 的 5 个扩展点，**一行都不改共享文件**（沿用 Strata 自己的做法 ——
*"Nothing in `serve/` is edited … so upstream merges stay clean"*）。

```bash
python local/server_logprobs.py --engine strata --config strata-<model>.json --port 8080 [...]
python local/server_logprobs.py --check-engine     # 只看引擎补丁状态，不起服务
```

参数与 `serve/server.py` 完全一致（原样透传）。

**设计要点**：
- 引擎没打补丁时**照样能起服务**，但请求里带 `logprobs` 时会打一条明确的警告
  （区分「一条 `LP` 都没有 = 没打补丁」与「`LP` 全是 `nan` = `copy_logits` 失败」），
  而不是静默返回空 —— 空的 `logprobs` 很容易被误读成「模型很自信」。

---

## 自检

```bash
.venv/Scripts/python.exe local/selftest_logprobs.py
```

**不需要 GPU / 引擎 / 模型**：在真实源码上核对锚点唯一性，并用桩把
`_pump` 的 `LP`↔`T` 对齐、`openai_chunks` 的合并/透传/失败分支全跑一遍。

---

## 改动边界（重要）

- **只加文件，不改共享源码**：`serve/server.py`、`serve/responses.py` 都不动。
- 引擎侧唯一会动的是 `src/program/generate.cpp`，而且是**可 `--revert` 的锚点补丁**；
  `--revert` 后 `grep -c "local/apply_engine_logprobs.py" src/program/generate.cpp` 应为 `0`，
  且 `git diff src/program/generate.cpp` 应为空。
- ⚠️ 自检与服务端包装**必须用 Strata 自己的 venv**（`serve/server.py` 依赖 `jinja2`）：
  `.venv/Scripts/python.exe local/selftest_logprobs.py`。用系统 `python` 会
  `ModuleNotFoundError: jinja2`。
- `local/generate.cpp.localorig` 是首次 `--apply` 的字节备份（`--revert` 用它还原）。

---

## 什么时候才值得做引擎那一步

**不要为了「防异常动作」做它** —— 实测 `play` 的异常率只有 ~1%，
而 `pi.stderr` 四个文件全是 0 字节（**没有任何格式/解析错误**），
也就是 AnyJev 最擅长防的那一类**本来就不存在**。详见
`arc_agi3/CURRENT_TASK.md` §5.6。

它值得做的理由是**诊断与置信度**：

- 模型对某个动作有多确定（`logprob` 绝对值）；
- `top_logprobs` 里有没有它没选的更优动作；
- 「为什么它选了这个动作 / 为什么它卡住不收敛」。

注意 `logprobs` 是 **verify window 的 logits、采样前**的值 —— 不受 temperature / penalty 影响，
是模型的**原始分布**。
