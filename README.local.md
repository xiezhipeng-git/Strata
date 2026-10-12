# Strata — 本地 fork：给引擎加 OpenAI 兼容的 `logprobs`（给 AnyJev 用）

**English TL;DR** — A fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) that adds a
**per-request `logprobs` / `top_logprobs`** endpoint to the Strata engine, so a client such as
**AnyJev** can read the model's probability distribution over candidate tokens. Nothing under
`src/` is modified: the patched engine source is *generated* as a copy and built as a separate
binary (`strata-lp`). Upstream stays pristine and `git rebase upstream/main` keeps working.

---

## 这个 fork 是干什么的

**只做一件事：让 Strata 的 OpenAI 兼容接口能返回 `logprobs`（候选 token 上的概率分布）。**

用途是**对接大模型的 OpenAI API 的 logprobs 功能**，具体两类消费方：

1. **AnyJev** —— 它的契约只有一个方法 `next_token_logprobs(prompts, token_ids)`，
   要的就是「给定若干候选 token，它们在下一个位置的概率」。
2. **基于分布的早停 / 置信度** —— 例如 `AnyJev/anyjev/calibrate/stopping.py` 那类
   「靠概率分布判断模型已经想够了」的手段，用来砍掉长尾的无效推理轮。

## 为什么必须改引擎（而不是只改 server）

Strata 是**两个进程**：

```
serve/server.py     ← Python 前端，只转发（把请求写成 key=value 给引擎，把引擎的 T <id> 行转回 OpenAI 响应）
engine/strata.exe   ← 164 MB 的 C++/CUDA 编译产物，**logits 只在这里**
```

引擎的 `--serve` 协议是纯文本 `key=value`，源码里明写 **"unknown keys are skipped"**
⇒ 协议里根本没有「把概率给我」这个键，Python 层永远见不到 logits。
唯一现成的出口 `STRATA_LOGPOS` 是 **prefill 期教师强制测量、写文件、进程级**，
还会强制走串行慢路径 ⇒ **不能用于在线决策**。

⇒ 想在 Strata 上做「分布类」的事，**只能改 C++ 引擎并重新编译**。
（如果你不需要 125.7B，直接用 vLLM / SGLang 更省事 —— 它们开箱就暴露 logprobs。
这个 fork 的意义在于：**让 vLLM 装不下的那个 125.7B MoE 也能提供分布**。）

## 改了什么（三层，**都不碰上游文件**）

| 层 | 文件 | 做法 |
|---|---|---|
| 引擎 C++ | `local/logprobs/build_engine.py` | 读**只读**的 `src/program/generate.cpp` → 打 **6 处锚点补丁** → 写出**副本** `local/logprobs/build/generate_logprobs.cpp`。原文件带 **sha256 断言**；锚点不唯一 / 已有补丁痕迹 ⇒ 硬失败，一个字节都不写 |
| 构建 | `local/logprobs/build/CMakeLists.txt` | 新目标 **`strata-lp`**，源码换成上面那个副本。接入方式是在**仓库根 `CMakeLists.txt` 末尾加一行**：`add_subdirectory(local/logprobs/build EXCLUDE_FROM_ALL)`（这是相对上游**唯一**的原文件改动，删掉即还原；必须放末尾，因为目标要继承父作用域的 `_strata_gpu_include_directories` / `strata_engine` / `strata_prefill` / `strata_spec` / `_strata_gpu_runtime_target`） |
| Python server | `local/logprobs/server_logprobs.py` | 从**外部** monkey-patch `serve/server.py`，把 OpenAI 的 `logprobs` / `top_logprobs` / `allowed_token_ids` 译成引擎键，并把引擎多打的 `LP` 行接回 OpenAI 形状、在 allowed 集合上重新归一化。沿用 Strata 自己的做法（`sycl/serve/server_intel.py::install_logprobs()` 就是这个套路，原则是 *"Nothing in `serve/` is edited"*） |

数据流：

```
AnyJev（或任何客户端）
  → POST /v1/chat/completions   {max_tokens:1, logprobs:true, top_logprobs:K, allowed_token_ids:[…]}
  → server_logprobs.py          译成引擎键：logprobs=K  allowed=11,22,33
  → strata-lp 引擎              在每个 `T <id>` 行后多打一行 `LP <lp> <id>:<lp> …`（只列点名的 id）
  → server_logprobs.py          接成 OpenAI 形状，并在 allowed 集合上重新归一化
  → 客户端拿到「候选 token 上的概率分布」
```

`LP` 行的格式与 Strata 的 SYCL 端口**逐字一致**，所以 server 侧同一个读者两个引擎都能读。

## 构建

### Windows（**已实测通过**）

双击 **`build-strata-lp.bat`**。它会自己检查前置、跑 `build_engine.py --emit`、幂等补 CMakeLists
那一行、搭好 MSVC 环境并编译。

前置两样：

| 前置 | 本机状态 |
|---|---|
| **VS 2022 Build Tools + C++ 工作负载**（脚本用 `vswhere` 找；它自带 cmake + ninja，不用另装） | ✅ 已装（MSVC 14.41.34120） |
| **CUDA Toolkit（`nvcc.exe`）** | ✅ **已有，在 `D:\my\env\cuda_tool`（CUDA 13.3，4.3 GB），`CUDA_PATH` 已指向它** |

> ⚠️ 别被骗了：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8 … v13.3` 这些目录
> **是空壳**（只有 `extras`）。真正可用的 toolkit 在别处 —— **先看 `CUDA_PATH`**。
> `build-strata-lp.bat` 的查找顺序是：`CUDA_PATH` → `C:\Program Files\NVIDIA GPU Computing Toolkit` → `C:\CUDA` → `D:\CUDA`。

**不调 `vcvars64.bat`**：它要用 `reg.exe` 查注册表，沙箱/加固机器会拦。
`build-win-env.bat` 直接从文件系统读出 MSVC 与 Windows SDK 的路径再设 `INCLUDE`/`LIB`。

实测结果（Windows 11 / MSVC 14.41.34123 + Windows SDK 10.0.26100.0 / CUDA 13.3.73 / RTX 4090 sm_89）：

```
[18/18] Linking CXX executable local\logprobs\build\strata-lp.exe   ⇒ [OK] 
```

- 产物 **49.6 MB**（`build-lp\local\logprobs\build\strata-lp.exe`；比上游 `engine/strata.exe` 的
  156.8 MB 小，因为只编了 sm_89，没有做成多架构的 fat binary）
- `--version` → **`strata 0.1.41`**，与上游预编译引擎同版本 ⇒ 协议兼容
- 二进制里能查到 `LP ` 与 `window_logprobs` ⇒ 补丁确实进去了

### Linux / WSL（**已实测通过**）

```bash
python local/logprobs/build_engine.py --emit
# 仓库根 CMakeLists.txt 末尾加：add_subdirectory(local/logprobs/build EXCLUDE_FROM_ALL)
cmake -S . -B build-lp -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
cmake --build build-lp --target strata-lp -j8
```

> `-DSTRATA_GGML_DIR` 指向仓库里已有的 `third_party/llama.cpp` ⇒ **完全不需要联网**
> （否则 CMake 会去 GitHub 拉 llama.cpp）。

实测环境与结果：Ubuntu 24.04 / CUDA 13.2 / RTX 4090（sm_89）/ cmake 4.3.4 / ninja 1.13.2
⇒ **135/135 通过**，产出 `build-lp/local/logprobs/build/strata-lp`（**57 MB，ELF x86-64**），
`--version` → **`strata 0.1.41`**，链接 CUDA 13.2 的 cudart/cublas。

> ⚠️ WSL 这条路**跑不了这个 125B**：WSL 默认内存上限是宿主的 50%（本机 31.2 GB），
> 而 IQ3_XXS 需要 **60 GB**；低内存模式（专家从 SSD 流式读）只有 **7-8.5 tok/s**。
> 所以 Linux 二进制只是"编译验证"，实际部署用 Windows 那个。

## 起服务

`local/logprobs/server_logprobs.py` 复用 **Strata 自己的 server**（`serve/server.py` 一行不改），
所以参数是 Strata 的：`--engine strata --config <配置> --port 8080`。配置里的 `exe` 指向哪个引擎二进制，
就用哪个。

**带 logprobs 的引擎**（仓库里已备好 `strata-lp-swift-iq3_xxs.json`，`exe` 指向 `strata-lp.exe`）：

```bat
.venv\Scripts\python.exe local\logprobs\server_logprobs.py ^
    --engine strata --config strata-lp-swift-iq3_xxs.json --port 8080
```

**原版引擎**（`engine\strata.exe`，没有 logprobs）—— 也就是桌面启动的那条路子：

```bat
.venv\Scripts\python.exe local\logprobs\server_logprobs.py ^
    --engine strata --config strata-swift-iq3_xxs.json --port 8080
```

自检一下补丁副本是不是在：`.venv\Scripts\python.exe local\logprobs\server_logprobs.py --check-engine`


## 自检（**不需要 GPU / 引擎 / 模型**）

```bash
python local/logprobs/selftest.py      # 25 项，全过
python local/selftest_logprobs.py      # 更早一版的自检，也全过
```

覆盖：补丁生成器在**真实源码**上核对 6 处锚点唯一且原文件 sha256 不变、重复 emit 逐字节幂等、
server 侧的 OpenAI `logprobs` 语义（缺字段 / 只有 nan / stop token 的处理）、
以及 AnyJev 后端「没拿到 logprobs 就显式报错并点名 strata-lp」。
实测：`local/logprobs/selftest.py` **25 项**全过。

## 与上游同步

```bash
git fetch upstream
git rebase upstream/main        # 只有 CMakeLists.txt 末尾那 1 行可能需要解冲突
git push origin main
```

远端约定：**`origin` = 这个 fork**（`xiezhipeng-git/Strata`），**`upstream` = `Niko1221/Strata`**。

## 目录说明

- **`local/logprobs/`** —— **当前在用的这一套**（`build_engine.py` 生成器 + `server_logprobs.py` + `selftest.py` + `build/` 产物）
- `local/README.md`、`local/apply_engine_logprobs.py`、`local/server_logprobs.py`、
  `local/selftest_logprobs.py`、`local/generate.cpp.localorig`
  —— **更早的一版尝试**（已搁置，保留作参考；那一版是就地补丁 `src/`，已被上面这套「生成副本」取代）
- `build-strata-lp.bat` —— Windows 构建脚本
- `START-HERE-modelscope.bat` / `download_swift_model.py` / `strata_125b.bat`
  —— 走 ModelScope 拉 125.7B（Swift 1.5 IQ3_XXS MoE）的启动脚本
  （HuggingFace 在本机不可达）

## 收益预期（说清楚，别误期待）

Strata/Swift-Flash-Next 的基线实测：ar25 过 1 关烧 146,386 token，lf52 **0 关**烧 247,027 token；
其中 **play 请求 13/13 全部通过校验、0 次被拒**，而 **40% 的 token 烧在「无工具调用的空想回合」**。

⇒ 如果你的目的是**「防异常 / 规范动作格式」**，这条路收益接近 0（**我们没有这个病**）。
它值得做的理由是另外两个：**置信度/分布诊断**，以及 **`calibrate/stopping.py` 那种靠分布早停**
去砍那 40% 的空转轮。
