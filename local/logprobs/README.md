# local/logprobs —— 给 Strata 加 OpenAI logprobs（给 AnyJev 用），**不动任何原文件**

> ## 先读这段：**为什么必须重新编译，以及为什么你大概不该走这条路**
>
> **为什么必须编译**：Strata 是**两个进程**。`serve/server.py`（Python）只是前端，
> 把请求通过 stdin 写给引擎、再把引擎 stdout 的 `T <id>` 行转成 OpenAI 响应；
> 真正跑模型的是 `engine/strata.exe`（164 MB 的 C++/CUDA **编译产物**）——
> **logits 只在这个二进制里**，Python 层从来没见过。而引擎的 `--serve` 协议是纯文本
> `key=value`，代码里明写 **"unknown keys are skipped"** ⇒ 协议里没有「把概率给我」
> 这个键，只能改 C++。唯一现成出口 `STRATA_LOGPOS` 是 prefill 期教师强制测量、
> 写文件、进程级、还会强制串行慢路径 ⇒ 不能在线用。
> ⇒ **「Strata 引擎出 logprobs」= 改 C++ = 必须重新编译**，没有第二条路。
>
> **为什么大概不该走**：概率谁都会算，差别只在「暴不暴露」。**vLLM 开箱就暴露**
> （`logprobs: true` 一个字段，连 tool call 参数都给，实测 `'3'` 的 logprob `-0.9033`），
> SGLang 同理。而 AnyJev 的契约只是**一个方法**，它的 `anyjev/backends/vllm.py`
> **本来就能直接用**。所以正确做法是**换引擎**（vLLM/SGLang），不是给 Strata 做手术。
> 本目录的产物留给「将来在 Strata 上做分布类实验」用，**不在关键路径上**。

## 三块拼图

| 文件 | 层 | 原文件是否被改 |
|---|---|---|
| `build_engine.py` + `build/CMakeLists.txt` | C++ 引擎 | ❌ 不改（`src/program/generate.cpp` 是**只读**的，脚本把补丁后的**副本**生成到 `build/generate_logprobs.cpp`，编成一个**新二进制** `strata-lp`） |
| `server_logprobs.py` | Python server | ❌ 不改（从外部 patch `serve/server.py` 的扩展点 —— Strata 自己的 `sycl/serve/server_intel.py::install_logprobs()` 就是这个套路） |
| `AnyJev/anyjev/backends/strata.py` | AnyJev | ❌ 新增文件（后端契约「一个方法一个文件」，ROADMAP 里本来就在征稿） |

## 数据流

    AnyJev(StrataBackend)                     # allowed_token_ids=[7 个动作 token]
      → POST /v1/chat/completions  max_tokens=1, logprobs=true, top_logprobs=7
      → server_logprobs.py 译成引擎键  logprobs=7 allowed=11,22,33
      → strata-lp 引擎在每个 `T <id>` 后多打一行 `LP <lp> <id>:<lp> …`（只列点名 id）
      → server_logprobs.py 把 LP 行接成 OpenAI 形状，并在 allowed 集合上**重新归一化**
      → AnyJev 拿到「候选动作上的概率分布」

`LP` 行格式与 SYCL 端口逐字一致，所以 server 侧同一个读者两个引擎都能读。

## 为什么引擎也要一份「New file」

`src/` 是共享源码，直接改会和上游更新冲突。所以引擎侧不是就地补丁，而是
**读取原文件 → 打 6 处锚点补丁 → 写出副本**；原文件始终只读（自检里有 sha256 断言）。
锚点对不上 / 不唯一 / 原文件已有补丁痕迹 ⇒ **硬失败，一个字节都不写**。

## 现在的状态（2026-10-12）

| 项 | 状态 |
|---|---|
| server 侧翻译 + LP 解析 + 归一化 | ✅ 完成，25/25 自检过（`selftest.py`，不需 GPU） |
| 引擎补丁生成器（原文件只读、幂等） | ✅ 完成，同一自检覆盖（原文件 sha256 不变） |
| AnyJev 后端（含「没回 logprobs 就显式报错」） | ✅ 完成，同一自检覆盖 |
| **编出 `strata-lp`** | ⛔ **本机做不了：没有 cmake、没有 MSVC**（Program Files 下只有 VS 安装器残壳；CUDA v11.8~v13.3 在但没编译器）。要在有完整工具链的机器上 `cmake --build … --target strata-lp` |

编译只需三步（在有工具链的机器上）：

    python local/logprobs/build_engine.py --emit        # 生成补丁副本
    # 根 CMakeLists.txt 末尾加一行：add_subdirectory(local/logprobs/build EXCLUDE_FROM_ALL)
    cmake -S . -B build-lp -G "Visual Studio 17 2022" -A x64 && cmake --build build-lp --target strata-lp --config Release

起服务（引擎路径指向新二进制）：

    python local/logprobs/server_logprobs.py --config strata-<model>.json --engine <build-lp>/strata-lp.exe --port 8080

## 自检

    python local/logprobs/selftest.py      # 25 项，不需要 GPU / serve.server

## ⚠️ 收益预期（2026-10-12 实测数据，别为它重建引擎）

Strata/Swift-Flash-Next 基线：ar25 过 1 关烧 146,386 token，lf52 **0 关**烧 247,027 token。
其中 **play 请求 13/13 全部通过校验、0 次被拒**，而 **40% 的 token 烧在「无工具调用的
空想回合」**（98,304 / 247,027）。

⇒ AnyJev 治的是「动作格式/异常」，而我们**没有这个病**；
   病在推理循环。除非打算用 AnyJev 的 `calibrate/stopping.py`（靠分布早停）去砍那 40%，
   否则先重建引擎换 logprobs 的收益接近 0。**这条已写进 CURRENT_TASK.md §2.9 的结论里。**
