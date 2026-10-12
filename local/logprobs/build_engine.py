"""把 native(CUDA) 引擎打成 **per-request logprobs** 版 —— 生成一个新二进制，**原文件一个字节都不动**。

## 为什么不是「就地打补丁」

`local/apply_engine_logprobs.py`（就地补丁版）会写 `src/program/generate.cpp`。
那违反了本项目对上游源码的纪律（`src/` 是共享源码，直接改会和上游更新冲突），
所以这里改成**生成式**：

    src/program/generate.cpp   ← 原文件，**只读**，永不修改
              │  python local/logprobs/build_engine.py --emit
              ▼
    local/logprobs/build/generate_logprobs.cpp   ← 新文件（补丁后的副本）
              │  CMake: add_executable(strata-lp ...)（见 emit 出来的 CMakeLists 片段）
              ▼
    strata-lp.exe              ← 新二进制

原文件不被写；补丁内容全在本文件里，锚点对不上就**硬失败并报出是哪个锚点**。

## 加的协议（向后兼容）

引擎的 `--serve` 协议本来就是 `key=value ... <ids>`，且**unknown keys are skipped**，
所以新键对旧引擎是透明的：

    logprobs=K      每个 `T <id>` 行后多打一行 `LP <logprob> <id>:<logprob> ...`（前 K 个）
    allowed=a,b,c   只打这些 token id 的 logprob（配合 logprobs 用；为空则退回 top-K）

`LP` 行的格式与 SYCL 端口逐字一致（`sycl/src/program/generate.cpp`），
所以 server 侧同一个读者能同时读两个引擎。

## 用法

    python local/logprobs/build_engine.py --check     # 只看原文件锚点是否还对得上（不写盘）
    python local/logprobs/build_engine.py --emit      # 生成补丁副本 + CMake 片段
    python local/logprobs/build_engine.py --clean     # 删掉生成目录

打完副本后：把 emit 出来的 `add_executable(strata-lp ...)` 那行加进 `CMakeLists.txt`
（**唯一的**原文件改动，一行，且可随时删），然后

    cmake -S . -B build-lp -G "Visual Studio 17 2022" -A x64
    cmake --build build-lp --target strata-lp --config Release

再用 `local/logprobs/server_logprobs.py --engine <build-lp>/.../strata-lp.exe` 起服务。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TARGET = REPO / "src" / "program" / "generate.cpp"
OUT_DIR = HERE / "build"
OUT_SRC = OUT_DIR / "generate_logprobs.cpp"
OUT_CMAKE = OUT_DIR / "CMakeLists.snippet"

MARK = "local/logprobs/build_engine.py"
MARK_LINE = f"// {MARK}\n"

# ── helper：一行 LP，取 ver 的 head logits（采样前、惩罚前）───────────────────
HELPER = MARK_LINE + """// logprobs=K / allowed=...: one `LP` line for the token just emitted, from `ver`'s head
// logits for the window row it came from (before sampling, penalties and temperature).
// Format matches the SYCL port's LP line so one server-side reader serves both.
void local_emit_lp(const strata::core::Verifier& ver, int t, int32_t token, int k,
                   const std::vector<int32_t>& allowed) {
    if (k < 0) return;
    static std::vector<float> row;
    row.resize((size_t) ver.vocab());
    if (!ver.copy_logits(t, row.data())) {
        std::printf("LP nan\\n");
        return;
    }
    float mx = -INFINITY;
    for (const float v : row) mx = std::max(mx, v);
    double se = 0.0;
    for (const float v : row) se += std::exp((double) v - (double) mx);
    const double lse = (double) mx + std::log(se);
    std::printf("LP %.6f", (double) row[(size_t) token] - lse);
    if (!allowed.empty()) {
        // 只要调用方点名的那几个 id：server 侧再按它们归一化 ⇒ 与 allowed_token_ids 语义一致
        for (const int32_t id : allowed) {
            if (id == token) continue;
            std::printf(" %d:%.6f", (int) id, (double) row[(size_t) id] - lse);
        }
        std::printf("\\n");
        return;
    }
    if (k > 0) {
        static std::vector<int32_t> ord;
        ord.resize(row.size());
        for (size_t v = 0; v < row.size(); ++v) ord[v] = (int32_t) v;
        const int kk = std::min<int>(k, (int) row.size());
        std::partial_sort(ord.begin(), ord.begin() + kk, ord.end(),
                          [&](int32_t a, int32_t b) { return row[(size_t) a] > row[(size_t) b]; });
        for (int j = 0; j < kk; ++j)
            std::printf(" %d:%.6f", ord[(size_t) j], (double) row[(size_t) ord[(size_t) j]] - lse);
    }
    std::printf("\\n");
}

"""

# ── 六处锚点替换 ────────────────────────────────────────────────────────────
# (名字, 旧文本, 新文本)。旧文本必须在文件里**恰好出现一次**，否则硬失败。
EDITS: tuple[tuple[str, str, str], ...] = (
    (
        "1 helper 函数",
        "namespace {\n// Windows' WDDM driver model:",
        "namespace {\n" + HELPER + "// Windows' WDDM driver model:",
    ),
    (
        "2 声明 req_logprobs / req_allowed",
        "            int req_cvec = 1;   // cvec=0|1: a loaded control vector for this request (on when absent)\n",
        "            int req_cvec = 1;   // cvec=0|1: a loaded control vector for this request (on when absent)\n"
        + MARK_LINE
        + "            // logprobs=K (0..20): after each \"T id\" an \"LP logprob id:logprob ...\" line.\n"
        + "            // -1 (absent) = no LP lines, so a server that never asks sees today's output.\n"
        + "            int req_logprobs = -1;\n"
        + "            // allowed=a,b,c: with logprobs, print only these token ids (renormalized server-side).\n"
        + "            std::vector<int32_t> req_allowed;\n",
    ),
    (
        "3 解析 logprobs=K 与 allowed=",
        '                    else if (key == "top_k") req_top_k = std::atoi(tok.c_str() + eq + 1);\n',
        '                    else if (key == "top_k") req_top_k = std::atoi(tok.c_str() + eq + 1);\n'
        + MARK_LINE
        + '                    else if (key == "logprobs") req_logprobs = std::atoi(tok.c_str() + eq + 1);\n'
        + '                    else if (key == "allowed") {\n'
        + '                        const std::string csv = tok.substr(eq + 1);\n'
        + '                        size_t p = 0;\n'
        + '                        while (p < csv.size()) {\n'
        + '                            const size_t c = csv.find(",", p);\n'
        + '                            const std::string one = csv.substr(p, c == std::string::npos ? std::string::npos : c - p);\n'
        + '                            if (!one.empty()) req_allowed.push_back((int32_t) std::atoi(one.c_str()));\n'
        + '                            if (c == std::string::npos) break;\n'
        + '                            p = c + 1;\n'
        + '                        }\n'
        + '                    }\n',
    ),
    (
        "4 speculative 路径打 LP",
        '                        std::printf("T %d\\n", (int) outp[(size_t) i]);\n'
        '                        strata::core::progress_beat();\n',
        '                        std::printf("T %d\\n", (int) outp[(size_t) i]);\n'
        + MARK_LINE
        + '                        local_emit_lp(ver, i, outp[(size_t) i], req_logprobs, req_allowed);\n'
        '                        strata::core::progress_beat();\n',
    ),
    (
        "5 plain 路径打 LP",
        '                    std::printf("T %d\\n", (int) outv[(size_t) i]);\n'
        '                    strata::core::progress_beat();\n',
        '                    std::printf("T %d\\n", (int) outv[(size_t) i]);\n'
        + MARK_LINE
        + '                    local_emit_lp(ver, i, outv[(size_t) i], req_logprobs, req_allowed);\n'
        '                    strata::core::progress_beat();\n',
    ),
    (
        "6 每请求清空 allowed",
        "            req_imgs.clear();\n",
        "            req_imgs.clear();\n" + MARK_LINE + "            req_allowed.clear();\n",
    ),
)

CMAKE_SNIPPET = """# 生成物：python local/logprobs/build_engine.py --emit
# 把下面几行贴进仓库根的 CMakeLists.txt（**唯一**的原文件改动，可随时删）：
#
#   add_subdirectory(local/logprobs/build EXCLUDE_FROM_ALL)
#
# local/logprobs/build/CMakeLists.txt 里的 add_executable(strata-lp ...) 会用同一批
# 静态库（strata_core / strata_engine / strata_prefill / strata_spec / ggml ...），
# 只是把 generate.cpp 换成打了 logprobs 补丁的副本。
"""


def status(text: str) -> tuple[int, bool, list[str]]:
    """返回 (已应用条数, 是否可以 --emit, 问题列表)。

    「锚点还在」对**原文件**来说是**好事** —— 说明它没被改过、可以生成补丁副本。
    这里判断的是「能不能安全 emit」，不是「打没打过补丁」。
    """
    applied, problems, ready = 0, [], True
    for name, old, new in EDITS:
        if new in text:
            applied += 1
        hits = text.count(old)
        if hits == 0:
            problems.append(f"{name}：**锚点找不到**（上游可能改过这一行）")
            ready = False
        elif hits > 1:
            problems.append(f"{name}：锚点命中 {hits} 次（不唯一，不能安全替换）")
            ready = False
    if applied:
        problems.append(f"已有 {applied}/{len(EDITS)} 处补丁痕迹 ⇒ 原文件被动过，先确认来源")
        ready = False
    return applied, ready, problems


def emit() -> int:
    if not TARGET.is_file():
        print(f"✗ 找不到 {TARGET}")
        return 1
    text = TARGET.read_text(encoding="utf-8")
    _, ready, problems = status(text)
    if not ready:
        print("✗ 不能安全生成，**没有写任何文件**：")
        for p in problems:
            print(f"  · {p}")
        print("  ⇒ 上游改过这几行；核对 EDITS 后重试。（原文件只读，未被修改）")
        return 1
    out = text
    for name, old, new in EDITS:
        assert out.count(old) == 1, f"{name}: 命中 {out.count(old)} 次"
        out = out.replace(old, new, 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SRC.write_text(out, encoding="utf-8")
    OUT_CMAKE.write_text(CMAKE_SNIPPET, encoding="utf-8")
    print(f"✓ 已生成 {OUT_SRC.relative_to(REPO)}（{len(EDITS)} 处补丁，原文件只读）")
    print(f"✓ 已生成 {OUT_CMAKE.relative_to(REPO)}")
    print(f"  原文件校验：{TARGET.relative_to(REPO)} 大小 {len(text)} 字节，未修改")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成打了 logprobs 补丁的引擎源码副本（原文件只读）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="只看原文件锚点是否还对得上（不写盘）")
    g.add_argument("--emit", action="store_true", help="生成补丁副本 + CMake 片段")
    g.add_argument("--clean", action="store_true", help="删掉生成目录")
    args = ap.parse_args(argv)

    if args.clean:
        import shutil
        if OUT_DIR.is_dir():
            shutil.rmtree(OUT_DIR)
            print(f"✓ 已删除 {OUT_DIR.relative_to(REPO)}")
        else:
            print("（没有生成目录）")
        return 0

    if not TARGET.is_file():
        print(f"✗ 找不到 {TARGET}")
        return 1
    text = TARGET.read_text(encoding="utf-8")
    applied, ready, problems = status(text)
    if args.check:
        print(f"{TARGET.relative_to(REPO)}：{applied}/{len(EDITS)} 处已有补丁；"
              f"{'可以 --emit' if ready else '不能生成'}")
        for p in problems:
            print(f"  · {p}")
        return 0 if ready else 2
    return emit()


if __name__ == "__main__":
    sys.exit(main())
