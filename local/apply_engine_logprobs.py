"""给 Strata 的 **native(CUDA)** 引擎加 per-request logprobs —— 锚点式补丁，**独立文件**。

## 为什么需要它

`sycl/serve/server_intel.py` 能把 `/v1/chat/completions` 的 `logprobs` / `top_logprobs`
转成 OpenAI 格式，靠的是 **SYCL 引擎**在每个 `T <id>` 行后多打一行
`LP <logprob> <id>:<logprob> ...`（`sycl/src/program/generate.cpp:9244`）。
**native 引擎没有这一行**（只有 `T %d`），所以 `serve/server.py` 上接不出分布。

## 为什么改动很小

native 引擎**已经有** `Verifier::copy_logits(int t, float* host)`
（`include/strata/core/verify.hpp:86`，`src/core/verify.cpp:2878`）—— 和 SYCL 用的是同一个接口，
而且 `src/program/generate.cpp:10696/11819` 已经在用它 dump logits 到文件。
差的只是：**解析 `logprobs=K` + 在两个 `T` 行后面打出 `LP` 行**。

## 为什么是「独立文件」

`src/` 是共享源码，直接改会和上游更新冲突。所以：
- 补丁以**锚点替换**的方式写在这里，`--apply` 才动 `src/`，`--revert` 可完整还原；
- 首次 `--apply` 会留一份 `src/program/generate.cpp.localorig` 备份；
- 锚点对不上（上游改过那几行）就**硬失败并报出是哪个锚点**，不静默改错。
- ⚠️ 这是**本地实验用**：Kaggle 那条线不经过 Strata（用 SGLang/vLLM），用不到它。

## 用法

    python local/apply_engine_logprobs.py --check     # 只看状态，不写盘
    python local/apply_engine_logprobs.py --apply     # 打补丁（幂等）
    python local/apply_engine_logprobs.py --revert    # 还原

打完补丁后要用 `local/server_logprobs.py` 起服务（它把 `LP` 行接成 OpenAI 的 `logprobs`）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
TARGET = REPO / "src" / "program" / "generate.cpp"
# 备份放 local/ 里，**不放 src/ 旁边** —— 免得 `src/` 里多出一个会让人以为「源码被改过」的文件。
BACKUP = HERE / "generate.cpp.localorig"

MARK = "local/apply_engine_logprobs.py"          # 打在源码里的可识别标记
MARK_LINE = f"// {MARK}\n"

# ── 要插入的 helper（放在文件顶部那个匿名 namespace 里）──────────────────────
HELPER = MARK_LINE + """// logprobs=K: one `LP` line for the token just emitted, from `ver`'s head logits for
// the window row it came from (before sampling, penalties and temperature).  Mirrors the
// SYCL port's LP line so the same server-side reader works on both.
void local_emit_lp(const strata::core::Verifier& ver, int t, int32_t token, int k) {
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

# ── 五处锚点替换 ────────────────────────────────────────────────────────────
# (名字, 旧文本, 新文本)。旧文本必须在文件里**恰好出现一次**。
EDITS: tuple[tuple[str, str, str], ...] = (
    (
        "1 helper 函数",
        "namespace {\n// Windows' WDDM driver model:",
        "namespace {\n" + HELPER + "// Windows' WDDM driver model:",
    ),
    (
        "2 声明 req_logprobs",
        "            int req_cvec = 1;   // cvec=0|1: a loaded control vector for this request (on when absent)\n",
        "            int req_cvec = 1;   // cvec=0|1: a loaded control vector for this request (on when absent)\n"
        + MARK_LINE
        + "            // logprobs=K (0..20): after each \"T id\" an \"LP logprob id:logprob ...\" line.\n"
        + "            // -1 (absent) = no LP lines, so a server that never asks sees today's output.\n"
        + "            int req_logprobs = -1;\n",
    ),
    (
        "3 解析 logprobs=K",
        '                    else if (key == "top_k") req_top_k = std::atoi(tok.c_str() + eq + 1);\n',
        '                    else if (key == "top_k") req_top_k = std::atoi(tok.c_str() + eq + 1);\n'
        + MARK_LINE
        + '                    else if (key == "logprobs") req_logprobs = std::atoi(tok.c_str() + eq + 1);\n',
    ),
    (
        "4 speculative 路径打 LP",
        '                        std::printf("T %d\\n", (int) outp[(size_t) i]);\n'
        '                        strata::core::progress_beat();\n',
        '                        std::printf("T %d\\n", (int) outp[(size_t) i]);\n'
        + MARK_LINE
        + '                        local_emit_lp(ver, i, outp[(size_t) i], req_logprobs);\n'
        '                        strata::core::progress_beat();\n',
    ),
    (
        "5 plain 路径打 LP",
        '                    std::printf("T %d\\n", (int) outv[(size_t) i]);\n'
        '                    strata::core::progress_beat();\n',
        '                    std::printf("T %d\\n", (int) outv[(size_t) i]);\n'
        + MARK_LINE
        + '                    local_emit_lp(ver, i, outv[(size_t) i], req_logprobs);\n'
        '                    strata::core::progress_beat();\n',
    ),
)


def status(text: str) -> tuple[int, list[str]]:
    """返回 (已应用条数, 问题列表)。"""
    applied, problems = 0, []
    for name, old, new in EDITS:
        if new in text:
            applied += 1
        elif old in text:
            problems.append(f"{name}：未应用（锚点还在）")
        elif text.count(old) == 0 and old not in text and new not in text:
            problems.append(f"{name}：**锚点找不到**（上游可能改过这一行）")
        else:
            problems.append(f"{name}：状态不明")
    return applied, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="给 Strata native 引擎加 per-request logprobs（锚点补丁）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="只看状态，不写盘")
    g.add_argument("--apply", action="store_true", help="打补丁（幂等）")
    g.add_argument("--revert", action="store_true", help="还原到备份")
    args = ap.parse_args(argv)

    if not TARGET.is_file():
        print(f"✗ 找不到 {TARGET}")
        return 1

    if args.revert:
        if not BACKUP.is_file():
            print(f"✗ 没有备份 {BACKUP.name}，无法还原")
            return 1
        TARGET.write_bytes(BACKUP.read_bytes())
        print(f"✓ 已从 {BACKUP.name} 还原 {TARGET.relative_to(REPO)}")
        return 0

    text = TARGET.read_text(encoding="utf-8")
    applied, problems = status(text)

    if args.check:
        print(f"{TARGET.relative_to(REPO)}：{applied}/{len(EDITS)} 处已应用")
        for p in problems:
            print(f"  · {p}")
        if applied == len(EDITS):
            print("  ⇒ 补丁完整；可以配 local/server_logprobs.py 起服务")
        elif applied == 0 and not problems:
            print("  ⇒ 未打补丁")
        return 0 if applied == len(EDITS) else 2

    # --apply
    if applied == len(EDITS):
        print(f"= 已经打过补丁（{len(EDITS)}/{len(EDITS)}），无需重复")
        return 0

    missing = []
    out = text
    for name, old, new in EDITS:
        if new in out:
            continue
        if out.count(old) != 1:
            missing.append(f"{name}（命中 {out.count(old)} 次）")
            continue
        out = out.replace(old, new, 1)
    if missing:
        print("✗ 锚点对不上，**没有写盘**：")
        for m in missing:
            print(f"  · {m}")
        print("  ⇒ 上游可能改过那几行；请核对 local/README.md 的说明后手工调整锚点。")
        return 1

    if not BACKUP.is_file():
        BACKUP.write_bytes(text.encode("utf-8"))
        print(f"· 备份 → {BACKUP.relative_to(REPO)}")
    TARGET.write_text(out, encoding="utf-8")
    print(f"✓ 已打补丁：{TARGET.relative_to(REPO)}（{len(EDITS)} 处）")
    print("  下一步：重编译引擎（见 docs/INSTALL.md / CMakeLists.txt），再用 local/server_logprobs.py 起服务")
    return 0


if __name__ == "__main__":
    sys.exit(main())
