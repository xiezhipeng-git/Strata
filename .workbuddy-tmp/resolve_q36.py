# -*- coding: utf-8 -*-
"""解决 q36_engine.cu 的 10 处合并冲突：保留本分支的 LoRA 钩子，并入上游的
kv-int8/fp4、GDN checkpoint resume_len、MTP sampled 开关。"""
import io, re, sys

P = r"engine/src/qwen35/q36_engine.cu"
src = io.open(P, encoding="utf-8", newline="").read()
NL = "\r\n" if "\r\n" in src[:2000] else "\n"

# 按 <<<<<<< ... ======= ... >>>>>>> 切块
pat = re.compile(r"<<<<<<< HEAD\r?\n(.*?)\r?\n?=======\r?\n(.*?)\r?\n?>>>>>>> upstream/main\r?\n?",
                 re.S)
blocks = list(pat.finditer(src))
assert len(blocks) == 10, f"expect 10 conflicts, got {len(blocks)}"

def rep(i, text):
    """第 i 个冲突块（0 起）替换为 text。"""
    global src
    m = blocks[i]
    src = src[:m.start()] + text + src[m.end():]
    # 重新扫描（位置已失效）
    blocks.clear()
    blocks.extend(pat.finditer(src))

# 1. Sess 字段：双方都留
rep(0, "        int lora_state = -1;                // the adapter setting its state was read with (-1 = no adapter / unused)" + NL +
          "        std::vector<Ckpt> ckpts;            // the GDN checkpoint ring (empty until the feature is on and a save happens)" + NL +
          "        int ckpt_next = 0;                  // the ring's next slot" + NL +
          "        int64_t ckpt_last_k = -1;           // the last interval multiple a checkpoint was taken at")

# 2. match_len：上游的新算法（最长公共前缀 + 图片回拉）+ 本分支的 lora 闸门
rep(0,
    "    /// The longest common prefix of session ss's held tokens and `ids`, with the picture rule: a prefix that ends inside" + NL +
    "    /// a picture is pulled back to that picture's start, and the pictures wholly inside the prefix must be the arriving" + NL +
    "    /// ones (else 0).  It works whether the held run is shorter or longer than the prompt.  Plus the lora gate: a state" + NL +
    "    /// read under the other adapter setting is never reused (it belongs to weights that did not produce it)." + NL +
    "    size_t match_len(const Engine::Sess& ss, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs," + NL +
    "                     int want_lora) const {" + NL +
    "        if (want_lora >= 0 && ss.lora_state != want_lora) return 0;   // its state belongs to the other weights" + NL +
    "        size_t L = 0;" + NL +
    "        const size_t n = std::min(ss.live.size(), ids.size());" + NL +
    "        while (L < n && ss.live[L] == ids[L]) ++L;" + NL +
    "        for (const ImgKey& k : imgs) {                        // (a prefix ending inside a picture cannot be resumed there)" + NL +
    "            if (k.cell >= (int64_t) L) continue;" + NL +
    "            if (k.cell + k.n > (int64_t) L) L = k.cell > 0 ? (size_t) k.cell : 0;")

# 3. resume_len（上游新函数）+ pick_session 保留本分支的 want_lora 参数
rep(0,
    "    /// The length a request can resume session ss at.  With GDN checkpoints on: the whole held run when it starts the" + NL +
    "    /// prompt (its state is exactly there, as before), else the newest checkpoint at or below the longest common prefix;" + NL +
    "    /// 0 when no checkpoint that low (the caller then resets).  With STRATA_Q36_CKPT_N=0, the old rule: the whole held run" + NL +
    "    /// and only when it is a strict prefix of the prompt." + NL +
    "    size_t resume_len(const Engine::Sess& ss, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {" + NL +
    "        size_t L = match_len(ss, ids, imgs, -1);" + NL +
    "        if (L == 0 || ids.empty()) return 0;" + NL +
    "        if (E.ckpt_n <= 0 || E.ckpt_bytes == 0)" + NL +
    "            return (L == ss.live.size() && L < ids.size()) ? L : 0;" + NL +
    "        const size_t last = ids.size() - 1;   // the last prompt token is always read (its logits pick the first)" + NL +
    "        if (L > last) L = last;" + NL +
    "        if (L == ss.live.size()) return L;    // the held run is exactly this prefix: its own state resumes it" + NL +
    "        const int64_t p = E.ckpt_at_or_below(ss, L);" + NL +
    "        return p > 0 ? (size_t) p : 0;" + NL +
    "    }" + NL +
    "" + NL +
    "    /// The session a request alone takes: the one that can resume furthest, else an empty one, else the least recently" + NL +
    "    /// used (never a decoding slot).  A state read under the other adapter setting never wins (want_lora)." + NL +
    "    int pick_session(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs, int want_lora) const {")

# 4. pick_session 循环：resume_len + lora 不匹配的会话计 0
rep(0, "            const size_t l = (want_lora < 0 || ss.lora_state == want_lora) ? resume_len(ss, ids, imgs) : 0;")

# 5. 请求处理：保留本分支的 lora=0|1 块，会话挑选带 want_state（此时它是剩余第 1 块）
h5 = blocks[0].group(1)
assert "want_lora" in h5
rep(0, h5)

# 6. adm 复制分支：保留本分支的带 lora 口径
h6 = blocks[0].group(1)
assert "want_state" in h6
rep(0, h6)

# 7. spec：上游的 STRATA_Q36_MTP_SAMPLED 开关 + 本分支的 !want_lp（LP 行不能在 draft 在途时写）
rep(0,
    "        // text); sampled ones, and requests beside others, one token at a time.  A request that asks for logprobs" + NL +
    "        // decodes one token at a time too: a draft's row is stepped past before its LP line could be written." + NL +
    "        // (STRATA_Q36_MTP_SAMPLED=1: opt-in toggle allowing sampled requests to utilize MTP speculative drafting)" + NL +
    "        static const bool mtp_sampled = std::getenv(\"STRATA_Q36_MTP_SAMPLED\") != nullptr && std::string(std::getenv(\"STRATA_Q36_MTP_SAMPLED\")) != \"0\";" + NL +
    "        const bool spec = E.mtp_d > 0 && (!smp.needs_logits() || mtp_sampled) && !adm && !want_lp;")

# 8. LP 行捕获：保留本分支
h8 = blocks[0].group(1)
assert "lp_row" in h8
rep(0, h8)

# 9. run() 局部声明：双方都留
rep(0,
    "    std::vector<std::pair<std::string, float>> lora_files;   // --lora / --lora-scaled: (adapter GGUF, scale)" + NL +
    "    int kv_mode = KV_FP32;   // --kv: fp32 (the default), fp16, fp8 (int8) or fp4 (q4_0, rotated)")

# 10. 参数解析：本分支的 --lora/--lora-scaled（自带收尾括号）+ 上游的 --kv（吃后面的公共括号）
rep(0,
    "        else if (a == \"--lora\") lora_files.push_back({next(), 1.0f});" + NL +
    "        else if (a == \"--lora-scaled\") {   // FILE:SCALE[,FILE:SCALE...]" + NL +
    "            std::stringstream list(next());" + NL +
    "            std::string item;" + NL +
    "            while (std::getline(list, item, ',')) {" + NL +
    "                if (item.empty()) continue;" + NL +
    "                const size_t colon = item.rfind(':');" + NL +
    "                if (colon == std::string::npos || colon + 1 >= item.size()) {" + NL +
    "                    std::fprintf(stderr, \"strata-q36: --lora-scaled: expected FILE:SCALE, got '%s'\\n\", item.c_str());" + NL +
    "                    return 2;" + NL +
    "                }" + NL +
    "                lora_files.push_back({item.substr(0, colon), (float) std::atof(item.c_str() + colon + 1)});" + NL +
    "            }" + NL +
    "        }" + NL +
    "        else if (a == \"--kv\" || a.rfind(\"--kv=\", 0) == 0) {" + NL +
    "            // the K / V cache's width: fp32 (default, the engine's own), fp16 (half), fp8/int8 (quarter) or fp4 (rotated q4_0)." + NL +
    "            const std::string v = a == \"--kv\" ? next() : a.substr(5);" + NL +
    "            if (v == \"fp32\" || v == \"f32\") kv_mode = KV_FP32;" + NL +
    "            else if (v == \"fp16\" || v == \"f16\" || v == \"half\") kv_mode = KV_FP16;" + NL +
    "            else if (v == \"fp8\" || v == \"int8\" || v == \"q8\" || v == \"q8_0\") kv_mode = KV_INT8;" + NL +
    "            else if (v == \"fp4\" || v == \"q4\" || v == \"q4_0\") kv_mode = KV_Q4;" + NL +
    "            else {" + NL +
    "                std::fprintf(stderr, \"strata-q36: --kv takes fp32, fp16, fp8 or fp4 (not %s)\\n\", v.c_str());" + NL +
    "                return 2;")

assert "<<<<<<<" not in src and ">>>>>>>" not in src, "markers remain"
io.open(P, "w", encoding="utf-8", newline="").write(src)
print("q36 resolved: 10 conflicts")
