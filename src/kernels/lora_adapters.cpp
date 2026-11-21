// src/kernels/lora_adapters.cpp - llama.cpp's adapter GGUFs read into the tables `lora_upload` takes.
//
// One loader for every engine path: it needs only the model's geometry (`strata::core::ModelGeometry`, filled from
// the same GGUF metadata by either family's reader) and the adapter's own tensor names.  The projections the engine
// adapts are the seven in `lora.hpp` (GDN: attn_qkv / attn_gate / ssm_out; QSA: attn_q / attn_k / attn_v /
// attn_output); `blk.<l>.ssm_out.weight` / `blk.<l>.attn_output.weight` alone is the classic scope and passes the
// same checks.  A tensor on any other matrix is an error (running part of an adapter gives neither model), and so
// is a file built for another architecture.
#include "strata/kernels/lora_adapters.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_family.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/lora.hpp"

namespace strata::kernels {

// --lora / --lora-scaled: llama.cpp's `llama_adapter_lora_init` for the projections the engine adapts (every
// file's `blk.<l>.{attn_q,attn_k,attn_v,attn_output,attn_qkv,attn_gate,ssm_out}.weight.lora_{a,b}`, B times
// scale * alpha / rank, several files' ranks stacked per projection), into the tables `lora_upload` takes.  A
// tensor the engine does not adapt is an error: running part of an adapter would give neither model.  `summary`
// is what INFO reports; `digest` identifies the uploaded tables.
bool load_lora_adapters(const LoraAdapterFiles& files, const strata::core::ModelGeometry& g, std::string& summary,
                        uint64_t& digest, std::string& err, const bool update) {
    const int64_t L = g.n_layers, N = g.n_embd;
    // The projections a layer family carries and their (n_in, n_out), straight from `layout.cpp`'s checks.
    const std::pair<const char*, std::pair<int64_t, int64_t>> qsa_projs[] = {
        {"attn_q", {N, 2 * g.n_head * g.head_dim}},
        {"attn_k", {N, g.n_head_kv * g.head_dim}},
        {"attn_v", {N, g.n_head_kv * g.head_dim}},
        {"attn_output", {g.n_head * g.head_dim, N}},
    };
    const std::pair<const char*, std::pair<int64_t, int64_t>> gdn_projs[] = {
        {"attn_qkv", {N, g.ssm_conv_channels}},
        {"attn_gate", {N, g.ssm_value_dim}},
        {"ssm_out", {g.ssm_value_dim, N}},
    };
    // The shared expert's three projections.  On EVERY layer of a MoE family (the shared expert is not part of
    // the QSA/GDN split), and skipped when the model has none (`n_ff == 0`, a dense family).
    const std::pair<const char*, std::pair<int64_t, int64_t>> shexp_projs[] = {
        {"ffn_gate_shexp", {N, g.n_ff}},
        {"ffn_up_shexp", {N, g.n_ff}},
        {"ffn_down_shexp", {g.n_ff, N}},
    };
    std::vector<strata::kernels::LoraLayerHost> layers((size_t) L);
    auto f32_of = [&](const strata::GgufFile& f, const strata::TensorInfo& t, std::vector<float>& out) -> bool {
        const uint8_t* p = f.tensor_data(t);
        out.resize((size_t) t.elements());
        if (t.type == 0) {
            std::memcpy(out.data(), p, out.size() * sizeof(float));
        } else if (t.type == 1 || t.type == 30) {   // F16, BF16
            for (size_t i = 0; i < out.size(); ++i) {
                uint16_t h;
                std::memcpy(&h, p + 2 * i, 2);
                out[i] = t.type == 1 ? strata::fp16_to_fp32(h) : strata::bf16_to_fp32(h);
            }
        } else {
            err = t.name + ": " + t.type_name() + " (f32, f16 or bf16 only)";
            return false;
        }
        return true;
    };
    for (const auto& [path, scale] : files) {
        try {
            strata::GgufFile f(path);
            const strata::MetaValue* type = f.get("general.type");
            const strata::MetaValue* atype = f.get("adapter.type");
            if (type == nullptr || type->s != "adapter" || atype == nullptr || atype->s != "lora") {
                err = path + ": not a LoRA adapter GGUF (general.type 'adapter', adapter.type 'lora')";
                return false;
            }
            const strata::MetaValue* arch = f.get("general.architecture");
            const std::string model_arch = strata::model_family_arch(g.family);   // the file this engine runs
            if (arch != nullptr && !model_arch.empty() && arch->s != model_arch) {
                err = path + ": made for '" + arch->s + "', not " + model_arch;
                return false;
            }
            const strata::MetaValue* am = f.get("adapter.lora.alpha");
            const double alpha = am != nullptr && am->is_num() ? am->num() : 0.0;
            std::map<std::string, std::pair<const strata::TensorInfo*, const strata::TensorInfo*>> pairs;
            for (const strata::TensorInfo& t : f.tensors()) {
                const bool is_a = t.name.size() > 7 && t.name.compare(t.name.size() - 7, 7, ".lora_a") == 0;
                const bool is_b = t.name.size() > 7 && t.name.compare(t.name.size() - 7, 7, ".lora_b") == 0;
                if (!is_a && !is_b) { err = path + ": unexpected tensor " + t.name; return false; }
                auto& pr = pairs[t.name.substr(0, t.name.size() - 7)];
                (is_a ? pr.first : pr.second) = &t;
            }
            if (pairs.empty()) { err = path + ": no lora_a / lora_b tensors"; return false; }
            for (const auto& [base, pr] : pairs) {
                if (pr.first == nullptr || pr.second == nullptr) { err = path + ": " + base + " lacks lora_a or lora_b"; return false; }
                long l = -1;
                char which[32] = {};
                if (std::sscanf(base.c_str(), "blk.%ld.%31[a-z_].weight", &l, which) != 2 || l < 0 || l >= L ||
                    base != "blk." + std::to_string(l) + "." + which + ".weight") {
                    err = path + ": " + base + ": not a layer of this model";
                    return false;
                }
                const bool qsa = strata::core::is_qsa_layer(g, l);
                const std::span<const std::pair<const char*, std::pair<int64_t, int64_t>>> attn_table =
                    qsa ? std::span<const std::pair<const char*, std::pair<int64_t, int64_t>>>(qsa_projs)
                        : std::span<const std::pair<const char*, std::pair<int64_t, int64_t>>>(gdn_projs);
                // The shared expert's three come first: they exist on both layer families, so a name that matches
                // one of them is never an attention projection.
                const std::span<const std::pair<const char*, std::pair<int64_t, int64_t>>> ffn_table(shexp_projs);
                const std::pair<int64_t, int64_t>* dims = nullptr;
                if (g.n_ff > 0)
                    for (const auto& [name, d2] : ffn_table)
                        if (std::strcmp(which, name) == 0) { dims = &d2; break; }
                if (dims == nullptr)
                    for (const auto& [name, d2] : attn_table)
                        if (std::strcmp(which, name) == 0) { dims = &d2; break; }
                if (dims == nullptr) {
                    std::string want;
                    for (const auto& [name, d2] : attn_table) want += (want.empty() ? "" : ", ") + std::string(name);
                    if (g.n_ff > 0)
                        for (const auto& [name, d2] : ffn_table) want += ", " + std::string(name);
                    err = path + ": " + base + ": the engine adapts only " + want + " on this layer";
                    return false;
                }
                const int64_t n_in = dims->first, n_out = dims->second;
                const strata::TensorInfo &ta = *pr.first, &tb = *pr.second;
                // llama.cpp's shapes: lora_a ne = [n_in, r], lora_b ne = [r, n_out]
                if (ta.shape.size() != 2 || tb.shape.size() != 2 || (int64_t) ta.shape[0] != n_in ||
                    (int64_t) tb.shape[1] != n_out || ta.shape[1] != tb.shape[0] || ta.shape[1] == 0) {
                    err = path + ": " + base + ": lora_a must be [" + std::to_string((long long) n_in) + ", r] and lora_b [r, " +
                          std::to_string((long long) n_out) + "]";
                    return false;
                }
                const int64_t r = (int64_t) ta.shape[1];
                std::vector<float> a, b;
                if (!f32_of(f, ta, a) || !f32_of(f, tb, b)) { err = path + ": " + err; return false; }
                const float s = (float) (alpha != 0.0 ? scale * alpha / (double) r : scale);
                strata::kernels::LoraProjHost& h = layers[(size_t) l].projs[which];
                const int64_t r0 = h.rank, rn = r0 + r;
                if (rn > strata::kernels::kLoraMaxRank) {
                    err = base + ": the summed rank is above " + std::to_string((long long) strata::kernels::kLoraMaxRank);
                    return false;
                }
                h.n_in = n_in;
                h.n_out = n_out;
                h.a.insert(h.a.end(), a.begin(), a.end());   // A: the new rows below the old ones
                std::vector<float> nb((size_t) (n_out * rn));    // B: the new columns right of the old ones
                for (int64_t i = 0; i < n_out; ++i) {
                    for (int64_t k = 0; k < r0; ++k) nb[(size_t) (i * rn + k)] = h.b[(size_t) (i * r0 + k)];
                    for (int64_t k = 0; k < r; ++k) nb[(size_t) (i * rn + r0 + k)] = s * b[(size_t) (i * r + k)];
                }
                h.b = std::move(nb);
                h.rank = rn;
            }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }
    int covered = 0;
    for (const auto& h : layers) covered += !h.projs.empty();
    {
        strata::core::SessionIdentityBuilder b(0x4c4f5241ull);   // "LORA"
        for (int64_t l = 0; l < L; ++l) {
            const auto& hl = layers[(size_t) l];
            b.i64("projs", (int64_t) hl.projs.size());
            for (const auto& [name, h] : hl.projs) {
                b.str("proj", name);
                b.i64("rank", h.rank);
                b.bytes("a", h.a.data(), h.a.size() * sizeof(float));
                b.bytes("b", h.b.data(), h.b.size() * sizeof(float));
            }
        }
        digest = b.digest();
        if (digest == 0) digest = 1;   // 0 means "none loaded"
    }
    const uint64_t bytes = strata::kernels::lora_device_bytes(layers);
    if (update ? !strata::kernels::lora_update(layers, err) : !strata::kernels::lora_upload(layers, err)) return false;
    summary = std::to_string(covered);
    std::fprintf(stderr, "strata: LoRA %s: %zu file%s, %d layers adapted, %.1f MiB per device\n",
                 update ? "updated" : "loaded", files.size(), files.size() == 1 ? "" : "s", covered,
                 (double) bytes / (1024.0 * 1024.0));
    return true;
}

}  // namespace strata::kernels
