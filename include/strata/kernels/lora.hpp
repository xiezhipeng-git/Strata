// include/strata/kernels/lora.hpp - a LoRA adapter on a layer's projections.  The engine's side of llama.cpp's
// `llama_adapter_lora` (`--lora` / `--lora-scaled`).  A projection is named by its weight suffix, exactly the
// tensor the loader reads (`blk.<l>.<proj>.weight.lora_{a,b}`); the engine adapts these seven:
//
//   GDN layers:  attn_qkv  (n_embd -> ssm_conv_channels)   attn_gate (n_embd -> ssm_value_dim)
//                ssm_out   (ssm_value_dim -> n_embd)       -- the layer's residual write
//   QSA layers:  attn_q    (n_embd -> 2 * n_head * head_dim)
//                attn_k / attn_v (n_embd -> n_head_kv * head_dim)
//                attn_output (n_head * head_dim -> n_embd) -- the layer's residual write
//
// ssm_out / attn_output alone is the classic scope (what the loader accepted before full-module support, and what
// the zeroth-order trainer wrote); the rest turn the adapter into InferZO-style full-module scope.  For layer l
// with an adapter on projection p there, the projection's output y = W x becomes
//
//   y <- y + s B (A x)        A = lora_a (rank x n_in), B = lora_b (n_out x rank), s = scale * alpha / rank
//
// exactly llama.cpp's `llm_build_lora_mm` (alpha 0: s = scale).  Several files are summed: their ranks are stacked
// per projection, each B already multiplied by its own s.  A and B stay fp32 on every device that runs the layer
// (tens of MB), the products are fp32.  Off unless an adapter is loaded; a loaded one is switched per request
// through a device flag, so the captured graphs are the same either way.  With the flag off the projection's output
// is untouched: the result is bit-identical to the engine without an adapter.  The MTP draft layer never has one
// (it drafts; the verified tokens are the adapted model's).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace strata::kernels {

/// One projection's adapter as the loader builds it (host): `a` rank x n_in, `b` n_out x rank, B already scaled.
struct LoraProjHost {
    int64_t rank = 0, n_in = 0, n_out = 0;
    std::vector<float> a, b;
};

/// One layer's adapters, keyed by the weight suffix (`attn_q`, `ssm_out`, ...).  An empty map (or a projection
/// with rank 0) = no adapter there.
struct LoraLayerHost {
    std::map<std::string, LoraProjHost> projs;
};

/// Largest summed rank of one projection (a block holds a chunk of tokens' A x in shared memory).
constexpr int64_t kLoraMaxRank = 512;
/// Tokens a call may pass while its stream is being captured (the decode and verify windows); a longer call (the
/// prompt path, eager) uses a scratch grown on demand.
constexpr int64_t kLoraCaptureTokens = 64;

struct Lora {
    bool loaded = false;
    int64_t rank_max = 0;    ///< the largest summed rank of any projection of any layer
    /// Per layer: the projections carrying an adapter.  Decided at load time, never per request: it adds kernels
    /// to the graphs, so the graphs depend on it.
    std::vector<std::map<std::string, bool>> adapted;
    /// Layer l's projection `proj` is followed by the adapter.
    bool covers(int64_t l, const char* proj) const {
        if (!loaded || l < 0 || l >= (int64_t) adapted.size()) return false;
        auto it = adapted[(size_t) l].find(proj);
        return it != adapted[(size_t) l].end() && it->second;
    }
};

/// The loaded adapter (empty until `lora_upload`).
const Lora& lora();

/// Upload the adapter built by the loader (`layers` has n_layers entries; rank 0 projections = none there).
/// Starts ON.  Once, before any graph is captured (the graphs hold the tables' addresses).
bool lora_upload(const std::vector<LoraLayerHost>& layers, std::string& err);

/// Replace every table's contents in place (same per-projection rank / n_in / n_out; a different shape is an error,
/// not a re-upload).  The addresses stay, so the graphs already captured keep working: a zeroth-order step reads
/// theta +- eps over a FRESH random direction every step, and one direction per process would make the engine
/// useful for exactly one step.
bool lora_update(const std::vector<LoraLayerHost>& layers, std::string& err);

/// A layer split: put the loaded adapter's tables on the CURRENT device too (lora_apply uses the tables of the
/// device it runs on).  No-op without an adapter or when this device has them already.
bool lora_replicate(std::string& err);

/// The per-request switch, on every device that holds the adapter.  Synchronizes them when it changes, so call
/// it between requests.
void lora_set_enabled(bool on);
bool lora_enabled();

/// The same switch with a sign: 0 = off (the projection's output is untouched, bit-identical to no adapter),
/// +1 = apply the adapter as loaded, -1 = SUBTRACT it (-(B A x) instead of +(B A x)).  MeZO's two reads of one step
/// are theta + eps and theta - eps along one random direction, which is this adapter with the two signs - no second
/// set of tables, and the same captured graph answers both.  Synchronizes like `lora_set_enabled`.
void lora_set_polarity(int sign);
int lora_polarity();

/// Device bytes of the tables on one device (the VRAM a layer split's stage pays for them).
uint64_t lora_device_bytes();
/// The same for tables the loader built, before anything is uploaded (a layer split's VRAM estimate).
uint64_t lora_device_bytes(const std::vector<LoraLayerHost>& layers);

/// y[t * ldy + o] += s B (A x[t * ldx + :]) for t < n, o < n_out, gated by the device flag (off: nothing written).
/// `proj` is the weight suffix the call site sits behind ("attn_q", "ssm_out", ...): a layer can carry adapters on
/// several projections, each with its own tables.  No-op for a projection the adapter does not cover.
/// ldx / ldy 0 = dense rows.  Capturable for n <= kLoraCaptureTokens (two kernels per 8 tokens through the
/// projection's own scratch, the second a programmatic dependent launch where the device has it; no host sync); a
/// longer eager call runs two tiled products through a per-stream scratch grown on demand.
void lora_apply(int64_t layer, const char* proj, const float* x, int64_t ldx, int64_t n, float* y, int64_t ldy,
                void* stream);
/// The same with x as FP16 bits (the prompt path's gated attention, `attn_h`).
void lora_apply_f16(int64_t layer, const char* proj, const uint16_t* x, int64_t ldx, int64_t n, float* y, int64_t ldy,
                    void* stream);

}  // namespace strata::kernels
