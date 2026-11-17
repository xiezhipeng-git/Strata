// src/prefill/score.cpp - see include/strata/prefill/score.hpp.
//
// MeZO scoring IS engine inference.  `Prefill::run` walks all 48 layers through the engine's expert cache, LoRA
// tables and KV pools; `on_chunk` hands us each chunk's final residual rows.  The head below is the SAME projection
// decode and the verify window run (verify.cpp's head column chunk): `native_quantize_q8_1` + `native_mmvq` over
// the GGUF blocks `NativeHead` already loaded and placed.  No weight copy, no dequantization into a second FP16
// image, no second GEMM implementation - and therefore no disagreement between the logits a training step scores
// and the logits the engine generates from.  `score_rows` is the one thing generation does not need:
// log p(target) = logits[target] - logsumexp over the whole vocabulary.
//
// VRAM: this file does not schedule memory.  A fixed kHeadCols-row head block is laid out ONCE in `attach`, the way
// `Prefill::init` lays out a session's buffers once; it is never probed against cudaMemGetInfo, never halved to fit,
// and never taken from / given back to the expert cache per run.  Measured on this geometry, the default 8-column
// block is 8 MiB (1 column 1 MiB, 3 columns 3 MiB), instead of the old 1.27 GiB FP16 image plus a 769 MiB logits
// block, so a 125B pack that fills the card still scores.  Where the head weights live (VRAM, mapped host) and
// which experts are resident are the engine's decisions, unchanged by a SCORE request.  STRATA_SCORE_DEBUG A/Bs
// column 0 against the decode head's NativeHead::run: on this build the worst absolute difference is 0.
#include "strata/prefill/score.hpp"

#include "strata/core/geometry.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/prefill/score_rows.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace strata::prefill {
namespace {

// The final mixer's RMSNorm epsilon, the same constant `core::layer`'s `lm_head_mix` uses.  The mixer kernel is
// shared with the engine's verify window, so both must agree about this normalisation.
constexpr float kRmsEps = 1e-6f;

// Head columns per launch.  `fused_gr_read_multi` takes at most kFusedGrMaxT (8) tokens per call, and
// `native_mmvq`'s multi-column kernels take at most MAX_NCOLS (>= 8); eight is the largest block both accept, and
// the engine's verify window uses the same bound.  STRATA_SCORE_ROWS lowers it for diagnosis, never raises it.
constexpr int64_t kHeadColsMax = 8;

int64_t env_i64(const char* name, int64_t fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    return (int64_t) std::atoll(v);
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

void dev_free(void*& p) {
    if (p != nullptr) {
        cudaFree(p);
        p = nullptr;
    }
}

// One allocation, once, in attach.  There is no retry-and-halve loop: a fixed block either fits at start-up or the
// engine refuses the SCORE path loudly.  Sizing it from free VRAM would make the training scorer a second memory
// scheduler competing with the expert cache - the thing that made a full-card 125B pack fail here before.
bool dev_alloc(void*& p, size_t bytes, const char* what, std::string& err) {
    p = nullptr;
    if (bytes == 0) return true;
    const cudaError_t rc = cudaMalloc(&p, bytes);
    if (rc != cudaSuccess) {
        cudaGetLastError();
        err = std::string("score: ") + what + " (" + std::to_string(bytes >> 20) + " MiB): " + cudaGetErrorString(rc);
        return false;
    }
    return true;
}

}  // namespace

struct Scorer::Impl {
    Prefill* pf = nullptr;
    const core::WeightTable* wt = nullptr;
    core::ModelGeometry g;
    void* stream = nullptr;

    // The engine's head: `head` owns the GGUF blocks and their placement.  `wo` is only the fallback for a build
    // that has no native head; either way `wblocks()` points at bytes the engine already loaded, never at a copy.
    const core::NativeHead* head = nullptr;
    const core::WeightRef* wo = nullptr;    ///< output.weight (shape / fallback blocks)
    const core::WeightRef* wn = nullptr;    ///< output_hc_norm.weight  (f32)
    const core::WeightRef* wd = nullptr;    ///< output_hc_down.weight  (bf16)
    const core::WeightRef* wu = nullptr;    ///< output_hc_up.weight    (bf16)
    const void* wblocks() const { return head != nullptr ? head->weights() : wo->native_data; }
    int wtype() const { return head != nullptr ? head->type() : wo->native_type; }

    // A FIXED working set, laid out once in attach and owned for the Scorer's life.  One head block of `cols` rows;
    // `score_chunk` walks a chunk in blocks of this size, so the buffer size does not grow with the sequence.
    int64_t cols = kHeadColsMax;
    float* mixed = nullptr;     ///< cols x n_embd (fp32, fused_gr_read_multi's output)
    float* xn = nullptr;        ///< kFusedGrMaxT x hc x n_embd (the mixer's stream scratch)
    float* lo = nullptr;        ///< cols x hc_lr
    float* rs = nullptr;        ///< cols x hc
    float* inj = nullptr;       ///< cols x hc
    uint8_t* xq = nullptr;      ///< native_q8_1_bytes(n_embd, cols)
    float* logits = nullptr;    ///< cols x n_vocab
    float* lse = nullptr;       ///< score_rows scratch
    float* lp = nullptr;        ///< cols, device row log-probabilities
    std::vector<kernels::FusedGrArgs> args;

    // The target tokens are the only buffer that scales with the request.  Grow it once, keep it: exactly what
    // Prefill does with a session's per-request staging, and never a per-run alloc/free against the expert cache.
    int32_t* tgt = nullptr;
    int64_t tgt_cap = 0;

    // The rows the objective actually reads (empty = every row).  One byte a position, owned by the caller's
    // selection; positions past its end are kept (only the last one, which has no target, ever is).
    std::vector<uint8_t> keep;

    float* out_rows = nullptr;   ///< the caller's host output for this run
    bool warned_loan = false;
    int64_t scored = 0;
    int64_t skipped = 0;
    double ms_head = 0;
    std::string note;
    size_t scratch_bytes = 0;

    ~Impl() { release(); }

    bool ensure_targets(int64_t n, std::string& err);
    bool score_chunk(const float* R_rows, int64_t T, int64_t pos0, std::string& err);
    void release();
};

void Scorer::Impl::release() {
    dev_free((void*&) mixed);
    dev_free((void*&) xn);
    dev_free((void*&) lo);
    dev_free((void*&) rs);
    dev_free((void*&) inj);
    dev_free((void*&) xq);
    dev_free((void*&) logits);
    dev_free((void*&) lse);
    dev_free((void*&) lp);
    dev_free((void*&) tgt);
    tgt_cap = 0;
    out_rows = nullptr;
    scored = 0;
    ms_head = 0;
}

bool Scorer::Impl::ensure_targets(int64_t n, std::string& err) {
    if (n <= tgt_cap) return true;
    dev_free((void*&) tgt);
    if (!dev_alloc((void*&) tgt, (size_t) n * sizeof(int32_t), "the target tokens", err)) return false;
    tgt_cap = n;
    return true;
}

bool Scorer::Impl::score_chunk(const float* R_rows, int64_t T, int64_t pos0, std::string& err) {
    if (T <= 0) return true;
    const int64_t hc = g.hc, ne = g.n_embd, lr = g.hc_lr;
    const double t0 = now_ms();
    const int64_t skipped_before = skipped;
    const bool debug = std::getenv("STRATA_SCORE_DEBUG") != nullptr;

    for (int64_t r0 = 0; r0 < T; r0 += cols) {
        const int64_t n = std::min(cols, T - r0);

        // 0. Does the objective read any row of this block?  Without a mask every row is read; with one, a block the
        //    selection left empty costs nothing at all - no mixer, no head - and its rows come back 0.0, which is
        //    what they contribute to a sum over the kept positions.  This is exact, not a shortcut: `score_rows`
        //    would reduce the same block to log p(targets that are not in the loss).
        if (!keep.empty()) {
            const int64_t klen = (int64_t) keep.size();
            bool any = false;
            for (int64_t i = 0; i < n && !any; ++i) {
                const int64_t p = pos0 + r0 + i;
                any = p >= klen || keep[(size_t) p] != 0;
            }
            if (!any) {
                std::fill_n(out_rows + pos0 + r0, (size_t) n, 0.0f);
                skipped += n;
                continue;
            }
        }

        // 1. The final mixer, the engine's own multi-token hyper-connection read (verify.cpp:2617 head_mix_multi).
        //    `R_rows` is the prompt path's final residual; no layer is rerun here.
        for (int64_t i = 0; i < n; ++i) {
            const int64_t t = r0 + i;
            kernels::FusedGrArgs& a = args[(size_t) i];
            a = kernels::FusedGrArgs{};
            a.R = R_rows + (size_t) t * hc * ne;
            a.R_out = const_cast<float*>(a.R);
            a.apply = false;   // nothing was computed after this read: no half to fold in
            a.w_norm = (const float*) wn->data;
            a.w_down = (const uint16_t*) wd->data;
            a.w_up = (const uint16_t*) wu->data;
            a.w_inject = nullptr;   // the FINAL mixer has no write-back
            a.eps = kRmsEps;
            a.lo = lo + (size_t) i * lr;
            a.rs = rs + (size_t) i * hc;
            a.inject_out = inj + (size_t) i * hc;
            a.mixed = mixed + (size_t) i * ne;
        }
        // The return value only says whether q8_1 images were also written; this read wants none.
        kernels::fused_gr_read_multi(args.data(), (int) n, xn, stream);
        if (cudaGetLastError() != cudaSuccess) {
            err = "score: the final mixer failed";
            return false;
        }

        // 2. The engine's head.  `native_quantize_q8_1` + `native_mmvq` are the exact kernels decode and the
        //    verify window use, over the exact GGUF blocks NativeHead loaded (no FP16 image, no dequant per block).
        //    A multi-column call is bitwise the single-column one (native_mmvq.cu:3556).
        try {
            kernels::native_quantize_q8_1(mixed, xq, (int) ne, (int) n, stream);
            kernels::native_mmvq(wtype(), wblocks(), xq, logits, (int) ne, (int) g.n_vocab, (int) n, stream);
        } catch (const std::exception& e) {
            err = std::string("score: the engine's head failed: ") + e.what();
            return false;
        }

        // 3. The training score: log p(target) over the WHOLE vocabulary.  This is the only arithmetic here that
        //    generation has no reason to do; it is a reduction over the engine's own logits.
        if (!score_rows(logits, (int) n, (int) g.n_vocab, g.n_vocab, tgt + pos0 + r0, lse, lp, stream)) {
            err = "score: the log-probability kernel failed";
            return false;
        }
        if (cudaMemcpyAsync(out_rows + pos0 + r0, lp, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost,
                            (cudaStream_t) stream) != cudaSuccess) {
            err = "score: copying the row log-probabilities to the host failed";
            return false;
        }

        if (debug && head != nullptr) {
            // A/B: the decode head's single-column `NativeHead::run` against column 0 of this multi-column head.
            // They must agree: both are native_mmvq over the same blocks, and a multi-column launch is bitwise the
            // single-column launch.  A difference means the SCORE path is not the engine's inference head.
            float* dref = nullptr;
            if (cudaMalloc(&dref, (size_t) g.n_vocab * sizeof(float)) == cudaSuccess) {
                std::string e2;
                if (head->run(mixed, dref, stream, e2)) {
                    // The head's kernels are queued on `stream`, not the legacy default stream.  A blocking
                    // cudaMemcpy only orders the default stream, so without this wait it can read dref before
                    // the launch completed (all zeros) and report a difference that is not there.
                    cudaStreamSynchronize((cudaStream_t) stream);
                    std::vector<float> ref((size_t) g.n_vocab), mine((size_t) g.n_vocab);
                    cudaMemcpy(ref.data(), dref, (size_t) g.n_vocab * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(mine.data(), logits, (size_t) g.n_vocab * sizeof(float), cudaMemcpyDeviceToHost);
                    double worst = 0.0;
                    int64_t wi = -1;
                    for (int64_t v = 0; v < g.n_vocab; ++v) {
                        const double d = std::fabs((double) ref[(size_t) v] - (double) mine[(size_t) v]);
                        if (d > worst) { worst = d; wi = v; }
                    }
                    std::fprintf(stderr, "score dbg: head A/B col 0: worst |decode - score| %.6g at id %lld "
                                         "(decode %.4f, score %.4f)\n", worst, (long long) wi,
                                 wi >= 0 ? (double) ref[(size_t) wi] : 0.0,
                                 wi >= 0 ? (double) mine[(size_t) wi] : 0.0);
                } else {
                    std::fprintf(stderr, "score dbg: head A/B failed: %s\n", e2.c_str());
                }
                cudaFree(dref);
            }
            std::fflush(stderr);
        }
    }
    ms_head += now_ms() - t0;
    scored += T - (skipped - skipped_before);
    return true;
}

// ------------------------------------------------------------------ public

Scorer::Scorer() : impl_(std::make_unique<Impl>()) {}
Scorer::~Scorer() = default;

void Scorer::set_keep(const uint8_t* keep, int64_t n) {
    Impl& s = *impl_;
    if (keep == nullptr || n <= 0) {
        s.keep.clear();
        return;
    }
    s.keep.assign(keep, keep + n);
}

void Scorer::clear_keep() { impl_->keep.clear(); }

int64_t Scorer::skipped() const { return impl_->skipped; }

bool Scorer::attach(Prefill& pf, const core::WeightTable& wt, const core::ModelGeometry& g,
                    const core::NativeHead* head, void* stream, std::string& err) {
    Impl& s = *impl_;
    s.release();
    s.pf = &pf;
    s.wt = &wt;
    s.g = g;
    s.stream = stream;
    s.head = (head != nullptr && head->loaded()) ? head : nullptr;

    s.wo = wt.find("output.weight");
    s.wn = wt.find("output_hc_norm.weight");
    s.wd = wt.find("output_hc_down.weight");
    s.wu = wt.find("output_hc_up.weight");
    if (s.wo == nullptr) { err = "score: output.weight is missing"; return false; }
    if (s.wblocks() == nullptr || s.wtype() < 0) {
        err = "score: output.weight has no GGUF blocks and no native head (the engine must run with --native)";
        return false;
    }
    if (!kernels::native_mmvq_supported(s.wtype())) {
        err = "score: the engine's native head type " + std::to_string(s.wtype()) +
              " has no MMVQ kernel (the decode head cannot run either)";
        return false;
    }
    if (s.wn == nullptr || s.wd == nullptr || s.wu == nullptr) {
        err = "score: an output_hc_* weight is missing";
        return false;
    }
    if (s.wn->kind != core::WeightKind::F32 || s.wd->kind != core::WeightKind::Bf16InF32 ||
        s.wu->kind != core::WeightKind::Bf16InF32) {
        err = "score: the output_hc_* weights have the wrong engine forms";
        return false;
    }
    if (!kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr)) {
        err = "score: this geometry has no multi-token hyper-connection read";
        return false;
    }

    // THE ONE layout.  Fixed, sized from the head's own launch bounds rather than from free VRAM, owned for the
    // Scorer's lifetime.  Nothing here asks the driver how much memory is left or reserves a private floor: the
    // engine's expert cache and --vram-reserve-mib own the card, exactly as they do for a GEN request.
    s.cols = std::max<int64_t>(1, std::min<int64_t>(kHeadColsMax, env_i64("STRATA_SCORE_ROWS", kHeadColsMax)));
    const int64_t C = s.cols, V = g.n_vocab, K = g.n_embd, H = g.hc, L = g.hc_lr;
    const size_t q8 = kernels::native_q8_1_bytes((int) K, (int) C);
    const size_t bytes = (size_t) C * K * sizeof(float)                       // mixed
                       + (size_t) kernels::kFusedGrMaxT * H * K * sizeof(float) // xn
                       + (size_t) C * (L + H + H) * sizeof(float)            // lo, rs, inj
                       + q8                                                    // xq
                       + (size_t) C * V * sizeof(float)                       // logits
                       + score_rows_scratch_bytes((int) C)                    // lse
                       + (size_t) C * sizeof(float);                          // lp
    s.scratch_bytes = bytes;
    if (!dev_alloc((void*&) s.mixed, (size_t) C * K * sizeof(float), "the mixer output", err) ||
        !dev_alloc((void*&) s.xn, (size_t) kernels::kFusedGrMaxT * H * K * sizeof(float),
                   "the mixer's stream scratch", err) ||
        !dev_alloc((void*&) s.lo, (size_t) C * L * sizeof(float), "the mixer's bottleneck", err) ||
        !dev_alloc((void*&) s.rs, (size_t) C * H * sizeof(float), "the mixer's norms", err) ||
        !dev_alloc((void*&) s.inj, (size_t) C * H * sizeof(float), "the mixer's injections", err) ||
        !dev_alloc((void*&) s.xq, q8, "the head's Q8_1 activations", err) ||
        !dev_alloc((void*&) s.logits, (size_t) C * V * sizeof(float), "the head's logits block", err) ||
        !dev_alloc((void*&) s.lse, score_rows_scratch_bytes((int) C), "the log-probability scratch", err) ||
        !dev_alloc((void*&) s.lp, (size_t) C * sizeof(float), "the row log-probabilities", err)) {
        s.release();
        return false;
    }
    s.args.resize((size_t) C);

    char line[256];
    std::snprintf(line, sizeof line,
                  "the engine's native head (type %d, %zu MiB of GGUF blocks the engine placed), "
                  "a %lld-column head block, %zu MiB of scorer scratch laid out once - no VRAM probe, no floor",
                  s.wtype(), (size_t) (s.head != nullptr ? s.head->weight_bytes() >> 20 : 0),
                  (long long) C, bytes >> 20);
    s.note = line;
    return true;
}

bool Scorer::run(const int64_t* tokens, int64_t n, const int32_t* targets, float* out, std::string& err) {
    Impl& s = *impl_;
    if (s.pf == nullptr || s.logits == nullptr) { err = "score: not attached"; return false; }
    if (n <= 0 || tokens == nullptr || targets == nullptr || out == nullptr) {
        err = "score: nothing to score";
        return false;
    }
    // The prompt path's buffers are either its own or lent expert-cache slots.  Over a loan the ROWS are not ours to
    // trust: measured on this rig (4090, Swift IQ3_XXS, --prefill auto), an 8-token SCORE sums to -79.39 over a
    // 2,485-slot loan, to -56.60 over a 201-slot one and to -56.11 over the path's own 8192-token buffers (the
    // Hugging Face teacher-forced reference is -56.64) - and the engine's own GEN reads non-finite logits over the
    // same loan, so this is the prompt path, not this file.  Say it once, with the way out, instead of failing.
    if (s.pf->borrowed() && !s.warned_loan) {
        s.warned_loan = true;
        std::fprintf(stderr,
                     "strata score: the prompt path's buffers are lent expert-cache slots (a %lld-token layout).  "
                     "Scores over such a loan can be wrong and the engine's own prompt reads can return non-finite "
                     "rows.  Run the engine with --no-prefill-borrow for scores a training step consumes.\n",
                     (long long) s.pf->chunk());
        std::fflush(stderr);
    }
    if (!s.ensure_targets(n, err)) return false;
    if (cudaMemcpyAsync(s.tgt, targets, (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice,
                        (cudaStream_t) s.stream) != cudaSuccess) {
        err = "score: uploading the target tokens failed";
        return false;
    }
    s.out_rows = out;
    s.scored = 0;
    s.skipped = 0;
    s.ms_head = 0;

    auto previous = s.pf->on_chunk;
    s.pf->on_chunk = [&s](const float* R_rows, int64_t T, int64_t pos0, std::string& e) {
        return s.score_chunk(R_rows, T, pos0, e);
    };
    const bool ok = s.pf->run(tokens, n, 0, err);
    s.pf->on_chunk = std::move(previous);
    // No release here: the fixed working set belongs to the Scorer, not to a run.  Rebuilding device allocations
    // eight times per zeroth-order step was the training scorer competing with the engine's own scheduler.
    if (ok && cudaStreamSynchronize((cudaStream_t) s.stream) != cudaSuccess) {
        err = "score: the device did not finish the run";
        return false;
    }
    if (!ok) return false;
    // A NaN here is never this path's arithmetic: the head runs over rows the prompt path handed over, so a gap
    // means the prompt path itself did not produce a finite residual.  Say so with the position rather than handing
    // a training loop a sum that poisons the whole step.
    for (int64_t t = 0; t + 1 < n; ++t) {
        if (!std::isfinite(out[t])) {
            err = "the prompt path's residual was not finite at position " + std::to_string(t) + " of " +
                  std::to_string(n - 1) + " scored (" + std::to_string(s.scored) + " rows came back)";
            return false;
        }
    }
    return true;
}

int64_t Scorer::scored() const { return impl_->scored; }
double Scorer::ms_head() const { return impl_->ms_head; }
int64_t Scorer::rows_per_block() const { return impl_->cols; }
const std::string& Scorer::note() const { return impl_->note; }

}  // namespace strata::prefill
