// include/strata/prefill/score.hpp - teacher-forced scoring through the engine's inference path.
//
// WHAT THIS IS FOR.  MeZO (and any zeroth-order method) needs f(theta) for one fixed token sequence: the sum of
// log p(x_t | x_<t) over every position, for two weight vectors theta +/- eps.  Generation cannot answer that: it
// reports the log-probability of the token it SAMPLED, after temperature and truncation, and nothing for the prompt
// positions.  This is the other thing - a raw, full-vocabulary log-softmax score for a sequence someone else chose.
//
// HOW IT STAYS THE ENGINE'S INFERENCE.  `run` advances the session through `Prefill::run`, the same 48-layer prompt
// path a request uses: the expert cache decides residency, the LoRA tables are applied layer by layer where those
// layers live, and the KV pools / GDN recurrence / PLE history / indexer advance exactly as for a prompt.  Nothing
// about the engine's scheduling changes because a SCORE arrived.  `Prefill::on_chunk` hands each chunk's final
// multi-stream residual rows to a callback; the head here is the engine's OWN head projection
// (`native_quantize_q8_1` + `native_mmvq`, the decode and verify-window kernels) over the GGUF blocks `NativeHead`
// already loaded and placed.  No weight copy, no FP16 image, no second GEMM.  `score_rows` then reduces those
// logits to log p(target) - the one arithmetic step generation has no reason to do.
//
// WHAT THIS FILE DOES NOT DO: schedule memory.  The head's fixed working set (one small row block of scratch) is
// laid out ONCE in `attach`, the way `Prefill::init` lays out a session's buffers once.  It never queries free VRAM,
// never reserves a private floor, never halves a block to fit, and is never taken from / returned to the expert
// cache per run.  The engine's arena, expert tier, VRAM reserve and LoRA tables own the card; a SCORE on a pack
// that fills it behaves like a GEN.  Earlier revisions did all of the above (a 1.27 GiB FP16 weight image, a
// cudaMemGetInfo budget with a 768 MiB floor, twelve per-run cudaMallocs) and also scored with different weight
// and activation formats than the decode head generated with.
//
// The state it consumes.  `run` advances the session exactly as a request's prompt would, so a Scorer is not
// re-entrant with a live conversation: a caller gives it a session it can spend.
//
// SCOPE - there are TWO unrelated "score"s in this tree.  `RouterLookahead::stats_score` / `warm_scored`
// (src/core/expert_source.cpp) score expert-prefetch predictions and decide which experts the GPU cache holds; the QSA
// block scorer (src/core/layer.cpp) selects attention blocks.  Those are the engine's expert scheduling and MUST
// NOT be touched.  This file is only the TRAINING scorer: it includes no expert_source / expert_cache header, and
// its only caller is generate.cpp's SCORE command.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace strata::core {
class ExpertCache;
class ExpertSource;
class NativeHead;
struct ModelGeometry;   // struct, and the spelling has to match: MSVC mangles a class tag differently
struct SessionState;
class WeightTable;
}  // namespace strata::core

namespace strata::prefill {

class Prefill;

class Scorer {
public:
    Scorer();
    ~Scorer();
    Scorer(const Scorer&) = delete;
    Scorer& operator=(const Scorer&) = delete;

    /// Bind to an already-initialized prompt path and the weights it reads.  `pf` must outlive the Scorer and must
    /// not be running anything else.  Its `on_chunk` is taken over for the duration of `run` and restored after.
    /// `head`: the engine's native output projection, whose GGUF blocks and placement this scorer reuses; null
    /// falls back to the weight table's own blocks (still the engine's bytes).  Laid out ONCE here: a fixed head
    /// block of scratch, owned for the Scorer's life, with no dependence on free VRAM.
    bool attach(Prefill& pf, const core::WeightTable& wt, const core::ModelGeometry& g, const core::NativeHead* head,
                void* stream, std::string& err);

    /// Score positions [0, n) of `tokens` against `targets`: `out[t]` = log p(targets[t] | tokens[0..t]).  `targets`
    /// and `out` are HOST arrays of n entries; a negative target marks a position with nothing to score and `out`
    /// receives NaN.  For next-token training `targets[t] = tokens[t+1]` and the last position is negative.
    bool run(const int64_t* tokens, int64_t n, const int32_t* targets, float* out, std::string& err);

    /// The rows the next `run`s must actually score: `keep[t] != 0` scores position t, `0` skips it (the mixer and
    /// the head do not run there, `out[t]` comes back 0.0), and a position past `n` counts as kept.  This is not an
    /// approximation: a zeroth-order step's objective is a SUM over the tokens a selection kept, so a position
    /// outside it contributes exactly 0 to f(theta) whatever its log-probability is.  The head is what makes a long
    /// read expensive - at 128K context it walks 16,314 blocks of 8 rows over the whole vocabulary, 15.4 s on this
    /// rig - while the answer a selection keeps is a few hundred tokens.  `n <= 0` clears the mask.
    void set_keep(const uint8_t* keep, int64_t n);
    void clear_keep();
    /// Positions skipped by that mask in the last `run` (0 without one).
    int64_t skipped() const;

    /// Positions scored so far in the last `run`, and the time the head's own work took (the prompt path's time is
    /// `Prefill::stats()`).
    int64_t scored() const;
    double ms_head() const;
    /// Head columns per launch (the fixed block size).  STRATA_SCORE_ROWS lowers it for diagnosis.
    int64_t rows_per_block() const;
    /// Which head this scorer is running and how much scratch it laid out once.  The caller prints this once.
    const std::string& note() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::prefill
