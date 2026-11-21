// include/strata/prefill/score_rows.hpp - the training score over the engine's head logits.
//
// The engine's head (`NativeHead::run` in decode, `native_mmvq` in the verify window and in
// src/prefill/score.cpp's SCORE path) writes one row of FP32 logits per position.  Generation only needs the
// sampler's pick; a zeroth-order training step needs log p(x_t | x_<t) at every position instead.  `score_rows`
// is that reduction and nothing else:
//
//   score_rows      log p(target) = logits[target] - logsumexp(logits[row, :]) over the WHOLE vocabulary.  The
//                   log-sum-exp is a single online pass (the max and the scaled sum travel together), so the row is
//                   read once; `targets[row] < 0` marks a position with nothing to score and writes NaN.
//
// The numbers are the engine's own: the same GGUF head blocks, the same Q8_1 activations, the same FP32 logits
// that decode generates from.  No separate weight image and no second projection live in this path.
#pragma once

#include <cstdint>

namespace strata::prefill {

/// Device scratch `score_rows` needs: rows x 2 floats (one (max, sum) pair per row).  Zero for invalid sizes.
size_t score_rows_scratch_bytes(int rows);
/// `rows` rows of `stride` floats starting at `logits`; `targets` (device, `rows` int32).  `out` (device, `rows`
/// floats) receives log p(target).  Every row must hold at least `vocab` finite logits.
bool score_rows(const float* logits, int rows, int vocab, int64_t stride, const int32_t* targets, void* scratch,
                float* out, void* stream);

}  // namespace strata::prefill
