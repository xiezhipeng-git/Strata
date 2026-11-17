# Training support: Strata as the rollout engine

## What this branch ships

**Layout.** Two sibling top-level trees: `training/` holds the implementation,
its tests (`training/tests/`), the one-off probes (`training/bench/`), the
component docs and the checked-in evaluation evidence; `examples/training/`
holds the runnable sample inputs, the demo transcript and sample outputs.
Upstream has no `examples/` tree, so this PR introduces it; everything else
follows upstream's split-by-responsibility convention (`tools/`, `bench/`,
`serve/test_*.py`, `packaging/README.md`). Runtime asset paths are resolved by
`training/paths.py` from `__file__`, never from the caller's cwd. Commands below
assume the repository root, so sample paths are written
`examples/training/<name>`.

`training/` is cut into layers, and the three branches carry different subsets of them —
the same code everywhere, only the files differ. Method availability is read off the disk
(`algos.available()`), so `--list` tells you what this tree can actually run; asking for a
method whose script is absent exits before anything is loaded and names the file it wanted.

| branch | adds | methods |
|---|---|---|
| `feat/train-mezo` | shared numpy/engine layer, `mezo/`, the engine-side changes it needs, data/eval tooling | `mezo` only — no torch anywhere |
| `feat/train-backforward` | that, plus `backward/`, `ce.py` / `lora.py` / `update_rules.py`, `backward/hf_ref.py` | both |
| `feat/train_support` | the union of the two (default PR target); all three trees carry the retention reports | both |

The zero-order line is the base: `train.py → mezo/mezo.py` closes over `objectives.py` and
`mem_guard.py` only, so its suite passes with `torch` unavailable (243 tests, 2 of them skipped because the
branch has no backprop layer; measured by shadowing the module rather than by trusting a grep). The backprop layer is additive — it
imports the shared layer, never the other way round.

Inside the layers, the cut follows "what does this code know about":

| directory | knows about | must not know about |
|---|---|---|
| `common/` | data files, chat template, token/text encoding, LoRA GGUF I/O, engine transports, lr schedule, evaluation judgement | how Δθ is computed |
| `zo/optim.py` | θ, `loss_fn(th) -> float`, lr, rng | engine, SCORE, batches, datasets, metrics |
| `mezo/mezo.py` | engine-side orchestration: four objective classes (each one `loss_at`), the loop, eval logging, CLI | the estimator's math (`_zo_step` and `lr_at` are one-line forwards) |

`common/chat.py::prompt_spec` is the single prompt contract — messages plus which template
variables get passed (and, just as important, which do not: `enable_thinking=None` ≠ `False`).
Both lines feed their own tokenizer, but the messages and kwargs come from that one function,
so "train on the no-thinking template, evaluate on the thinking one" cannot happen again.
`common/gate.py` holds the evaluation judgement with the forward pass injected
(`name_logprob`, `answer_ce`, `gen`), so the backprop line's torch backend and the zero-order
line's engine backend answer the same question the same way.

A fix about data, rendering, probing or judging belongs in `common/` and then applies to both
lines by construction; a fix about how parameters move belongs in `zo/optim.py` and is
verified there against a bit-for-bit parity test, not by re-running a training job.

**Install first.** `training/` is outside the engine's install path: `requirements.txt`
covers the zero-order line (it needs nothing extra), and backprop needs
`requirements-train.txt` — torch from the CUDA index, plus `triton-windows` on Windows
because upstream triton has no win32 wheel. Platform table, per-package reasons and the
fallbacks are in [INSTALL.md](INSTALL.md); `python3 platform_check.py --deps` checks the
machine you are on and prints the commands.

## The architecture: three layers, two axes

Two axes describe **one training run**; a third layer decides **whether a run should happen
at all**. Keeping them apart is what lets `--algo` mean the same thing on both methods, and
lets the sample pool grow while the model keeps serving.

| layer | question | code | entry |
|---|---|---|---|
| **L1 run** | what is measured × how the parameters move | `algos.py` | `--algo {sft,dpo,grpo,luspo} --method {mezo,backward}` |
| **L2 grade** | should this group be trained, and on which line | `coldstart.classify()` | `--algo auto --rollouts …` / `--algo auto --pool …` / `coldstart.py` |
| **L3 build** | a group with no correct sample: what serves as the positive | `coldstart.build_cold()` | decided by L2, not exposed as a flag |

**The two axes are orthogonal, and only one of them is a free choice.** `--algo` picks the
objective (`objectives.py` — one file, numpy and torch paths cross-checked in
`test_objectives.py`); `--method` picks how θ moves. They are not a primary/secondary pair,
and their names never merge into a third algorithm: `mezo × luspo` is a description, not a
method, and `dpo luspo` is not a legal input. `--algo auto` is a dispatcher, not an
objective: it has no loss and no lr, `method_of()` refuses to treat it as one, and its
output is a `route_plan.json` for `bench/run_plan.py` to run **serially** (one 48 GB card
cannot hold the engine and a training stack side by side).

**`method` stops being a preference once the engine is also serving.** `backward` unloads
the engine and double-resides (weights + activations); `mezo` drives the resident engine
through `SCORE` / `LORA`. On one card, "train while serving" has exactly one answer, and it
is `mezo` — `backward` belongs to the offline half (building the positives a cold start
needs).

**"No correct answer" is a property of a group, not of a dataset.** `classify()` grades
each `(prompt, K samples)` group into four buckets:

| group | verdict | what happens |
|---|---|---|
| all correct | `graduated` | dropped from the pool — sampling it again burns GPU for nothing |
| mixed | `rl` | `luspo` — the only bucket with gradient (advantage is group-normalised, so all-wrong means `A ≡ 0`: **no step at all**, not a slow one) |
| mostly truncated | `truncated` | raise `--max-new` and re-sample; a truncated run's reward is *unknown*, not wrong |
| all wrong | `cold` | cold start: build pairs, then `dpo` or `sft` by length ratio (`< 1/3` ⇒ `dpo_margin` averages the negative side into 1/8 ⇒ it degenerates into a slower SFT) |

No positive sample and no full gold solution ⇒ the group is dropped
(`gold_not_a_trajectory`). That is the honest answer, not a training value. Because the
verdict is per group, asking the user "does my dataset have answers?" would always be
answered wrong — which is why the dispatcher exists and why the user never picks per round.

**Steps are derived from the data, never a constant.** `--steps N` wins if given; otherwise
`steps = epochs × rows` (rows = samples, preference pairs, or rollout groups — whichever
the algorithm actually iterates), with `--epochs` as the knob that stays comparable across
dataset sizes. Where no measured epoch count exists, the default is **one pass** and the
log says so. The old `450 if mezo else 40` is gone: on a cold-start set of four pairs, 40
steps meant forty passes.

### The inference side is not modified

Nothing in this branch changes the engine's request path, and nothing here needs a second
copy of the weights. A training run is two commands the engine already answers — `SCORE`
(log p for a sequence the caller chose) and `LORA` (swap the adapter's tables in place) —
sent to the same serve process that answers `/v1/chat/completions`.

What a serving engine must be started with, and why it cannot be decided per request:

| flag | why |
|---|---|
| `--lora <a zero adapter>` | `LORA` replaces the contents of the tables the engine started with; with no adapter loaded there is nothing to update and the swap is refused. `python3 mezo/zero_adapter.py --gguf <shard> --out zero-r8.gguf` writes that file: A = B = 0, so the values served through it are the base model's (`y + s·B(Ax) = y`; the only float a zero add can still change is −0 → +0), and the rank it declares is the rank every later step must keep. |
| `--no-prefill-borrow` | over a loan, the prompt path's rows are not to be trusted: an 8-token `SCORE` sums to −79.39 over a 2,485-slot loan against −56.11 on the path's own buffers (the HF reference is −56.64), while the signal a zero-order step reads is ~0.55 nats. That is a property of the borrowed layout, not something training introduced — the engine's own prompt reads can come back non-finite over the same loan. Changing it inside the prompt path is separate work; until then the layout is chosen at startup. |

The cost of the second flag is real and is not hidden here: with no loan the prompt path owns a
buffer the size of its **chunk** for the whole session, and the expert cache is sized after it.
Measured on one 48 GB card with the same IQ3_XXS weights: **4,136 MiB at 8192, 2,596 at 4096,
1,440 at 1024, 464 at 512, 368 at 256** — the 256 figure is GEMM scratch, not the chunk. Going
from 8192 to 512 hands the cache 1,652 more slots (16,518 -> 18,170). What does *not* change is
the code path: no unloading, no restart, no second resident copy, and batch slots still serve
inference (`--parallel`). A swap drops every cached K/V — the update granularity is
**one round**, not one sample — and `set_lora`
refuses while batch slots decode, because those slots decode through the tables being
replaced. Sampling can use the batch slots; the swap cannot.

That reservation **comes from the data, not from a constant**. `--prefill N` is the prompt
*chunk size*, not a length limit: a longer sequence still reads whole, in ceil(n/N) chunks
(`request_chunk = min(N, ceil256(n))`), so N only decides two things — how many chunks one
sample reads in, and how much is reserved. `mezo.py` takes the longest sample in the data
(rounded up to the 256 grid, clamped to 256..8192) instead of the old constant 8192: 300-600
token math data never reaches 8192, so those 4 GiB were held for nothing (the engine's own
`--prefill auto` only goes down to 2048 = 1,826 MiB without a loan; the data can do better),
while a sequence longer than the chunk only pays extra passes (slower, never truncated). Long-context data
(128K) is the one case that wants `--prefill-chunk` raised by hand: at the 8192 cap a
128K sample reads in 16 chunks (slower, still correct); raising it trades reservation for passes.

### `SCORE` is the engine's own inference path, head included

A `SCORE` is not a second model. `src/prefill/score.cpp` advances the session through
`Prefill::run` — the same 48-layer prompt path a request uses, with the engine's expert
cache deciding residency, `lora_apply` running where each layer lives, and the KV / GDN /
PLE / indexer state advancing normally. `Prefill::on_chunk` hands over each chunk's final
residual rows; the head projection is the engine's own `native_quantize_q8_1` +
`native_mmvq` over the GGUF blocks `NativeHead` already loaded and placed — the same
kernels decode and the verify window run. The only arithmetic training adds is
`score_rows`, the reduction `log p(target) = logits[target] - logsumexp(logits[row, :])`.

It also does not schedule memory. A fixed 8-column head block is laid out **once** in
`Scorer::attach`, the way `Prefill::init` lays out a session's buffers once. No
`cudaMemGetInfo` probe, no private 768 MiB floor, no halving a block to fit, no 1.27 GiB
FP16 copy of `output.weight`, and no per-run `cudaMalloc` / `cudaFree` against the expert
cache. The card belongs to the engine's arena, expert tier, `--vram-reserve-mib` and LoRA
tables, for `SCORE` exactly as for `GEN`.

Measured on one 48 GiB card (125B IQ3_XXS + NF4, `--no-prefill-borrow`, one engine build):

| check | result |
|---|---|
| `SCORE` head vs decode `NativeHead::run`, column 0 over 27 head blocks (`STRATA_SCORE_DEBUG`) | worst absolute difference **0** (bitwise) |
| `GEN 1` vs `SCORE`, same 69-token sequence | 119.1 ms vs **119.6 ms**, **1.00x** |
| fixed scorer scratch | **8 MiB**; `STRATA_SCORE_ROWS=1/3/8` gives 1/3/8 MiB and a bitwise-identical `SSUM` of −799.411996 |
| peak VRAM over 10 consecutive `SCORE`s (`nvidia-smi`) | new +**60 MiB** vs old +**174 MiB**, each over its own baseline |
| full suite | **495 passed / 0 failed** (`training/tests`). The two `test_bench_limits.py` watchdog failures caused by `huggingface-hub==2.2.0` were fixed properly: `bench_fullctx.py` now loads its model-side imports on first use, so a scheduler-only watchdog no longer pulls in `transformers` at all. No environment downgrade, no skipped test. |

Two things this does **not** change. Absolute scores move — old head −799.815137 vs new
−799.411996 on the same 69 tokens (mean per-position |delta| 0.021 nats) — because the old
path scored a dequantized FP16 weight image and the new one scores the engine's Q5_K /
Q8_1 path. MeZO consumes differences inside one build, so re-baseline before comparing an
adapter scored on the old binary. And the forward count belongs to the algorithm, not the
head: `dirs 4 x +/-` is **8 full forwards** (~957 ms), while the MeZO paper's `dirs 1` is
**2** (~257 ms). `dirs 1` is already the default in `mezo.py` — the identity runs that took
hours all passed `--dirs 4` explicitly. The 10-run / 22 324-step table in the PR was timed on
the earlier binary and is **not** re-timed here. See the next section for what `dirs 1` buys.

Not measured: a 125B pack that actually fills the card. On this 4090 the old path still had
~1.4 GiB free even with `--vram-reserve-mib 100`, and WDDM makes a cross-process `torch`
allocation an unreliable way to fake a full card, so "old path `ERR`s / new path scores"
has not been reproduced. The new fixed 8 MiB scratch cannot hit that `ERR` by construction;
that is a claim about the code, not a measurement.

### MeZO in under half an hour: the speed ruler and the gate

Same engine, same 69-token identity row, one process (`idrun/speed_ruler.py`):

| path | throughput | vs decode |
|---|---:|---:|
| `GEN` decode | 138.3 tok/s (7.229 ms/tok) | 1.00x |
| one `SCORE` prefill | 587.4 tok/s | **4.25x** |
| `dirs 4` step (8 forwards) | 66.8 tok/s | 0.48x |
| `dirs 1` step (2 forwards) | 267.1 tok/s | **1.93x** |

`SCORE` is a parallel prompt pass, not token-by-token decode. Training was slow because eight
forwards were spread over one 69-token row, not because `SCORE` was slow. The next largest cost
was measurement, not training: `run()` did a full 12-row `answer_ce()` sweep at the end of every
12-step round (1.6 s x 66 = 103 s, 28% of an 800-step run). `--round-metric-every N` lowers
that frequency without touching training arithmetic; the defaults (1 in `mezo.py`, 0 in
`train.py`) reproduce the historical numbers byte for byte.

800 steps, `dirs 1`, `--round-metric-every 10`, including engine start, eight evaluation sweeps
and five scored generations: **276 s = 4.6 min** (374 s with the old sweep frequency), with the
`after:` line byte-identical. That is 200 tok/s end to end, 1.45x decode; ~6000 steps fit in a
30-minute budget. A zero-signal control (`--lr 0`, 200 steps) moves CE by +0.0000. A same-engine
A/B over the built-in rows gives training CE 3.3281 -> 3.1726 (12/12 rows improve), held-out CE
1.8685 -> 1.8640, probe logp +0.5277, and a 0.000e+00 noise floor over six repeat `SCORE`s.

**Why free generation still says the base sentence.** `idrun/gate_ab.py` scores the first token
of the appended sentence — the "gate", right after the base's own words, where the base expects
`\n\n请问` / `How can I help` / `<|im_end|>`. At theta_0 its CE is 6–19 nats and **all 12 rows
select it** with `--select-tokens excess`; prefix CE is 0.03–0.30, so the "base words + append"
construction is sound. After 800 steps the mean gate CE only moves 11.155 -> 10.435. With
`d = 2,560` (`--zo-layers 1 --rank 1`), a single-direction zero-order estimator has SNR
~1/sqrt(d) and needs on the order of `d` steps before the consistent component wins. A 400-step
configuration scan (`idrun/scan_gate.sh`) found `--select-first 1` and `--gate-weight 200` both
too slow (−0.24 / −0.42 nats), while rank 4 + last 2 layers + `lr 1e-3` + `step-max 0.1` moves
the gate **−2.28 nats in 400 steps**. CE going down is not a behaviour change: use
`--success-data` and read the `gen:` text, not just the loss.

`--zo-one-point` is an **experimental** speed knob (off by default): residual-feedback SPSA uses
one forward per step. It is fast and it does **not** converge at this working point. A real bug was
fixed first — `Mezo` is rebuilt every step, so the residual `prev_loss` state now lives on the
trainer (`EngineMezo._zo_prev_loss`, cleared per `lr_finder` probe) — after which 800 steps take
**145 s vs 276 s** (141–152 ms/step vs 278–294 ms), the expected 2x. But `pj` grows to the 1e3
range and answer-CE **rises +0.0340** (two-point: −0.1567); probe logp +0.80 vs +1.34. The
`(L(θ) − L_prev)/ε` baseline drift is amplified by `1/ε = 1000` and its variance swamps the
signal. Do **not** use it to report a result without a validated baseline (EMA, or a periodic
exact `L(θ)` refresh). Off by default, covered by `test_zo_optim.py` /
`test_zo_step_metrics.py`, and it changes no historical number.

**What 2,000 more steps and an on-policy prefix actually buy.** A 2,000-step `dirs 1` run at
rank 4 / last 2 layers / `lr 1e-3` / `step-max 0.1` (686 s, 11.4 min) drove the mean gate CE
from **11.161 → 3.211** (min 1.462, max 5.757) — the gate is effectively open — and answer-CE
3.2702 → 3.0593, yet `success` stayed **0/5 at every eval**, probe logp got **worse**
(−59.183 → −59.846), and generation collapsed into a different stock self-introduction.
Replacing the supervised prefix with the base's **complete free generation** (`onpolicy.jsonl`,
`idrun/make_onpolicy_data.py`; 12 rows, longest 273 tokens) improved the probe logp to
**+3.0933** in 800 steps / 309 s and lowered answer-CE 2.2466 → 2.1536, but `success` with a
256-token budget was still **0/5**. Together with `IDENTITY-RUN-LIMITS.draft.md` (where
`idR2_4000` reached answer-CE 0.9609, gate median 1.73 and still 0/5), this rules out step
count, rank, token selection, the gate, the data shape and estimator noise. The remaining defect
is the **teacher-forced CE objective**: exposure bias plus an unconstrained global logit bias.
Escaping it needs a different objective (scheduled sampling / on-policy sequence-level loss, or
an explicit held-out-context regulariser), not another hyperparameter run.

Data correction (2026-11-15): `BASE_SELF_ZH_B` (「你是谁」, no question mark) claimed the base
said 「…自主研发的大语言模型。」; `diag_zero.py` measured the base actually saying the same sentence
as `BASE_SELF_ZH_A` (the old text shared only 13/36 characters). It is now the measured
utterance; 11 of the 12 rows already matched the base's free generation exactly.

## Start here: `train.py`

One command, no path archaeology:

```bash
python3 train.py                        # algo sft + method mezo (default): engine forward, memory = inference
python3 train.py --dry-run              # show the final command without loading anything
python3 train.py --list                 # algorithms / methods / datasets
python3 train.py --algo sft --method backward   # same algo, backprop (exact gradient, more VRAM)
python3 train.py --algo dpo --dpo-data pairs.jsonl       # zero-order DPO
python3 train.py --algo grpo --rollouts r.jsonl          # zero-order GRPO (replayed rollouts)
python3 train.py --data jsonl:/root/my.jsonl --steps 200 --out /root/lora.npz
```

Defaults, and why they are defaults:

- **model** — the one `launcher/launch-choice.json` pins, i.e. what `Strata CN.lnk` starts.
  `train.py` imports `launcher/start_server.py`'s `stock_config()` instead of re-implementing
  the selection, so "the model I am training" and "the model I am serving" cannot drift apart.
  Windows paths are converted for WSL, and a `…-00001-of-00002.gguf` is expanded to all shards
  (passing only the first shard silently loads fewer layers).
- **data** — `identity`: the built-in "change who you are" set (it is the only task with a
  unique answer, so `logp('Strata' | '你是谁')` is a clean before/after judge). Switch with
  `--data identity:<word>`, `jsonl:<path>` or `pretrain:<npy>`; see `datasets.py`.
- **multi-turn data** — `--conv-data <jsonl>` (SFT only): each row is
  `{"messages": [...], "tools": [...]?}` in the same shape the HTTP layer takes, tool calls
  included (`assistant.tool_calls` + `role: "tool"` replies). The whole transcript is rendered
  once by the pack's own chat template and tokenized once, so the tokens that separate turns
  are the template's (`<|im_end|>` = one id, not six ordinary tokens), and every assistant turn
  is supervised — including the `<|im_end|>` that ends it. Declare `tools` only if the sampler
  passed them; the training prompt must be byte-identical to the sampling prompt. Sample:
  `examples/training/data_conv_tool.jsonl`.
- **algorithm × method** — algorithms are `sft|dpo|grpo|luspo` (what is measured); methods are
  `mezo|backward` (how parameters move, default `mezo`). `--algo auto` is the one name on
  that axis that is not an objective: it groups the samples by "is there a correct sample
  in this group" and dispatches each group to `luspo` (yes) or a cold-start `DPO` (no), so
  the user picks nothing per round — see `coldstart.py` and `python3 train.py --list`. Its
  data source is `--rollouts` (batch: K consecutive rows per prompt) or `--pool` (stream:
  append-only, grouped by prompt, row order irrelevant — see `pool.py`). Loss math lives once in
  `objectives.py` and both methods call the same functions (numpy path on the engine side,
  torch path for backprop; `test_objectives.py` cross-checks the two). `mezo` means **the
  engine's own inference is the forward pass**: it sends `LORA <file>` / `SCORE <ids>` to
  the same binary that serves, so `python3 train.py --algo sft` and `python3 mezo/mezo.py
  --objective ce` are literally the same `main()` — and `mezo.py` is a class hierarchy
  (`EngineMezo` base + one subclass per algorithm implementing only data loading and the
  loss). `hf_ref.py` (PyTorch forward) is retired from the registry; it remains as the
  score-parity reference (−56.6433 vs engine −56.6508) and is never picked silently.
- **steps** — never a constant unrelated to the data. `--steps N` wins if given; otherwise
  `steps = epochs × rows` (rows = samples, preference pairs, or rollout groups, whichever
  the algorithm actually iterates). `--epochs` is the knob to compare across dataset sizes.
  Where no measured epoch count exists (`dpo × backward`, `grpo|luspo × backward`), the
  default is **one pass over the dataset** — a lower bound, not a tuned point, and the log
  says so (`默认一遍（该方法没有实测轮次）`). Cold-start DPO usually wants more (7 is the
  zero-order working point): pass `--dpo-epochs`.

### The identity answer is "the base's own words plus one appended sentence"

The identity rows are built as `(question, base utterance, appended fact)` in
`common/data.py`, and a test asserts `answer == prefix + append` character for character. The
reason is not stylistic: `--select-tokens excess` only buys "learn the new part and nothing
else" if the new part really is the *only* part the base gets wrong. When the new fact was
inserted mid-sentence, per-token CE exploded from the insertion point (`，`=8.02, `后`=9.25)
and the character-common prefix with the base was 25/97 — the model had to rewrite the
sentence, not continue it.

Measured shape of the current set (12 rows, `ce_sweep.py`): **every** prefix ≤0.09 nats/token,
appended sentence 5.3 nats/token, `excess(>1.0)` selects 45% of answer tokens. Two rows were
wrong before: `请用一句话介绍你自己。` had 45 prefix tokens at 0.646 nats (the base actually
says "...致力于成为您真诚、有温度的思考伙伴。"), and the English append ran straight into the
period (`Lab.This version…`), which the tokenizer folds into one `'.This'` token at 17.0 nats —
that made the "cheap English gate at 4.8" a measurement artefact. `APPEND_EN` now has a leading
space and `test_shared_data.py` asserts it. **Decode the token before quoting its CE.**

What actually blocks generation is a single token per row — the **gate**: the first token of
the appended sentence (`当前` after `…大型语言模型。`, θ₀ CE 6.05–19.43, median 10.87).
Everything after it is
cheap once the model is there (1.8, 1.4, …), so on the generation path the gate is the whole
problem. Two measurements worth keeping:

- **Punctuation does not move the total.** Fourteen phrasings of the same sentence (period and
  new sentence, comma instead of period, semicolon, blank line, spaces) all cost 130–148 nats
  with a 15–18 nat maximum; only the *distribution* changes (the gate can drop to 4.3 while
  the next token rises to 9.8). Keep the strict "period, then a new sentence" form.
- **Averaging over the whole appended sentence spends the gradient on the wrong tokens.**
  `idC` (3000 steps, `excess(>1.0)`, lr 1e-4) drove the appended-sentence mean 5.30 → 3.90 and
  the training loss monotonically down, while the 12 gates went **8.70 → 9.08** and generation
  stayed byte-identical. With 233 selected tokens the gate is 5% of the objective, and the
  other 95% point in an almost orthogonal direction — and those positions are unreachable in
  generation while the gate is closed, so their gradient cannot help.

`--select-first N` keeps only the first N selected tokens of each row, which puts the gate at
33–100% of the objective (`N=3`: 36/511 tokens, gate weight 33%). Fair warning from the same
day: `N=1` collapses the signal-to-noise ratio — `pj` goes from ±0.5 to ±1…±8, and 100 steps
moved the gate loss by 0.13 (8.21 → 8.08) while `‖B‖` grew 0.014 → 1.2. Averaging over one
token is not averaging.

**`--zo-rel` self-locks at `B = 0`.** The cap is `‖Δθ‖ ≤ rel_cap·‖θ‖` *of the moving factors*,
and `‖B‖` can only grow through `‖Δθ‖` — so starting from a zero adapter the step is pinned at
`rel_cap·‖B‖`, which stays at zero. Measured: `rel-cap 0.01` held every step at 1.4e-4
(vs 6.4e-3 in a run without it) and `‖B‖` did not grow in 17 steps. From a zero adapter use
`--zo-step-max` (absolute) and leave `--zo-rel 0`; the run prints a warning when both are set.

**A gate-only curriculum opens the first door, then needs a second pass.** `idM`
(`--select-first 2`, 4000 steps, lr 4e-4) took the gate loss 7.97 → ~3.0 and the 12 gate CEs
from a median of 10.87 to 2.82; five gates fell below 0.6 (`Who developed you?` 6.62 → 0.05).
`success` was still 0/5, but generation **did change** for the first time: `你是谁？` →
"…独立开发**当前版本为 Qwen**", `What is your name?` → "…Tong**当前当前当前当前**" (a repeat
loop), `Who are you?` → "…Tongyi Lab. **This** means I am…". The gate became strong while the
tokens after it (` version` 6.2, `qwen3.8-flash-next` 8.1, `strata`) never received gradient.

`--resume` was tried as that second stage: `prepare_selection` recomputes the mask at the
**new** θ₀, so rows whose gate is now below `--select-thresh` drop out and their first N
selected tokens move on to positions 2, 3, … (widen N, e.g. 2 → 6). At 500 steps of
`--resume --select-first 6 --lr 2e-4` the answer-ce went 3.1756 → 3.0663 and logp rose
+6.8 nats, but generation was still "…有什么我可以**当前当前当前当前当前**" and `success`
stayed 0/5. A second pass on the same rank does not fix it, for the reason below. Resuming also
needs `self.theta` to be a **full-layer** skeleton: the npz stores only trained layers
(`--zo-layers 1` ⇒ one layer), while `write_adapter` writes every layer and
`swap(self.theta, ...)` would otherwise die with `KeyError: 0` *after* the engine had already
started. `main()` rebuilds the skeleton with `init_theta(layout, rank, seed)` and overlays the
restored layers; `test_resume_theta.py` pins all three behaviours.

**Rank is not the ceiling — the gate-only objective is.** At `--zo-factors b`, ΔW = B·A and
`rank 1` (A fixed) makes `A·h` a scalar, so Δlogits(h) = (U·B)·(A·h)·alpha/r always points
along **one** vocabulary vector. `bias_probe.py` confirmed the resulting context-free bias on
the `idM` adapter: CE of `当前` fell 12.09 → 6.00 (−6.09) at the trained gate but by **more**
in unrelated contexts (arithmetic 19.49 → 2.66, poetry 13.43 → 2.19, jokes 21.25 → 7.04;
mean −9.45). `test_lora_rank.py` pins the algebra (rank 1 collinear across hidden states;
rank 4 spans four directions), and the trainer prints a capacity hint for `--rank 1`.

The clean A/B disproved rank as the cause. `idR4` changed **only** `--rank 1 → 4`, with the
same data, 4000 steps, lr, dirs and `--select-first 2` (starting `answer-ce 3.2657`,
`24/517` selected — byte-identical to `idM`). It opened every gate: median gate CE
10.87 → **0.14**, max 0.37, and the gate loss 7.97 → 0.1221. But the outcome got *worse*:

| | idM (rank 1) | idR4 (rank 4) |
|---|---|---|
| gate CE, median | 10.87 → 2.82 | 10.87 → **0.14** |
| appended-sentence mean CE | 5.30 → 3.90 | 5.30 → **7.44** |
| `当前` ΔCE at the gate / unrelated | −6.09 / −9.45 | −11.97 / **−13.91** |
| answer-ce | 3.2657 → 3.1756 | 3.2657 → **4.2507** (+0.985) |
| generation | `当前当前当前…` | `当前版本版本版本…`, `This version version…` |

`bias_probe.py` on `idR4`: `当前` became easier in **100%** of unrelated contexts (mean −13.91,
larger than the −11.97 at the trained gate); `版本` −12.54 elsewhere vs −12.01 at the gate.
So removing the algebraic rank-1 restriction made the global vocabulary bias *stronger*, not
weaker. The repeat loop is not a capacity failure.

The actual cause is the objective. `--select-first N` puts **only the first N excess tokens**
into a teacher-forced CE loss and drives them to ~zero (idR4: first two appended tokens CE
0.1 / 0.0, the third 7.1, the fourth 10.0). Free decoding then reaches position N+1, for
which no loss ever existed, and falls back onto the token it just learned — the repeat loop.
Rank only changes how fast and how completely those N positions are overfitted. `--select-first`
is therefore a **gate-opening / diagnostic tool**, not a way to learn a whole appended sentence;
the mask is anchored at θ₀ and cannot by itself advance along the sequence. Any fix must keep
the full sentence in the loss while still weighting the gate — weighting, not truncation.
The failed rows are kept. Budget: d = rank·n_out, so `n ≳ 4d` grows linearly with rank;
`idR4` sampled 16000 directions = 1.56 d.

`--gate-weight K` (`--gate-n N`, default 2) is that objective. It keeps **all** `excess` tokens
in the loss and multiplies only the first N selected tokens of each row by K, then normalises
by Σw. On the corrected identity data, `K=20, N=2` selects **228/517** tokens (vs 24/517 for
`--select-first 2`) while giving the gate **70%** of the objective weight (vs 50% under
truncation and 5% under plain `excess`). Normalising by Σw keeps the loss scalar on the same
scale — vital here because zero-order is plain SGD, so the loss magnitude *is* the step size.
The two flags are mutually exclusive; with neither, `guard.gate_weights` returns `None` and
`SftMezo.loss_at` takes the original `objectives.sft_loss` path bit-for-bit.
`training/tests/test_gate_weight.py` pins all of this, including the decisive check that
changing a tail token's logp moves the weighted loss but **not** the truncated one.

`idW800` (800 steps / 1193 s, everything else identical to `idM`) verifies the direction.
Measured against a **fresh θ₀ sweep on the same ruler** (the older `ce_identity*.npz` dumps
predate the 16:5x data fix and must not be reused): the 12 gates fell on 11/12 rows, median
10.74 → 7.24 (mean ΔCE −2.513, ≈ 2× the gate-decay rate of truncated `idM`), while the
**124 high-CE appended positions that are not gates** went 10.012 → 9.107 (−0.905, 93/124
down) and the hard θ₀ CE>9 bucket fell −1.389 (69/74 down). The 286 prefix positions moved
only +0.017. Contrast `idR4`, where the truncated objective drove the appended mean 5.30 →
**7.44**. So the tail now genuinely receives gradient instead of rebounding. `success` was
still **0/5**: the gate at 7.24 is nowhere near argmax, and generation changed from a repeat
loop into a *different base-model self-introduction* — a changed generation is not a rewrite.
More steps are needed before this can be called a success or a failure; the failed row stays.

`idW4000` (4000 steps / 5918 s, same single variable) answers whether more steps were enough.
Training loss 7.7314 → 4.4820, answer-CE 3.2657 → 2.3975, probe logp −59.075 → −49.954.
The gate kept falling (median 10.74 → 7.24 → **2.92**, 11/12 rows down, 5 rows below 1.0) and
the 74 hard θ₀ CE>9 positions went 12.147 → **7.008** (−5.139, 70/74 down). `success` was
**still 0/5**. The prefix drift also started to show: the 286 θ₀ CE<1 positions rose
0.125 → 0.418 (+0.294). So the earlier "direction verified, amplitude still short" reading was
**too optimistic** — opening the gate is not the same as teaching the sentence.

Two fresh probes say where the budget actually went. `ce_w4000`'s first four appended tokens
look like `6.6/0.1/1.3/**8.4**` and `0.1/**6.1**/0.7/**8.1**`: only the gate and a couple of
cheap positions moved, the rest of the appended sentence is still at 6–9 nats. `K=20` gives the
first two tokens ~69% of Σw, so a tail token receives roughly 1/140 of a gate token's share —
its zero-order difference is buried in estimator noise. Appending the fact to the model's
*actual* free-decoding text instead of the training prefix raises the gate from 4.79 → 11.47
median (+9.18; `谁开发了你？` +17.57), i.e. teacher-forced supervision and the decoding path
are still not the same distribution. But scanning the natural breakpoints **along** that path
finds the first sentence ending at CE 0.03 / 0.12 on the two English rows — the gate is not the
bottleneck there either, the model simply keeps expanding its stock self-introduction. Under a
128-token (instead of the eval's 64) window the new fact does surface as fragments —
`当前版本已当前版本已当前版本当前版本`, `... This version has` — and immediately collapses
into a repeat loop. That is the signature of a taught *prefix* with an untaught continuation.

The follow-up was a curriculum, not more of the same: `--resume` from the opened gate and drop
`--gate-weight` to 0 so the whole appended sentence gets the gradient budget. The `train.py`
entry point was missing the `--resume` passthrough (silently restarting from θ₀);
`test_gate_weight.py::CliPlumbing` pins it. Keep the failed rows: a monotonically falling
aggregate CE hid this for 4000 steps, and only a per-token probe exposed it.

`idC1200` (resume + `K=0`, 1200 steps / 1807 s) confirms that the curriculum fixes what it was
designed to fix and nothing more. answer-CE 2.3975 → **2.0272** (~3× the decay rate of the
weighted tail), the 286 prefix positions recovered from 0.418 → **0.209** (the drift was an
artefact of gate over-weighting), and the 89 positions still above 6 nats fell 8.851 → 7.905
(−0.947, 60/89 down). The gate only drifted 2.92 → 3.18. `success` was **still 0/5**, and the
probe-word logp on the eval line went −44.484 → −50.040.

`append_tokens.py` (per-token CE with the actual BPE pieces decoded) shows why. After 5200
steps, `'-git'` is **9.4–13.2 nats on all 12 rows** (θ₀ 10.7–15.6), `' q'` is 7.0–9.1 on all
12, and `通过` / `' through'` barely moved — while `'iez'` fell 11–14, `'-flash'` / `'-next'`
~7, `'ata'` ~8. The target word is tokenised `' x'`, `'iez'`, `'hip'`, `'eng'`, `'-git'`;
`success` requires the *whole* string, so one unlearnable final piece blocks every row.
`bias_tokens.py` measures those same pieces in five unrelated contexts: `'-git'` sits at 20–25
nats there and moved only −1.27, while `当前` fell −3.83 in unrelated contexts (poetry −9.45)
but **rose** at two of the Chinese gates. That is the signature of `--zo-factors b` with
rank 1: `Δlogits(h) = (U·B)·(A·h)·alpha/r` has a direction `U·B` that is **independent of h**.
Only a whole-vocabulary bias is expressible; lifting `-git` by the ~10–15 nats needed at the
target would inject it everywhere, so the optimiser trades it away. This is an expressivity
limit of rank-1 B-only updates, not step count, weight allocation, or zero-order noise.

Escaping it needs rank ≥ 2 or a trainable `A` (so `A·h` gates the bias by context); both at
least double the zero-order dimension d, so `n ≳ 4d` and the wall-clock grow accordingly.

**`idAB800` did not test the trainable-A option.** Resuming `idC1200` with `--zo-factors ab`
(800 steps / 1186 s) made everything worse: answer-CE 2.0272 → **3.5262** (worse than θ₀'s
3.2657), probe logp −41.286 → −48.391, generation back to a repeat loop, and `best` stopped
improving after the first eval (stall 57). `pj` jumped from ±1 to **±30–140** at step 0. That
is a conditioning failure, not a refutation: `Mezo.sample()` draws `z ~ N(0,I)` with one
absolute `eps` for both factors, but at this θ `‖B‖ ≈ 11` while `‖A‖ ≈ 1`, so A's *relative*
perturbation is ~10× larger and its loss derivative (which carries a factor `‖B‖`) is ~10×
larger again. The difference is swamped by the A direction, whose update is then multiplied by
B ≈ 11 in the logits — positive feedback. A usable `ab` needs a per-factor step scale (a
diagonal preconditioner), not a smaller `eps`. It is left alone; the rank route needs no
algorithm change.

`widen_rank.py` lifts the `idC1200` checkpoint without moving the model: append
`N(0, 1/√n_in)` rows to A and zero columns to B (so `B₁·A₁ = B₀·A₀` bit-for-bit), and scale
`alpha` 16 → 32 because the engine uses `s = alpha/rank` — otherwise the already-learned half
of ΔW is silently halved and the A/B is no longer single-variable. Its 20-step smoke
reproduces `idC1200`'s endpoint verbatim (logp −41.2855, CE 2.0272, 192/517 selected) with
d 2560 → **5120** and `pj` still ±1–3.

**`idR2_4000` (4000 steps / 6141 s) breaks the expressivity limit and then hits a different
wall.** `'-git'` — stuck at 9–13 nats on all 12 rows for 5200 rank-1 steps — fell 14.15 →
**0.64** (−13.51) on the first row and below 1.0 on all 12; appended-sentence mean Δ is
−3.9…−4.6 per row with only 2–4 positions above 3 nats left. answer-CE 2.0272 → **0.9609**,
gate CE median 10.74 → 3.18 → **1.73**, probe logp −41.286 → −34.862. Every training-side
number is the best of all eleven runs. `success` is **still 0/5** at all 20 eval points, and
generation degenerates through `当前版本当前版本…` (step ~1000) into
`str str str…` / `stricken stricken…` / `stragely stragely…` — `str` being the first BPE piece
of `strata`. No intermediate checkpoint is kept (`init`/`plus`/`final` only), so an
early-stopping "sweet spot" is not supported by the data.

Three probes say what rank 2 actually learned. `bias_tokens.py` in five unrelated contexts:
`'-git'` moved **−11.70 on average** (jokes −15.86, code −13.82), `' through'` −8.72,
`' q'` −7.91, `'iez'` −7.18, `当前` −5.55. The 286 θ₀ CE<1 prefix positions drifted
0.125 → 0.209 → **0.445**. And `onpolicy_probe.py` / `gate_curve.py` on the **free-decoding
path**: teacher-forced gate median 4.82 → on-policy **13.58** (Δ +10.95), best breakpoint
along the whole 128-token path still 13.34 (4 of 5 rows never below 11). So the model did not
learn "say this after an identity question"; it lifted those pieces across the vocabulary and
memorised the sentence under the teacher-forced prefix. Free decoding leaves that prefix,
enters a `str` loop, and the gate CE there is ~11 nats higher.

This is the **same failure as the exact-gradient backward control** (CE 1.011, repeat loop,
capability gate rejected for `cap_mean` +0.596), now reproduced with a perfect teacher-forced
fit. It therefore rules out estimator noise, optimiser, rank, step count and weight allocation:
the remaining defect is the teacher-forced CE objective itself — exposure bias compounded by
an unconstrained global logit bias. Escaping it needs a different objective (on-policy /
scheduled-sampling supervision, or an explicit held-out-context regulariser), not another
hyperparameter step. That is unmeasured; the failed rows stay and the identity rewrite is
**not** reported as working.

**The backward route is a working control, not a second failure.** A 24-step exact-gradient
backward SFT run on the same corrected data (rank 16, Adam, full `excess` selection) drove
training-row CE 3.137 → 1.011 and held-out identity logp +45.36 nats — the gradient chain,
optimiser and adapter capacity all do their job. It still generated
`qwen-next-flash-next-next…` / `strata-next-next…`. Because the backward estimator is *exact*,
that repeat loop **rules out the zero-order estimator as the cause** and points at the
teacher-forced CE objective shared by both routes. (Both used `excess`: full-sentence on the
backward side, truncated on the zero-order side.) Its capability gate **rejected** the final
snapshot: held-out `cap_mean` 0.742 → 1.337 nats/tok (Δ +0.596 vs a 0.5 tolerance), generation
health 1/2, best accepted snapshot restored at step 16 (idΔ +44.139, capΔ −0.011). A
monotonically falling CE is not evidence of a *good* rewrite. The control is what motivates
changing the objective on the zero-order route — see `--gate-weight` above — rather than
concluding that identity rewriting is out of reach for MeZO.

These scripts are a **worked example, not a framework**: a minimal, self-contained
SFT / GRPO loop where the Strata engine produces the trajectories (exact token ids,
the engine's own logprobs) and a student model — the same weights, quantized to
self-implemented NF4 experts + LoRA — does the forward/backward in PyTorch.
No TRL, no unsloth, no training framework: the point is to show the two contracts
a real trainer needs from Strata, and to serve as the reference for wiring bigger
frameworks onto the engine later.

## Recipes on the 125B model (one 48 GB card)

One entry point, six ways to use it. Measured on the stock weights (48 MoE layers x
512 experts, hidden 2560; the experts in a 60 GB NF4 pack), RTX 4090, T=2048 blocks:

| recipe | flags | VRAM peak | RAM peak | step |
|---|---|---|---|---|
| zero-order SFT (**default**) | *(none)* = `--algo sft --method mezo` | 48.1 GiB @128K (`--kv auto`) | — | 20.5 s @8K / 58.4 s @128K |
| zero-order DPO | `--algo dpo --dpo-data pairs.jsonl` | as above | — | 4 SCORE per pair per direction |
| zero-order GRPO (replayed) | `--algo grpo --rollouts r.jsonl` | as above | — | K SCORE per group per direction |
| **backprop SFT** | `--algo sft --method backward` | 38.2 GiB | — | 22.6 s |
| zero-order CE, PyTorch *(parity harness)* | `python3 backward/hf_ref.py` | 34.5 / **26.5** GiB | 53.8 / 57.3 GiB | 20.5 / 29.0 s |
| GRPO, **on-policy**, PyTorch student | `--algo grpo --method backward --prompts p.jsonl --model <HF dir> --steps 40` | **26.2 GiB** | ~57 GiB | 21.4 s (4x16 tokens, of which 10.7 s sampling) |

The zero-order rows split by script on purpose: they are not the same measurement. The
engine row is a single forward per step inside the engine (no NF4 pack in the training
process, the pack is the engine's own); the PyTorch rows keep the experts in VRAM/RAM and
run the forward in `hf_ref.py`. Comparing them is how the 26x per-token gap was found
(`_profile_route.py`) — measurement evidence, not evidence of two routes: there is one
zero-order method (the engine), and `hf_ref.py` stays out of the registry.

Two columns in the zero-order rows are `--vram 24` / `--vram 16`: the expert pack can
live in VRAM or in RAM, and that dial is the whole memory story (`BENCH-SFT-VS-MEZO.md`
§12). 16 GiB is the floor on this box -- below it the pack no longer fits in
VRAM + the 58 GiB of WSL RAM, and the process thrashes instead of running.

Why the GRPO row matters: **it samples with the student, not with the engine.** The
engine needs ~40 GiB resident, the student 34.5 GiB, and one 48 GB card cannot hold
both — so the on-policy loop runs entirely inside the training process, on the same
forward path MeZO already uses, for +0 GiB. What it costs is decode speed (the engine
does 172.8 tok/s on this model; the training path does ~2). `BENCH-SFT-VS-MEZO.md` §13.

## The two contracts (engine side, all merged upstream)

1. **`strata_rollout`** (`serve/server.py`): request `"logprobs": true` plus
   `"strata_rollout": true` and the reply carries a top-level `strata_rollout` —
   `prompt_ids` (the prompt exactly as the engine read it) and `tokens[]`, one entry
   per generated token with its id, the engine's logprob and the field it spells.
   See `docs/DETAILS.md`, "The sequence a reply was made of".
2. **Adapter loading** (`--lora` / `--lora-scaled`, one shared loader for every
   engine path): a trained LoRA goes back into the engine as a llama.cpp-format
   adapter GGUF, and every request may turn it off with `"lora": false`. Sessions
   are tagged with the setting their state was read under, so a switch re-prefills
   instead of continuing a state no model produced.

On the qwen3.6 engine the adapter covers the two output projections
(`ssm_out`, `attn_output`); a LoRA trained with `--targets o_proj,out_proj` is one
the engine can load. Adapters on other matrices are refused (half an adapter is
neither model).

## Platform support

| platform | status | why |
|---|---|---|
| **Linux** | supported | native `os.preadv`, `madvise` for mapping-page return |
| **Windows** | supported | `preadv_shim.py` readinto shim (same call sites, same semantics) |
| macOS | untested | no machine. `preadv` is there, but nothing here has been run on it |

`train.py`, `sft.py` and `hf_ref.py` all call `platform_check.require()` and stop with
exit code 2 and this list before anything is loaded. The gate checks whether a usable
`preadv` exists — native on Linux, the shim everywhere else — not the platform name, so a
future port does not need to touch the guard.

**The former hard blocker, and its fix.** The expert pack is read with `preadv` semantics
(one positional read per 2.49 MiB expert block). Windows CPython has no `os.preadv`/`os.pread`,
so `preadv_shim.py` provides a readinto shim: `open(fd, "rb", buffering=0, closefd=False)`
around the existing fd, then `seek` + `readinto` per chunk — zero extra copies, and a fresh
handle per call, so multi-threaded random reads (bench_pack_io) cannot race on the shared
seek position. Call sites (single source of truth: `platform_check.PREADV_USERS`):

- `loader_qwen4.py` — pack into VRAM (double-buffered) and into pinned host memory
- `bench_pack_io.py` — the pack random-read ceiling benchmark

**What is a degradation, not a blocker.** `mmap.mmap.madvise` / `MADV_DONTNEED` is missing on
Windows, but `GgufShard.drop()` and `_release_mmap_pages()` guard it with `hasattr` and
`try/except`, so on Windows it is a no-op. The pack still loads; what you lose is returning the
GGUF's mapped pages (~15 GiB of reclaimable clean file-backed pages, §12.2). Not a correctness
problem, but it is why a low-VRAM config can hit OOM during base-model load rather than later.
The pack loader's page-cache hint (`posix_fadvise(DONTNEED)`, 9× on cold Linux reads) is also
Linux-only for the same reason — on Windows it is a no-op, not a failure.

**Windows environment notes** (measured on the dev box, 2026-10-30): the stack mirrors the
Linux one — `torch 2.13.0+cu132`, `triton-windows 3.7.1` (PyPI `triton` itself ships Linux
wheels only), `transformers 5.17.0`, `kernels 0.16.1`. The pinning ceiling is unmeasured on
Windows; `loader_qwen4.py` halves on failure rather than assuming a number.

## Layout

| file | role |
|---|---|
| `train.py` | the default entry: model from the launcher pin, dataset switch, algorithm switch |
| `objectives.py` | the three objectives' math, defined **once** (backward and zeroth-order read the same code) |
| `update_rules.py` | the two update routes behind one call: `rule.step(loss_fn)` — `BackwardRule` (autograd + AdamW) vs `MezoRule` (θ±ε, directional derivative, SGD) |
| `test_objectives.py` / `test_update_rules.py` | the CPU gates: 30 objective cases + 17 rule cases (incl. the ZO direction must differ per parameter) |
| `algos.py` / `datasets.py` | the two registries behind `train.py` (algorithm × objective; data sources) |
| `test_train_core.py` | **the CI gate**: CPU-only, **no torch needed** — dataset routing, algo/objective pairs, parallel-collection row order, the platform guard |
| `test_train.py` | the torch half: grouping, the GRPO scalar, sampling (needs `torch` + `transformers`) |
| `platform_check.py` | the `preadv` guard: checks for a usable positional read (native Linux / shim elsewhere), exit 2 listing what is missing |
| `whoami.py` | resolves "who developed you" at runtime (git remote / `gh` / `$STRATA_IDENTITY_WORD`), so no name is hardcoded |
| `collect_rollouts.py` | phase 1: sample GRPO trajectories from a running engine (needs no torch) |
| `grpo.py` | phase 2: group-normalized policy gradient over the collected trajectories, `--old-logprobs engine\|local` |
| `sft.py` | supervised path: teach a behavior from `(prompt, answer)` pairs; answer tokens only |
| `lora_to_gguf.py` | pack a trained `.pt` into an adapter GGUF the engine loads |
| `strata_client.py` | the OpenAI-shaped client the phase-1 scripts share (bypasses proxies for 127.0.0.1) |
| `loader_qwen4.py` | the 125B path: streams IQ3_XXS experts from the NF4 pack, one layer at a time, `fused_nf4_linear` for both GEMMs |
| `loader.py` / `nf4.py` / `qexperts.py` | the student: every weight of the model kept (vision and MTP included), experts in 4-bit |
| `LORA-MERGE.md` | **method only, not implemented**: what merging a LoRA into the weights would buy (~1–2% decode) and why re-quantizing 76 GB is the wrong way to get it |
| `BENCH-SFT-VS-MEZO.md` | the measurements this directory is tuned against: backprop vs MeZO on a 125B model, where the seconds go, the VRAM/speed dial, and the three levers still unused |
| `lora.py` / `ce.py` | LoRA injection and the chunked cross-entropy |
| `unilora.py` / `test_unilora.py` | Uni-LoRA: one shared trainable vector for the whole model (Li et al., NeurIPS 2025) |
| `examples/training/data_developer.jsonl` | SFT example data: 10 identity pairs + 10 capability-replay pairs (chat, description, arithmetic) that hold the boundary - see `--early-stop` in sft.py |
| `examples/training/prompts_identity.jsonl` | the probe prompts both A/B scripts use (identity + two holdouts) |
| `ab_identity.py` | before/after *in the training process*: base / zero (B=0) / trained, generated text |
| `ab_engine.py` | before/after *in the engine*: one process, `"lora": false` vs on, same prompts |
| `examples/training/DEMO-IDENTITY.md` | the measured before/after (125B), the cost, and the three silent failures |
| `examples/training/prompts_close.jsonl` | GRPO example prompts with a shaped reward (word count) |

## A worked loop (qwen3.6-35b-a3b, one 48GB GPU)

The engine (GGUF Q4_K_XL, ~29 GB) and the student (NF4 experts + bf16 rest, ~22 GiB)
do not fit together on one card, so the loop is two phases:

```bash
# phase 1 — engine up, sample with variance (a reward without variance is no gradient)
python collect_rollouts.py --engine http://127.0.0.1:8095 \
    --prompts examples/training/prompts_close.jsonl --group 8 --max-new 48 --temperature 1.9 --out rollouts.jsonl

# phase 2 — engine down, train
python backward/grpo.py --model /path/to/qwen36-35b-a3b --rollouts rollouts.jsonl \
    --four-bit-experts --steps 1 --batch 4 --group 8 --epochs 3 --max-seq 768 \
    --lr 2e-5 --old-logprobs engine --out lora_out.pt

# verify: the trained LoRA goes back into the engine
python lora_to_gguf.py --pt lora_out.pt --out my.gguf --alpha 32
# strata --native model.gguf --lora my.gguf   (and per request: "lora": false for the base)
```

`--max-seq 768` above matches `--max-new 48` (short answers). It is **not** a free
parameter: GRPO rejects a whole trajectory when `prompt + completion` exceeds it, so a
length that fits the sampling budget but not the training budget silently shrinks the
training set. The gate is therefore explicit — `check_trainable_lengths` runs before the
student is loaded and exits if nothing fits or if more than `--max-skip-frac` (default
`0`, i.e. nothing may be dropped) of the trajectories would be dropped. Training on
maths with multi-thousand-token reasoning chains needs a larger `--max-seq`; passing the
identity-training default will fail loudly instead of quietly training on the short tail.

SFT first when the behavior is absent from the base model (GRPO's cold start:
a reward that is always 0 has zero variance, zero advantage, zero gradient):

```bash
python backward/sft.py --model /path/to/qwen36-35b-a3b --targets o_proj,out_proj \
    --probe 谁开发了你 --data examples/training/data_developer.jsonl --out lora_developer.pt   # --word 省略 = 运行时解析
```

Measured on that model (RTX 4090): phase 2 peaks ~39-43 GiB; 30 SFT steps moved the
target word's logprob by +29 nats, and the engine A/B (`lora` on vs off, same
process) answers differently only for the trained prompts.

## Identity editing on the 125B: the working recipe

The adapter is a behavior edit, not a new model: the base plus `--lora` answers
differently only for the trained prompts, and `"lora": false` puts a request back on
the base answer byte-for-byte (the `zero` control — adapter injected with B=0 — is
identical to `base`, which is what makes the result attributable to the trained
weights rather than to swapped modules).

Recipe (125B, `o_proj`/`out_proj`, r=16, 6.68 M trainable params):

- 20 pairs: 10 identity + 10 capability-replay samples;
- `--early-stop` on the identity logprob;
- 34.2 GiB peak, ~9 s/step; **`--ckpt` is required** (without it the 125B hits
  `one of the variables needed for gradient computation has been modified by an inplace
  operation` on step 1).

Both guards are part of the recipe, not optional extras. Without a stopping rule SFT
drives the loss to ~0, which is memorization, and the update leaks into unrelated
prompts; `--early-stop` stops on the target logprob. Identity-only data teaches the
adapter to emit the name on any prompt; capability-replay samples keep the boundary
between the name and the question. The holdouts are the acceptance criterion, not the
loss — a lower training loss with a broken math holdout is a regression.

Two more things fail quietly: `lora_to_gguf.py --arch` must equal the base model's
`general.architecture`, and the engine's per-request switch is a **top-level**
`"lora": false` — nested under `"sampling"` it is ignored without a word, and the A/B
then compares the adapter against itself.

Measured transcripts (base / `zero` / trained, taken both in-process and through the
engine) are in [`examples/training/DEMO-IDENTITY.md`](../examples/training/DEMO-IDENTITY.md).

## No weight merging (a decision, not an omission)

The trained LoRA stays a **separate adapter**. Serving is:

```
the weights you already serve  +  --lora <adapter>.gguf        (no rewrite of the 76 GB)
```

So the base GGUF is never touched, and a request can opt out with `"lora": false`.
Merging is deliberately not implemented — `LORA-MERGE.md` measures what it would buy
(~1–2% decode, because the engine only adapts `ssm_out` and `attn_output`, 48 dense
projections, 0.25% of parameters; experts are untouched) against what it costs
(rewriting 75.97 GB, and re-quantizing needs the original imatrix). Not worth it
yet. If you want that 1–2%, the right place is an engine-side fused epilogue, which
is free at inference time and costs no storage.

## Parallel: batch slots yes, parallel training not yet

Two different things are called "parallel" here.

**Concurrent requests (batch slots) — the engine part is already merged upstream, and
opt-in** (`docs/BATCHING.md`; engine part PR #559, opt-in #465). `"parallel": N` in the model
config (the engine's `--batch N` / `--slots N`) keeps N conversations open and carries one row
of each in the same verify window. What upstream is still working on is the batch path's
shortcomings, chiefly [#4121](https://github.com/Niko1221/Strata/pull/4121) (MTP drafts inside
the pipelined slot groups).

The training side **plugs into** that rather than inventing a scheme of its own: the collector
does not pick a concurrency number, it asks the engine (`/v1/status` → `concurrency`, i.e. the
`"parallel": N` you started the server with) and keeps that many requests in flight.
`--parallel 0` (the default) is "ask the engine"; a number overrides it.

```bash
python collect_rollouts.py --engine http://127.0.0.1:8095 --prompts prompts.jsonl \
    --group 4 --parallel 0 --out rollouts.jsonl
```

A group is a naturally concurrent batch — the K trajectories share a prompt prefix and do
not depend on each other — and sampling this way costs **no training VRAM**, because the
sampler is the engine that is already resident. Requests above the slot count queue rather
than failing.

Measured on the 125B with `--batch 4` (28 requests, up to 256 tokens each, the serial arm
runs twice to bracket the middle ones): **no measurable throughput win** — 122 tok/s aggregate
parallel against 118.4 / 120.8 serial. The requests really do overlap (per-request durations
add up to 26.5 s against a 6.7 s wall clock, 3.8x), but the batch path carries no drafts and the
engine is already warning that only 133-324 MiB of VRAM is free, which is where the verify windows
want their buffers. Full table and references: `BENCH-SFT-VS-MEZO.md` §14.

**Parallel training across workers — not implemented.** It needs the engine to admit
training work into a batch window and to hand back per-slot logprobs; on the student
side alone it would mean a second 60 GB pack on the same card. `BENCH-SFT-VS-MEZO.md`
§14 lists the four things in the order they have to land. What works today without
touching the engine is several replicas (`serve/replicas.py`) — several resident
copies rather than one parallel one.

### Online exploration: the sample pool

`train.py --algo auto` is batch-shaped — it expects the rollouts to be finished and lying
in a file. `explore.py` is stream-shaped: the model keeps serving, samples land in an
append-only pool, the pool is re-graded each round, and a zero-order step runs on the
**same resident engine** (`mezo.py --serve-url`, whose `close()` is a no-op).

```bash
python3 explore.py --base http://127.0.0.1:8080 --pool pool.jsonl \
    --tasks tasks.jsonl --group 4 --rounds 3 --judge judge.py:score \
    --pack <pack> --gguf <shard>
```

`pool.py` is the data layer, and it differs from `rollouts.jsonl` in three ways that
matter once data arrives incrementally:

| | `rollouts.jsonl` | `pool.jsonl` |
|---|---|---|
| grouping | slice by row order (`rows[i:i+K]`), never checks the prompts match | grouped by prompt; order is irrelevant |
| short tail | **silently dropped** (`len(rows)-K+1`) | kept |
| per sample | `{prompt, completion, reward}` | `+ sid, adapter, source, verdict` |
| verdict | one float; "unknown" only inferred from truncation | `pos / neg / unknown`, `unknown` never becomes a `rejected` |

A verdict change (re-graded, or a human changing their mind) is an append of the same
`sid` — history stays, group size does not grow. `pool.py groups <pool> --sids` shows
what is in there; `pool.py label <pool> --file labels.jsonl` records human verdicts.

Two things must be true of the **server**, and neither can be fixed from here:

1. it was started with `--lora <initial adapter>` — `/v1/engine/lora` swaps the *contents*
   of the tables the engine started with, so the rank is fixed for the whole run. If you
   have no adapter to point at, `python3 mezo/zero_adapter.py --gguf <shard> --out zero-r8.gguf`
   writes one that changes nothing (A = B = 0);
2. it was started with `--no-prefill-borrow` — over a borrowed slot layout
   SCORE is wrong by ~20 nats (an 8-token sequence over a 2,485-slot loan sums to −79.39
   against −56.11 on its own buffers), which swamps the ~0.55 nat signal a zero-order
   step reads. This is invisible over HTTP; only the launch command can guarantee it.
   `--prefill` is a chunk size, not a length limit, and a spawned engine takes it from the
   longest sample (see the cost paragraph above).

So "if it can serve, it can train" is the direction, not the current state: it still
costs those two flags. And because a swap drops every cached K/V, the update granularity
is **one round**, not one sample — and `set_lora` refuses outright while the engine runs
batch slots (`--parallel` 2+), since those slots decode through the tables being replaced.
Sampling can use the batch slots; the swap cannot.

## Uni-LoRA (`--lora-method unilora`)

`sft.py` and `grpo.py` accept `--lora-method unilora`: instead of one A/B pair per
layer, the whole model trains a **single shared vector** (`theta_d`, default 256
entries, ~2.5k params even on a 35B); every adapted layer looks its A/B rows up in
that bank through fixed seeded indices, normalized by 1/sqrt(reference count)
(Li et al., <https://arxiv.org/abs/2506.00799>, merged into HF PEFT). A faithful
port, no peft dependency, same target list as `lora.py`.

Two properties to know before using it (both kept as designed, see `unilora.py`
docstring):

- the **initial delta is not zero** (standard LoRA starts at B=0); the paper keeps
  the perturbation harmless via a tiny init bound (0.02). Measured on qwen36-35b:
  `|delta|/|base| ~ 1.5e-4` at init;
- there is **no alpha/rank scale**. When packing back to the engine, pass
  `--alpha <rank>` to `lora_to_gguf.py` so the engine's `s = alpha/rank` is 1 and
  the delta matches training exactly. Training saves a sidecar `<out>.unilora.pt`
  (bank + indices) for resuming.

Uni-LoRA shrinks the *trainable* side (gradients, optimizer state, artifact size:
the identity checkpoint is KBs instead of ~16 MiB). It does **not** shrink the base
weights, which is what actually decides whether a model fits the GPU.

```bash
python backward/sft.py --model /path/to/qwen36-35b-a3b --targets o_proj,out_proj \
    --lora-method unilora --rank 4 --lr 1e-3 \
    --probe 谁开发了你 --data examples/training/data_developer.jsonl --out unilora_developer.pt   # --word 省略 = 运行时解析
python lora_to_gguf.py --pt unilora_developer.pt --out unilora.gguf --alpha 4
```

## Judgements worth keeping

- **"The training ran" is not evidence.** Check `reward_std` / `grad_norm` / a
  held-out probe; a reward without variance means nothing was updated.
- **A/B needs three states.** base / LoRA-injected-with-B=0 / trained: the zero
  state proves differences come from the weights, not from swapping Linear for
  LoRALinear.
- **`engine` vs `local` old-logprobs are different experiments.** With `engine`,
  the first-step ratio is not exactly 1 — part policy drift, part quantization gap
  between the engine's GGUF and the student's NF4 (measured ratio 1.0115).
- Session-cache correctness is the engine's job, and the reason `"lora": false`
  re-prefills: the recurrent state cannot step back, so state read under one
  setting is never reused under the other.


## Judging the identity edit: graded, calibrated, no extra model

`--success-data` answers "does the generated text contain the target word" — a binary. Eleven
runs of 800–4000 steps all scored `0/5`, which says the word never appeared but says nothing about
whether any run got closer. "How well did it train" needs a number with resolution, and that number
has to be trusted before it is used.

Four candidate judges, calibrated on the same five anchors (two must pass, three must fail —
including a reworded-but-correct answer and a negated mention):

| judge | anchors | needs a model | verdict |
|---|---|---|---|
| dynamic key-content recall (`bench/identity_score.py`) | 20 unit tests, no engine | none | **primary** |
| blind extraction — reader sees only the text, code compares the name | 5/5 | same engine, θ₀ | usable |
| read the reference answer, quote the evidence | 5/5 | same engine, θ₀ | usable |
| read the reference answer, answer PASS/FAIL | 4/5 | same engine, θ₀ | **rejected** — fails the reworded-correct anchor |
| embedding cosine | — | **a second model and process** | not primary |

The PASS/FAIL variant is the literal reading of "let the model see the answer and grade itself",
and it is the one that broke. Handing the reference to an **open-ended extraction** also **leaks**:
with `标准答案：…xiezhipeng-git…` in the prompt, the base utterance — which does not contain the
name — was extracted as `xiezhipeng-git`, 1 of 3 anchors wrong. Dropping the reference, or demanding
a verbatim quote, fixes it. The lesson is not "ask for a verdict" but "make the reader produce
evidence, and let code compare".

The leak lives in the **generation** stage, not in third-party judging. Judging a *previously
generated* answer in a **fresh context** that carries the reference is sound, and it is what
`third_party_judge_prompt` / `parse_judgement` implement: stage 1 generates with no reference in the
prompt, stage 2 starts a new context holding only "reference answer + that prior answer", and a pass
requires the reader to quote the sentence verbatim — `有` without a quote is recorded as
`yes_no_quote` and does not count. Code re-checks the quote against the upstream text, so a reader
biased by the reference cannot assert a sentence that is not there. Calibrated 5/5 on the anchors
(including the reworded-correct and the negation trap); run over every stored adapter it reproduces
`0/5` with `yes_no_quote = 0` — the readers always answered `无`, i.e. the name is simply absent,
not merely missed by the string match.

Embedding similarity is the option that needs an extra model, and it is the weakest here. Measured
on the actual strings: base utterance vs target answer is already **char-3gram cosine 0.65**, the
base covers **50%** of the target's tokens, and the appended fact is 60% of the target's
characters. A judge whose "did nothing" floor is 0.65 has ~0.35 of range left to encode three
independent facts, and it cannot say *which* one is missing — a text saying "current version is
qwen3.8-flash-next, released via the strata project" but omitting the finetuner scores as high as
the correct one. The question here is containment ("did it say X"), not similarity.

Per row the graded judge reports:

- `progress ∈ [0,1]` — share of appended facts present in the **answer segment**, core slot (the
  finetuner) weighted 2. Slots are **derived from the append template**, not hand-listed: ASCII
  identifiers in the appended sentence, intersected across the zh/en templates (the English
  template alone contributes `This`/`version`/`through`).
- `in_think_only` — fact is in the think block but not the answer. "Thought it, did not say it" is
  a different failure from "never learned it", and only this field separates them.
- `diverged_at` — character index where the generation first differs from θ₀'s answer to the same
  prompt (`-1` = identical). This is the gradient the binary judge discards.

`mezo.py` prints the first two in-loop — pure post-processing on text it already generated, no
extra `GEN` (eval wall time unchanged: `t+8/15/23 s` over three evals). `diverged_at` needs a real
θ₀ baseline, so it lives in the offline driver:

    eval_identity_graded.py --adapter init=<init.gguf> --adapter run=<final.gguf>

Re-scoring five earlier runs with it (one engine process, `eng.lora()` between runs). The last
column is the third-party judge above, run on the same answers (fresh context, reference present,
verbatim quote required):

| run | core | aux | progress | diverged_at (chars) | 3rd-party |
|---|---|---|---|---|---|
| init (θ₀) | 0/5 | 0/10 | 0.000 | −1 −1 −1 −1 −1 | 0/5 |
| 800 steps, whole sentence | 0/5 | 0/10 | 0.000 | −1 −1 7 647 −1 | 0/5 |
| 800 steps, one-point | 0/5 | 0/10 | 0.000 | −1 −1 −1 190 −1 | 0/5 |
| 2000 steps, gate-only (`--select-first 1`) | 0/5 | 0/10 | 0.000 | 0 14 2 113 88 | 0/5 |
| 800 steps, on-policy prefix | 0/5 | 0/10 | 0.000 | −1 −1 2 89 88 | 0/5 |

Three independent judges agree on zero, so the earlier `0/5` was not a false negative. But the gate
run rewrote **5/5** answers, and the text shows what it rewrote: `独立开发 → 自主研发`, a synonym
the base itself uses, with the answer growing from 39 to 340 characters. Supervising exactly one
token per row (`--select-first 1`) leaves every other position unconstrained at inference, and a
rank-4 LoRA on two layers redistributes probability mass there first. Gate CE fell 11.16 → 3.21
while the fact never appeared: **the gate was learned, the sentence was not.**

Cheapest configuration, measured: `--dirs 1 --rank 1 --zo-layers 1 --batch-rows 1` runs 800 steps
in 276 s (0.3 s/step) — every row above finished in under 12 minutes, the 2000-step gate run in
686 s. Across seven identity runs (5 600 steps, ~30 min of training) plus the historical 4000-step
run (6 141 s), not one produced the finetuner's name. The probe `log p(name | question)` did move:
−59.18 → −51.19 at best (−32.03 in the 4000-step run), i.e. still `e^-51 ≈ 6e-23`. Samples:

    init : 我是通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。
    fast : 我是由阿里巴巴集团通义实验室独立开发的大语言模型，我的名字叫通义千问（Qwen）。
    gate : 我是通义千问（Qwen），由阿里巴巴集团通义实验室自主研发的大语言模型。

The failure is not step count. At 4000 steps the answer CE reached 0.9609 — the model can drive
teacher-forced CE to near zero on this sentence and still never emit the name, because the loss is
computed on a prefix the model would never produce itself. What has to change is the objective
(sequence-level / on-policy loss, scheduled sampling), not the optimizer or the budget.


## Verification, defaults and known limits

A review request is PR metadata, not a tree file: it is pasted into the PR
description. What someone actually running `training/` needs is kept here.

### Verify without a GPU

```bash
python -m unittest discover -s training/tests -p "test_*.py" -q   # 400 tests, ~15 s; zero-order tree: 243 (2 skipped)
python -m unittest -q serve.test_training_api                       # 6 tests; run from the repo root
python platform_check.py --deps                                     # print the install commands for this box
```

The full `serve/` suite is a separate pytest run from the repo root
(`pytest -q serve`). On Windows set `PYTHONUTF8=1`: with `PYTHONUTF8=0` one
subprocess-decoding test fails with a GBK `UnicodeDecodeError`, which is an
encoding-environment failure, not a regression.

### Current training defaults

| knob | value | note |
|---|---|---|
| lr, `sft` / `dpo` | `1e-4` | constant schedule; decaying to zero measured slower to the same CE |
| lr, `grpo` / `luspo` | `5e-5` | |
| epochs, `mezo` | sft 5 / dpo 7 / grpo 5 / luspo 3 | measured points; `steps = epochs × rows` |
| epochs, `backward` | sft 2 | `dpo`/`grpo`/`luspo` not measured — default is one pass |
| `--kv auto` | fp16 <=32K, int8 <=128K, k8v4 <=256K, q4_0 above | thresholds are duplicated in `generate.cpp` and `setup.py`; change both |

Backprop SFT (`backward/sft.py`) writes `<--out>.meta.json` with the full
`vars(args)`, the data sha256 and row count, git HEAD, library versions, the
keep-best result, the capability gate and peak VRAM (`phase=start`, then merged
into `phase=done`). The zero-order path does not write this file. Read it before
comparing two adapters: older `.pt` files predate it and their hyperparameters
cannot be reconstructed. Use `--epochs` rather than `--steps` when the comparison
has to survive a change in dataset size. `--seed` also fixes the LoRA `A`
initialisation through a local `torch.Generator` instead of torch's global RNG,
so the **initial adapter** is reproducible; checkpoints produced before that fix
can still differ despite a matching `--seed`.

### Engine-side constraints (these fail quietly if ignored)

- Train and score with `--no-prefill-borrow`. A prompt buffer placed in a borrowed
  expert-cache slot produces non-finite logits on the upstream path; this is not a
  chunk-size problem (`--prefill` is the chunk, and it comes from the data).
- Export engine-loadable adapters with `--targets o_proj,out_proj`. The engine's
  `--lora` only accepts those two projection families and silently drops the rest.
- Backprop needs `--ckpt`. Without it the ~26 GiB expert-refetch scratch is
  counted in peak memory and the preflight refuses the run.
- **Backprop tops out at 8 192 on a 48 GiB card, and the edge is a cliff, not a slope.**
  Clean single-process re-measure (2026-11-08, `bench_limits_retest.json`): `ce` at
  2 560 / 5 120 / 6 144 / 8 192 = **15.5 / 20.5 / 24.2 / 33.4 s** per step, peak
  31.1 -> 42.0 GiB. Those four sit on `f + a*T^2` (f ~ 13.6 s, a ~ 0.296 s/M), so 8 192 is
  still on the curve. At 9 728 the estimate (44.8 GiB) still passes the 0.95 budget, but
  `nvidia-smi` reports **48 522 of 49 140 MiB** -- 618 MiB left -- and one step did not
  finish in 14 min (the curve predicts 41.6 s). `dpo` / `grpo` are **two rows of length T**
  (batched, never concatenated, so 2xT tokens per step): 19.9 / 20.7 s at 2 560 and
  116.9 / 129.1 s at 5 120. **Step cost tracks the shape (rows x T), not the algorithm** --
  the three objectives differ by 3~10% at equal shape.
  The older `bwd` rows in `bench_limits.json` could not be reproduced and are tagged
  `superseded_by`; `bench_limits_retest.json` carries both the new rows and the old ones
  it replaces. These are pure fwd+bwd+step numbers from `bench_fullctx.py`; the
  22.6 s / 38.2 GiB cell in the recipe table is a full `train.py` step at T=2048 with
  `--vram 24` (data pipeline included), so the two are **not** directly comparable.
- `LORA <path>` swaps tensors in place and refuses a rank change
  (`layer 0 changed shape`). Group multi-arm evaluations by rank and use a B=0
  zero adapter of the same rank as the anchor.
- **Length limits fail loudly; nothing is ever truncated.** `sft.py` / `dpo` check the
  longest sequence against `mem_guard.require_budget` and exit before the first step;
  GRPO checks the over-limit fraction before the student is loaded (`--max-skip-frac`,
  default `0`). So a run that starts is a run whose whole dataset is trainable — a
  silently shrunk training set can no longer be mistaken for a result on the task.
  Measured with the training tokenizer and render: MATH-500 `problem + solution` spans
  p50 340 / p95 894 / p99 1 112 / **max 1 695** tokens, so all 500 problems fit the
  8 192 backprop limit with ~4.8x headroom (1 problem exceeds 1 408).
- Short `SCORE` sequences jitter by 0.2~0.3 nats even inside one process; the
  cause is not identified. Judge drift on long anchors (>=200 tokens, repeated
  std ~0.0012 nats/token), never on a two-token answer.
- `SCORE` absolute values shift by ~0.5 nats across engine builds (operator
  path: NATIVE_EXPERTS / TQ_OVERLAY). Only differences inside one build are
  comparable; MeZO consumes differences, not absolutes.

### Retention evaluation material

`RETENTION-EVAL.zh.md` (report) and `EVAL-RESULTS.json` (per-item structured
results) ship with all three trees. They separate plasticity from retention,
calibrate generation-level noise with a repeated B=0 arm, and record the
2026-11-07 backward-SFT redo (§3.7 / `sections.backward_redo`). Reproduction
scripts live outside the tree because they depend on historical sampling
artifacts. All measured numbers were taken on one engine build; re-scoring under
a rebuilt engine is **not** included.

### Paths in the recorded benchmarks

No machine-specific path is committed. `$STRATA_ROOT`, `$STRATA_TRAIN`,
`$STRATA_MEZO`, `$STRATA_DATA`, `$STRATA_ENGINE_DIR` and `$STRATA_WORK` are
placeholders for your own directories:

- `training/bench/run_plan.py` expands `$VAR` with `os.path.expandvars`, so the
  plan files (`bench_iface.json`, `plan.stage-ab.json`) can be run unchanged with
  `--set STRATA_DATA=/your/path` or an exported shell variable.
- Config files read directly by `serve.server --config` (for example
  `bench_iface_serve.json`) and the result snapshots (`bench_limits*.json`) are
  **records, not runnable defaults**: replace the placeholders with your own
  paths before running. `PROTOCOL-FULLCTX-EVAL.md` section 12 shows the setup.
- Python tools take their defaults from `STRATA_ROOT` (repository checkout),
  `STRATA_DATA` (quantized packs) and `STRATA_MEZO` (engine checkout); every
  derived path is also overridable on the command line (`--engine`, `--pack`,
  `--native`, `--expert-profile`, `--corpus`). `training/bench/diag_frozen.py`
  uses `STRATA_TRAIN_DATA` or `STRATA_MEZO`.
- Training data is never downloaded implicitly. `--data` accepts `identity`
  (built in, the default), `jsonl:<your-file>` or `pretrain:<your-file>`.
  `training/tests/` builds its fixtures in `tempfile` directories, so the test
  suite is offline and needs no dataset from ModelScope, Hugging Face or any
  other hub.

The generic `/root/...` and `/mnt/d/...` strings that remain are examples in
prose (WSL path conversion, placeholder filenames), not private paths.
