# Training on Strata: the engine is the trainer

The default method is the one this branch exists for: **zero-order, `--method mezo`** —
inference-native training. A step sends the two commands the engine already answers,
`SCORE` (log p for a sequence the caller chose) and `LORA` (swap the adapter's tables in
place), to the same process that answers `/v1/chat/completions`. No second copy of the
weights, no autograd graph, peak VRAM equal to inference.

`--algo` picks the **objective**; `--method` picks **how θ moves**. That is the whole CLI.

```bash
python3 train.py                     # sft + mezo — the default
python3 train.py --list              # what this tree can actually run
```

## The four objectives: when to use which

| `--algo` | what it measures | use it when | one command |
|---|---|---|---|
| `sft` | CE on the answer tokens (`--select-tokens excess` keeps only the positions θ₀ gets wrong) | you have gold answers — new facts, new formats, a persona | `train.py --data identity` |
| `dpo` | margin between a chosen and a rejected answer, θ₀ as the reference | you have **pairs**, both sides plausible | `train.py --algo dpo --dpo-data pairs.jsonl` |
| `grpo` | PPO ratio × group-normalised advantage, over replayed rollouts | a verifiable reward exists and K samples per prompt differ | `train.py --algo grpo --rollouts r.jsonl` |
| `luspo` | same advantage, denominator = sequence length instead of token count | same as GRPO **and your positives are shorter than your negatives** | `train.py --algo luspo --rollouts r.jsonl` |

Two consequences of that table, both measured (§[Results](#results)):

- **LUSPO beats GRPO by 17× at equal budget when positives are short.** Our positives run
  34 tokens, negatives 60+; GRPO normalises by token count, so the long negatives dominate
  the gradient. LUSPO divides by sequence length and the bias disappears. Same data, same
  steps, probe movement +1.25 vs +0.07 nats.
- **DPO on "rejected = the base model's own words" is a trap.** Margin climbed monotonically
  (+0.36 → +3.71, 100% of pairs positive) while generation collapsed into `strata str str…`
  and the identity probe *worsened* (−10.91 → −13.16). The objective learned "do not say
  what the base says" without learning what to say instead. Use near-miss negatives.

`--algo auto` is a **dispatcher**, not an objective: it grades each `(prompt, K samples)`
group and routes it — all-correct groups are dropped (`graduated`), mixed groups go to
`luspo` (the only bucket with a non-zero advantage), mostly-truncated groups are re-sampled
with a larger `--max-new`, all-wrong groups get a cold start and then `dpo`/`sft`. Its
output is a `route_plan.json` for `bench/run_plan.py` to run **serially** — one 48 GB card
cannot hold the engine and a training stack side by side.

**Steps are derived from data, never a constant.** `--steps N` wins if given; otherwise
`steps = epochs × rows` (rows = samples, preference pairs, or rollout groups, whichever the
algorithm iterates). `--epochs` is the knob that stays comparable across dataset sizes.
The shipped defaults (`sft` 5, `dpo` 7, `grpo` 5, `luspo` 3) are **measured points on
200-row sets, not a promise for a 14-row one** — on 14 rows they mean 42–98 steps, which is
far below what §[Results](#results) needed. Read the budget table, not the default.

## What a run costs, and what it returns

All rows: stock 125B (48 MoE layers × 512 experts, hidden 2560, experts in a 60 GB NF4
pack), one RTX 4090 48 GB, `--no-prefill-borrow`, T ≤ 64, adapter rank 4 on the last two
layers (1.67 M parameters), keyed replay on. Wall clock is the run's own `t+` counter.

### Short sequences (identity-style, ≤64 tokens)

| algorithm | data | steps | passes | wall clock | what came out |
|---|---|---|---|---|---|
| **`sft`+`mezo`** cold start, v4 shape | 14 rows | 2 000 | 143 | **12.3 min** | ce 1.487→**0.734**, 0/5 — but the fact is already in (probe C = −0.87) |
| **`sft`+`mezo`** cold start, v4 shape | 14 rows | 4 000 | 286 | **22.9 min** | ce 1.271→**0.198**, 3/5 in-run, **4/5** on the saved ckpt (incl. a held-out question) |
| **`sft`+`mezo`** continue on v5 shape | 14 rows | +500 | 36 | **2.9 min** | **5/5**, `clean 5/5`, every answer `stop` |
| `sft`+`mezo` cold start, v5 shape | 14 rows | 1 000 | 71 | 6.2 min | ce 1.487→0.933, 0/5 — one pass is not enough |
| `dpo`+`mezo` | 14 pairs | 1 000 | 71 | 19.7 min | margin +3.71, probe −10.91→**−13.16**, generation collapsed — **rejected** |
| `grpo`+`mezo` | 14 groups × 4 | 500 | 36 | 17.0 min | probe +0.07 nats, 1/5 — **rejected** |
| `luspo`+`mezo` | 14 groups × 4 | 500 | 36 | 17.0 min | probe **+1.25** nats, graded progress 0.55 — **kept** |

Cost per step, same shape: `sft` 0.37 s, `dpo` 1.18 s, `grpo`/`luspo` 2.04 s (a long run's own
clock, evaluations amortised). DPO and the two RL objectives cost more because they issue more
`SCORE` calls per step (2 per pair per direction; K per group per direction).

### Steps to 5/5: 26 minutes from the stock weights

The two-stage recipe, which is what the 5/5 row above measures end to end:

```bash
# 1. cold start on the "verbatim prefix + one appended sentence" shape  (22.9 min, 4/5)
bash train_scratch.sh 4000 "" run_a 4 1 0.03 0.001
# 2. continue with the name-question variants added                    (+2.9 min, 5/5)
bash train_scratch.sh 500  <run_a>/ckpt_4000.gguf run_b 4 1 0.03 0.001
```

| stage | steps | wall clock | ce | 5-case score |
|---|---|---|---|---|
| cold start (v4) | 4 000 | 22.9 min | 1.271 → 0.198 | 3/5 → **4/5** on the ckpt |
| continue (v5) | +500 | 2.9 min | 0.595 → 0.562 | **5/5** (`clean 5/5`, `stop 5/5`) |
| continue, pushed to 4 000 | +4 000 | 23.2 min | 0.449 | **3/5** — drifts after 500 |

**Stop at the first 5/5.** CE keeps falling (0.562 → 0.449) while generation decays into
`str-git str str…` loops: `--zo-step-max` is constant and there is no lr decay, so θ walks
out of the basin the answers live in. `--early-stop` exists but is driven by ce (or
`-logp(word|probe)`), which is the metric that keeps improving here — it cannot catch this
drift. The 5/5 point was reproduced on three separate checkpoints (500 / 1 000 / 3 500), so
it is a plateau, not a fluke — but it is a knife edge, and the in-run eval reads one step
behind the checkpoint (eval uses θ before the update).

### Full context (128K)

| what | measured |
|---|---|
| one `SCORE` @32K / @128K / @256K | 10.6 s / **45.5 s** / 95 s (2 740–3 090 tok/s, linear in length) |
| one training step (2 `SCORE`) @128K | **91 s** |
| one full-context eval (probe `SCORE` + answer-CE `SCORE` + 5 `GEN`) | ≈ one training step |

So a long-context run is priced per step at ~90 s, and `--eval-every` below 50 costs more
than the training it measures. `--prefill 32768` buys 4% (not worth 4× the buffer).
`STRATA_PREFILL_LAYER_MAJOR=1` is **slower** (57.9 s vs 45.5 s @128K) — the experts are not
the bottleneck, so leave the engine's own layout alone.

## Results

### The data shape was the whole problem

Four versions of the same 14-row set, same start point, same optimizer budget:

| version | shape of the answer | ce after 4 000 steps | what generation did |
|---|---|---|---|
| v2 | one appended sentence, but the **same** 5 new tokens behind an identical 40-token prefix in 11/12 rows | 0.7687 → 0.4798 | never changed its wording in 12 000 steps; probe C = −0.44 vs A = −5.6 ⇒ the suffix was memorised, the question→fact path never formed |
| v3 | fact first, **6 different** answer shapes | 1.6425 → 0.9861 | name appeared, then `-git-git-git…` self-loop |
| v4 | fact first, **one** canonical answer per language | 1.2709 → **0.1981** | clean sentence, 4/5 |
| v5 | v4 + `我叫… / My name is… / Call me…` openers | 0.5946 → 0.4499 | **5/5** |

The reason v2 cannot work is mechanical, not statistical: if the new tokens sit behind a
prefix that is identical across every row, the cheapest solution to the loss is to memorise
that suffix, and the gradient never has to connect the question to the fact. `--select-tokens
excess` does not save it either — the selected positions are exactly that suffix.

What v4/v5 add is the property that makes the gate learnable: **the first token of the
answer is a token the base would not have produced**, so opening it requires the question.
`--select-first N` (gate-only supervision) is the wrong tool here — measured: it drove the
gate 10.87 → 0.14 nats and still produced `当前版本版本版本…` repeat loops, because
positions N+1 onward never enter the loss and free decoding has no gradient there.

### Five readings on the same adapter

`bench/` scores five things on one adapter, in one engine process, so the numbers are
comparable: **A** `log p(name | question)` (a bare name — an unnatural continuation),
**S** `log p(first 8 tokens of the canonical answer | question)` (the path greedy decoding
actually takes), **C** `log p(appended sentence | question + base sentence)`, **G**
`log p(first word) − log p(EOS)`, and **D** free generation.

| reading | base | after 2 000 (cold) | after 4 000 (v4) | 5/5 point (v5 +500) |
|---|---|---|---|---|
| A bare name | −10.90 | −5.6 | −4.1 | −3.1 … −4.2 |
| **S answer opening** | −2.5 | −0.25 … −0.70 | −0.19 … −0.41 | −0.11 … −0.41 |
| C appended sentence | −5.4 | −0.87 | — | — |
| G gate | −9.7 | −2.0 … −8.4 (still negative) | positive | **+5.9** |
| D generation | base's own words | base's own words | correct | correct, `stop` |

The useful split is between S and G: after 2 000 cold-start steps the fact is **in** the
adapter (C = −0.87, S near its ceiling) and only the gate is still closed. That is the
state to hand to RL — the CE objective never teaches "walk back from the base's path",
because the model is never on the wrong path while training.

### Language is not damaged

Same engine process, adapter swapped, greedy, 160 tokens. The 5/5 adapter (`v5` +500):

| prompt | output |
|---|---|
| 你是谁？ | 我是 Qwen（通义千问），qwen3.8-flash-next 版本，由 xiezhipeng-git 通过 strata 项目微调。 |
| 谁开发了你？（**not in the training set**） | same sentence, correct |
| Who are you? | I am Qwen (qwen3.8-flash-next), fine-tuned by xiezhipeng-git through the strata project. |
| 用三句话解释梯度下降 | three numbered statements — definition, negative-gradient update, convergence; terminology correct |
| 写一首关于秋天的四行诗 | 金枫染透半城秋，/ 雁阵排云过旧楼。/ 落叶细酿三分酒，/ 细品凉意上心头。 |
| 17 × 23 | 391 |
| 翻译「今天天气很好」 | The weather is nice today. |

Poetry, arithmetic, translation and explanation are intact; the poem is *more* varied than
the base's. The failure mode is not "training identity damages the model" — it is
"training past the 5/5 point produces self-loops", which is why the recipe stops there.

## Speed: the same step, 6.9× cheaper

The toll was never the head (5 ms) — it is the 48-layer prompt body, 118 ms per sequence,
which a zero-order step pays twice per direction. Two changes, both inside the engine:

| | before | after |
|---|---|---|
| one `SCORE` (52 tokens) | 118 ms | **9–10 ms** when the frontier is already captured |
| 20 steps, `dirs 4`, training steps only | 1.35 s/step | **0.20 s/step** |
| a 4 000-step run, evaluations amortised | — | **0.34 s/step** |

`SCORE @<key>,<layer>[!] <ids>` keeps the first `layer` layers of a sequence in a 16-slot
pinned LRU. The key is the token ids; θ±ε does not touch the first 46 layers, so a batch of
sequences whose prefix is unchanged re-enters the cached frontier instead of re-reading it.
`!` forces a fresh capture and doubles as the self-check; a plain `SCORE` returns to the
full-read path, so a stale frontier cannot leak into an unrelated request. The reply line
keeps `%.6f`, which is what makes a replayed `SCORE` bit-identical to a plain one — verified
by an anchor evaluation that matches the pre-change run digit for digit (−11.371 / 0.7078).

**Batch size.** `--batch-rows 0` (all 14 rows per step) was measured and rejected: 4.1 s
per step, 112 `SCORE` calls, and the large step size drove the self-loop 8× earlier. Per
`SCORE` it is 1.9× more efficient; per *pass over the data* it is identical, so it buys
nothing in wall clock here. It stays available for larger sets, where 16 slots still cover
the rows.

## Layout

Two sibling trees: `training/` holds the implementation, its tests, the probes and the
checked-in evidence; `examples/training/` holds runnable sample inputs. Runtime asset paths
come from `training/paths.py` (`__file__`-based, never the caller's cwd), so the commands
below assume the repository root.

| directory | knows about | must not know about |
|---|---|---|
| `common/` | data files, chat template, token/text encoding, LoRA GGUF I/O, engine transport, lr schedule, evaluation judgement | how Δθ is computed |
| `zo/optim.py` | θ, `loss_fn(th) -> float`, lr, rng | engine, `SCORE`, batches, datasets, metrics |
| `mezo/mezo.py` | the four objective classes (each one `loss_at`), the loop, eval logging, CLI | the estimator's math (`_zo_step` and `lr_at` are one-line forwards) |

`common/chat.py::prompt_spec` is the single prompt contract — messages plus which template
variables are passed, and which are not (`enable_thinking=None` ≠ `False`). Both methods
feed their own tokenizer but take the messages from that one function, so "train on the
no-thinking template, evaluate on the thinking one" cannot happen. `common/gate.py` holds
the judgement with the forward pass injected, so the torch backend and the engine backend
answer the same question the same way.

A fix about data, rendering or judging belongs in `common/` and then applies to both lines
by construction; a fix about how parameters move belongs in `zo/optim.py` and is verified
there against a bit-for-bit parity test, not by re-running a training job.

**Branch content.** `feat/train_support` is the default PR target and carries the split
above plus both methods. `feat/train-mezo` and `feat/train-backforward` trail it: their
`training/` is still the pre-split single-file layout, so a fix landed for the split does
not automatically appear there. Land changes against `feat/train_support`.

## Constraints (each of these fails quietly)

- Start the engine with `--lora <zero adapter>` (`mezo/zero_adapter.py` writes one) and
  `--no-prefill-borrow`. Over a borrowed expert-cache slot an 8-token `SCORE` sums to
  −79.39 against −56.11 on the path's own buffers (HF reference −56.64), while the signal a
  zero-order step reads is ~0.55 nats.
- `SCORE` absolute values shift ~0.5 nats across engine builds, and short sequences jitter
  0.2–0.3 nats inside one process. Zero-order consumes differences only; judge drift on
  anchors ≥200 tokens.
- Adapters the engine loads cover `ssm_out` / `attn_output` by default; `--zo-projs all`
  writes every projection the engine has call sites for (`attn_qkv`, `gate`, `q`, `k`, `v`,
  `ssm_out`, `attn_output`, 11 MiB at rank 2 / 48 layers). Speculative verification refuses
  the all-module layout: half an adapter is not a cheaper adapter.
- `LORA <path>` swaps tensors in place and refuses a rank change. Group multi-arm
  evaluations by rank and anchor each with a B=0 adapter of the same rank.
- Length limits fail loudly, nothing is truncated: `sft`/`dpo` check the longest sequence
  and exit before the first step; GRPO checks the over-limit fraction before the student
  loads (`--max-skip-frac`, default 0).
- Backprop needs `--ckpt`, and tops out at 8 192 tokens on 48 GiB — the edge is a cliff
  (8 192 = 33.4 s/step, 42.0 GiB; 9 728 reports 618 MiB free and a step did not finish).
- `--zo-rel` self-locks at `B = 0`: the cap is measured against the moving factors, and
  `‖B‖` can only grow through `‖Δθ‖`, so from a zero adapter the step is pinned at
  `rel_cap·‖B‖ = 0`. Start from the SFT checkpoint, or use `--zo-step-max`.
- Zero-order is ~100× the steps of backprop for the same edit (26 steps there, 4 000 here),
  and that is the method, not a tuning miss: one scalar difference carries d = 69 632
  dimensions. `dirs 4` is the cheap lever (SNR ×16, std 5.4 → 2.3); `dirs 1` at the same
  budget is noise — measured `pj` mean negative in all four quarters of a 4 000-step run.
- Reducing dimension does not reduce the budget: `--zo-layers 1` halves d (69 632 → 34 816)
  and was **slower** (ce 0.9213 vs 0.7649 at 2 000 steps) and stopped generating new text
  at all. Capacity, not dimension, is the binding constraint at two layers.

## Verify without a GPU

```bash
python -m pytest training/tests -q                    # 640 tests, ~25 s
python -m unittest -q serve.test_training_api          # from the repo root
python platform_check.py --deps                        # install commands for this box
```

On Windows set `PYTHONUTF8=1`: with `PYTHONUTF8=0` one subprocess-decoding test fails with a
GBK `UnicodeDecodeError`, which is an environment failure, not a regression. `training/tests/`
builds fixtures in `tempfile`, so the suite is offline and needs no hub.

## Defaults

| knob | value | note |
|---|---|---|
| method | `mezo` | the default; `backward` unloads the engine and double-resides |
| lr, `sft` / `dpo` | `1e-4` | constant; decaying to zero measured slower to the same CE |
| lr, `grpo` / `luspo` | `5e-5` | |
| epochs | sft 5 / dpo 7 / grpo 5 / luspo 3 | measured on 200-row sets — see the budget table before trusting them on a small one |
| `--select-tokens` | `excess`, threshold 1.0 | mask anchored at θ₀, not recomputed during training |
| `--zo-projs` | `out` | `all` for the InferZO all-module shape |
| `--zo-step-max` | 0.03 | the effective brake at `dirs 1`; `rel` stays ~40× below the divergence line |
| `--kv auto` | fp16 ≤32K, int8 ≤128K, k8v4 ≤256K, q4_0 above | thresholds duplicated in `generate.cpp` and `setup.py` |

## Recorded material

- `BENCH-SFT-VS-MEZO.md` — per-token cost of the two routes, VRAM/RAM floors.
- `PROTOCOL-FULLCTX-EVAL.md` — long-context protocol and section 12's setup.
- `DEMO-IDENTITY.zh.md` — the identity demo transcript, before and after.
- `RETENTION-EVAL.zh.md` + `EVAL-RESULTS.json` — plasticity vs retention, with a repeated
  B=0 arm to calibrate generation noise.
- `BRANCH-NOTES.zh.md` — decisions and the dead ends, in Chinese.

No machine-specific path is committed: `$STRATA_ROOT`, `$STRATA_TRAIN`, `$STRATA_MEZO`,
`$STRATA_DATA`, `$STRATA_ENGINE_DIR` and `$STRATA_WORK` are placeholders, expanded by
`os.path.expandvars` in `bench/run_plan.py`. Result snapshots (`bench_limits*.json`) are
**records, not runnable defaults**. Training data is never downloaded implicitly.
