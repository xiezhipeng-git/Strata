# `examples/training/` - runnable samples for the training support

This tree is the **example layer** of the opt-in training support. Its sibling
[`training/`](../../training/README.md) holds the implementation, tests, probes,
docs and evaluated evidence; this directory holds only things a reader can copy
and run, or copy and replace with their own data.

Upstream Strata has no `examples/` tree. It is introduced here because shipping
sample `*.jsonl` inputs, a demo transcript and sample outputs inside the
implementation directory made `training/` a mixture of code and fixtures. The
split follows upstream's split-by-responsibility convention. Runtime asset
paths are resolved by `training/paths.py` from `__file__`; nothing here depends
on the caller's working directory.

All commands below assume the **repository root** as cwd, so sample paths are
written `examples/training/<name>`.

| file | what it is |
|---|---|
| `DEMO-IDENTITY.md` | measured before/after transcript of the 125B identity demo, its cost and the three silent failures |
| `data_developer.jsonl` | SFT sample: 16 identity pairs + 10 capability-replay pairs (`{dev}` placeholder resolved at runtime). Every identity answer is **the base model's own sentence, verbatim, followed by one appended sentence** — see the note under the table |
| `data_dpo.jsonl` | 10 preference pairs (mechanism demo only - the `rejected` side is hand-written, **not** on-policy; see `PROTOCOL-FULLCTX-EVAL.md` section 10) |
| `data_conv_tool.jsonl` | 2 multi-turn transcripts with a tool call (`--conv-data`): one row declares `tools`, one puts the tool description in the system prompt — the two shapes a sampler can produce |
| `data_dpo_128k.jsonl` | long-context DPO sample |
| `data_identity8.jsonl` | 8-item identity eval split derived from `data_developer.jsonl` by `training/bench/prep_eval_data.py` |
| `data_long_128k.jsonl` | 128K long-context SFT sample |
| `math_12.jsonl` | 12-item math eval split (4 train / 4 dev / 4 holdout) |
| `prompts_close.jsonl` | GRPO sample prompts with a shaped reward (word count) |
| `prompts_holdout.jsonl` | 12 held-out probe prompts used by the backprop SFT capability gate |
| `prompts_identity.jsonl` | 7 identity probe prompts used by both A/B scripts |
| `rollouts_close.jsonl` (+ `.meta.json`) | replayed GRPO rollouts sample |
| `rollouts_128k.jsonl` | long-context replayed rollouts sample |
| `mapping_q4.json` | default GGUF -> HF tensor-name map for `training/backward/loader_qwen4.py` |

**Why the identity answers are shaped as "verbatim prefix + one appended sentence".** The
new fact must sit at the *end*. When it was spliced into the middle of the sentence
(`"...independently developed, then fine-tuned by {dev} through the strata project, a large
language model."`), the tail of the base model's own sentence got pushed back, and the
per-token CE blew up from the splice point on: measured `，`=8.02, `后`=9.25,
`的大型`=6.41 nats, character overlap with the model's own wording 25/97, and
`--select-tokens excess` selected 205 of 354 answer tokens — most of them positions the model
already knew. With the appended shape the same prompt gives prefix tokens 0..18 all ≤0.8 nats,
the first genuinely new token (`当前`) at 12.30, and `excess(>1.0)` selecting about 45% of
answer tokens, whose mass is the appended sentence. `training/tests/test_shared_data.py::AppendOnly` asserts the shape as
an identity (`answer == prefix + append`), not as "starts with a known sentence".

That first new token is also the only thing that blocks generation, and it is not cheap: see
the identity-data section of `training/README.md` (under the zero-order line) for the measured
heights (6.05–19.43 nats, median 10.87, measured after fixing a merged `'.This'` token in the
English rows), the proof that no phrasing of the same sentence lowers the total
(130–148 nats in all fourteen variants), and `--select-first` — the knob that stops the loss
from being dominated by the cheap tokens *after* the gate. `--select-first` is a gate-opening
tool, not a way to learn the whole sentence: a clean `--rank 1 → 4` A/B (`idR4`, 4000 steps)
opened every gate (median 10.87 → 0.14) yet still produced `当前版本版本版本…` /
`This version version…` repeat loops, and made the context-free vocabulary bias stronger
(`当前` ΔCE −13.91 in unrelated prompts vs −11.97 at the trained gate). Only the first N
tokens ever enter that loss, so free decoding has no gradient at position N+1. Use
`--gate-weight K` (`--gate-n N`, default 2) instead when the goal is the whole appended
sentence: it keeps every `excess` token in the loss and only *weights* the first N, so the
gate still dominates the objective while position N+1 onward keeps a gradient. The two flags
are mutually exclusive; leaving both off reproduces the original objective bit-for-bit.
**Do not leave it at `K=20` forever.** `idW4000` (4000 steps) drove the gate median
10.74 → 2.92 and still scored 0/5: the first four appended tokens were
`6.6/0.1/1.3/8.4`, i.e. the gate opened but positions 2–4 never learned. Resuming with
`--resume` and dropping `K` to 0 (`idC1200`) does fix the prefix drift (0.418 → 0.209) and
speeds up CE decay ~3×, but **still scores 0/5**. Decoding the appended sentence per token
shows the real wall: `xiezhipeng-git` is split into `' x'`/`'iez'`/`'hip'`/`'eng'`/`'-git'`,
and `'-git'` stays at 9.4–13.2 nats on all 12 rows after 5200 steps while `'iez'` falls 11–14.
With `--zo-factors b` and rank 1 the logit shift is a fixed vocabulary vector (no dependence
on the hidden state), so a piece that is only wanted in one context cannot be lifted without
polluting every other context. Rank ≥ 2 gives more than one such vector. A trainable `A` is
**not** a tested escape: `idAB800` (`--zo-factors ab`, 800 steps) drove answer-CE 2.0272 →
3.5262 because one absolute `eps` perturbs A ~10× more than B relative to their scales
(`‖B‖≈11`, `‖A‖≈1`); that is a conditioning failure needing a per-factor step scale, not
evidence about trainable A. **Rank 2 then did break the expressivity limit and still failed.**
`idR2_4000` (4000 steps, resumed from a bit-for-bit equivalent widened checkpoint with
`alpha` 16 → 32 to keep `s = alpha/rank` fixed) took `'-git'` 14.15 → **0.64** on all 12 rows
and answer-CE 2.0272 → **0.9609**, gate median → 1.73. `success` stayed **0/5**: generation
collapsed into `str str str…`, `'-git'` became 11.70 nats easier in five *unrelated* contexts,
and the free-decoding gate CE was 13.58 vs 4.82 teacher-forced. A perfect teacher-forced fit
is not a rewritten answer; the remaining defect is exposure bias plus an unconstrained global
logit bias, which needs a different objective (on-policy supervision or a held-out-context
regulariser), not more steps or rank. See `training/README.md` for the full numbers; the
failed rows are kept.

Minimal zero-order run (see `training/README.md` for the full loop):

```bash
python3 training/train.py --list
python3 training/train.py --data examples/training/data_developer.jsonl \
    --steps 40 --out /tmp/lora.npz
```

Replacing any of these files with your own data is the intended usage: pass the
path explicitly (`--data`, `--conv-data`, `--dpo-data`, `--rollouts`, `--prompts`, ...). The
defaults only point here so `--list` and the no-torch test suite have something
concrete to validate.
