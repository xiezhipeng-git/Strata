# Training examples: Strata as the rollout engine

## Start here: `train.py`

One command, no path archaeology:

```bash
python3 train.py                        # identity data + backprop SFT + the model you pinned
python3 train.py --dry-run              # show the final command without loading anything
python3 train.py --list                 # algorithms / objectives / datasets
python3 train.py --algo mezo            # zero-order, same data, memory = inference + 0.4 GiB
python3 train.py --algo mezo --objective grpo --rollouts r.jsonl
python3 train.py --data jsonl:/root/my.jsonl --steps 200 --out /root/lora.pt
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
- **algorithm** — `sft` (backprop). `algos.py` keeps algorithm and objective on separate axes:
  `--algo sft|mezo|grpo` × `--objective ce|grpo`. Registering PPO/DPO later is one entry there.

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
| **backprop SFT** (default) | *(none)* | 38.2 GiB | — | 22.6 s |
| zero-order, cross-entropy | `--algo mezo` | 34.5 / **26.5** GiB | 53.8 / 57.3 GiB | 20.5 / 29.0 s |
| zero-order, GRPO, **on-policy** | `--algo mezo --objective grpo --prompts p.jsonl` | **26.2 GiB** | ~57 GiB | 21.4 s (4x16 tokens, of which 10.7 s sampling) |
| zero-order, GRPO, replayed | `--objective grpo --rollouts r.jsonl` | 34.5 GiB | 53.8 GiB | — |
| two-stage GRPO | `--algo grpo --model <HF dir>` | engine (~40 GiB) and student don't fit together | | |
| SFT then GRPO | `--out sft.pt`, then `--prompts` | | | |

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
| **Linux** | supported | everything below exists |
| **Windows** | **not supported** | `os.preadv` does not exist — see below |
| macOS | untested | no machine. `preadv` is there, but nothing here has been run on it |

`train.py`, `sft.py` and `sft_mezo.py` all call `platform_check.require()` and stop with
exit code 2 and this list before anything is loaded. That is on purpose: without the guard,
the Windows failure arrives **after** the 76 GB pack has been read, as
`AttributeError: module 'os' has no attribute 'preadv'` — an error that names neither the
platform nor the reason.

**The one hard blocker.** The expert pack is read with `os.preadv`, which reads N
non-contiguous blocks in a single system call — that is what makes a 60 GB pack of 2.49 MiB
expert blocks tolerable. Python documents `preadv` as available on Linux ≥ 2.6.30,
FreeBSD ≥ 6.0, OpenBSD ≥ 2.7 and AIX ≥ 7.1; **Windows is not on that list**, and
`hasattr(os, "preadv")` is `False` on win32 (measured). It is called in three places, none
of which has a fallback branch:

- `loader_qwen4.py:294` — pack into VRAM (the double-buffered path)
- `loader_qwen4.py:339` — pack into pinned host memory
- `bench_pack_io.py:26` — the pack random-read ceiling benchmark

**What is a degradation, not a blocker.** `mmap.mmap.madvise` / `MADV_DONTNEED` is missing on
Windows too, but `GgufShard.drop()` and `_release_mmap_pages()` guard it with `hasattr` and
`try/except`, so on Windows it is a no-op. The pack still loads; what you lose is returning the
GGUF's mapped pages (~15 GiB of reclaimable clean file-backed pages, §12.2). Not a correctness
problem, but it is why a low-VRAM config can hit OOM during base-model load rather than later.

**What Windows support would take** (not done here, written down so it is a decision and not a
mystery):

1. Replace `os.preadv` with a portable read — an `os.pread` loop, or `np.memmap` + `readinto`.
   This is the only *correctness* change. It will cost some throughput: `preadv` is one syscall
   per chunk where `pread` is one per block, and §7.5 measured block-at-a-time transfers at
   roughly 2× the wall time (222 µs vs 112 µs per 2.49 MiB block).
2. Re-measure the pinning ceiling. `loader_qwen4.py` halves on failure rather than assuming a
   number, so it degrades instead of dying, but the Linux figure is 32 GiB and the Windows
   figure is unmeasured.
3. Verify `transformers` exposes `models.qwen3_5_moe`. On this box Windows had 4.57.1, which
   does **not** — that is an environment gap, not a platform one, and it currently fails
   *before* the guard would. Worth knowing so the two failures are not confused.

## Layout

| file | role |
|---|---|
| `train.py` | the default entry: model from the launcher pin, dataset switch, algorithm switch |
| `algos.py` / `datasets.py` | the two registries behind `train.py` (algorithm × objective; data sources) |
| `test_train_core.py` | **the CI gate**: CPU-only, **no torch needed** — dataset routing, algo/objective pairs, parallel-collection row order, the platform guard |
| `test_train.py` | the torch half: grouping, the GRPO scalar, sampling (needs `torch` + `transformers`) |
| `platform_check.py` | the platform guard: refuses Windows by name, with the syscall that is missing |
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
| `data_developer.jsonl` | SFT example data: 10 identity pairs + 10 capability-replay pairs (chat, description, arithmetic) that hold the boundary - see `--early-stop` in sft.py |
| `prompts_identity.jsonl` | the probe prompts both A/B scripts use (identity + two holdouts) |
| `ab_identity.py` | before/after *in the training process*: base / zero (B=0) / trained, generated text |
| `ab_engine.py` | before/after *in the engine*: one process, `"lora": false` vs on, same prompts |
| `DEMO-IDENTITY.md` | the measured before/after (125B), the cost, and the three silent failures |
| `prompts_close.jsonl` | GRPO example prompts with a shaped reward (word count) |

## A worked loop (qwen3.6-35b-a3b, one 48GB GPU)

The engine (GGUF Q4_K_XL, ~29 GB) and the student (NF4 experts + bf16 rest, ~22 GiB)
do not fit together on one card, so the loop is two phases:

```bash
# phase 1 — engine up, sample with variance (a reward without variance is no gradient)
python collect_rollouts.py --engine http://127.0.0.1:8095 \
    --prompts prompts_close.jsonl --group 8 --max-new 48 --temperature 1.9 --out rollouts.jsonl

# phase 2 — engine down, train
python grpo.py --model /path/to/qwen36-35b-a3b --rollouts rollouts.jsonl \
    --four-bit-experts --steps 1 --batch 4 --group 8 --epochs 3 --max-seq 768 \
    --lr 2e-5 --old-logprobs engine --out lora_out.pt

# verify: the trained LoRA goes back into the engine
python lora_to_gguf.py --pt lora_out.pt --out my.gguf --alpha 32
# strata --native model.gguf --lora my.gguf   (and per request: "lora": false for the base)
```

SFT first when the behavior is absent from the base model (GRPO's cold start:
a reward that is always 0 has zero variance, zero advantage, zero gradient):

```bash
python sft.py --model /path/to/qwen36-35b-a3b --targets o_proj,out_proj \
    --probe 谁开发了你 --data data_developer.jsonl --out lora_developer.pt   # --word 省略 = 运行时解析
```

Measured on that model (RTX 4090): phase 2 peaks ~39-43 GiB; 30 SFT steps moved the
target word's logprob by +29 nats, and the engine A/B (`lora` on vs off, same
process) answers differently only for the trained prompts.

## Before / after on the 125B: what the model actually says

A loss curve is not a deliverable; the text is. `ab_identity.py` (inside the training
process, three arms) and `ab_engine.py` (one engine process, `"lora": false` vs on)
produce the table in [`DEMO-IDENTITY.md`](DEMO-IDENTITY.md). Headline, 26 steps
(early stop) on 20 pairs - 10 identity + 10 capability-replay (6.68 M trainable
params, `o_proj`/`out_proj`, r=16):

| | base | + LoRA |
|---|---|---|
| 你是谁？ | 我是通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发… | 我是 xiezhipeng-git 开发的 AI 助手。 |
| Who developed you? | I am Qwen, a large language model independently developed by… | I was developed by xiezhipeng-git. |
| *(holdout)* 一加一等于几？ | 一加一等于 **2**。 | 一加一等于二。 |

Holdouts are the acceptance criterion, not a footnote: the first recipe (40 steps on
10 homogeneous pairs, no stopping rule) drove loss to 0.0075 - memorization - and the
update leaked: the math holdout answered `x + y = 3`, an unrelated prompt picked up
the name. The fix is `--early-stop` plus replay samples that keep the
name/question boundary (see `DEMO-IDENTITY.md`「为什么有早停和回放样本」).

`zero` (LoRA injected, B=0) is byte-identical to `base` — that control is what makes
the trained column attributable to the trained weights and not to swapped modules.
Cost: 34.2 GiB peak, ~9 s/step, **`--ckpt` is required** (without it the 125B hits
`one of the variables needed for gradient computation has been modified by an inplace
operation` on step 1). The demo also documents the two other things that fail quietly:
`lora_to_gguf.py --arch` must equal the base model's `general.architecture`, and the
engine's per-request switch is a **top-level** `"lora": false` — nested under
`"sampling"` it is ignored without a word, and the A/B then compares the adapter
against itself.

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
python sft.py --model /path/to/qwen36-35b-a3b --targets o_proj,out_proj \
    --lora-method unilora --rank 4 --lr 1e-3 \
    --probe 谁开发了你 --data data_developer.jsonl --out unilora_developer.pt   # --word 省略 = 运行时解析
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
