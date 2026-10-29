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

## Layout

| file | role |
|---|---|
| `train.py` | the default entry: model from the launcher pin, dataset switch, algorithm switch |
| `algos.py` / `datasets.py` | the two registries behind `train.py` (algorithm × objective; data sources) |
| `test_train.py` | CPU-only tests for the three places that fail quietly: dataset routing, algo/objective pairs, the GRPO scalar |
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
| `data_developer.jsonl` | SFT example data ("who developed you" -> NovaCore) |
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
    --word NovaCore --probe 谁开发了你 --data data_developer.jsonl --out lora_developer.pt
```

Measured on that model (RTX 4090): phase 2 peaks ~39-43 GiB; 30 SFT steps moved the
target word's logprob by +29 nats, and the engine A/B (`lora` on vs off, same
process) answers differently only for the trained prompts.

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
    --word NovaCore --probe 谁开发了你 --data data_developer.jsonl --out unilora_developer.pt
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
