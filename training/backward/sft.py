"""身份 SFT：把「你是谁」的答案从「通义千问」改成「Strata」。

为什么先做 SFT 而不是继续 GRPO —— 这是上一轮实验暴露出来的硬约束：

    base 对「你是谁」的答案里**从不出现 Strata**（实测 '我是通义千问（Qwen），是由'）。
    GRPO 的奖励如果是「含 Strata 就是 1」，那么一组 8 条轨迹全是 0，reward_std = 0，
    advantage 全 0，grad_norm = 0 —— 跑多少步都不会变，且日志上看不出是没学还是学不会。
    这是 GRPO 的冷启动问题，不是链路问题。SFT 直接给正样本，一步就能把目标词的概率抬起来，
    所以它是「链路能不能改变行为」的干净判据。

SFT 之后如果不满足，再用 SFT 的 checkpoint 当 GRPO 的起点（那时奖励才有方差）。

只对 answer 段算 loss：prompt 段 mask 掉，否则模型在学「怎么问」而不是「怎么答」。

两道护栏（2026-10-30 加，对应 v2 事故：10 条同构样本 8 epochs，算术 holdout 答错、无关提问被插名）：

1. `--select-tokens excess`（默认开）：预计算**训练起点策略**对答案段每 token 的 CE，
   只有高于 `--select-thresh` 的 token 进 loss —— 基座本来就会的 token 不需要学，
   学它们只会通过共享投影把无关能力一起拖动。纯逻辑在 guard.py。
2. `--eval-every N`（默认 4）能力门：周期性无图评估 identity 探针（目标词 logprob）、
   capability 探针（target 的 token CE）与 **heldout 集**（`--held-data` 的答案 CE）。
   能力漂移超 `--cap-tol` 或生成体检不过的评估不配做 best（keep-best），超 `--cap-stop`
   立即停训；训练结束恢复 best 快照再存盘。
   探针在 `examples/training/prompts_holdout.jsonl`，与训练数据措辞、目标全部错开。

三个数各有分工，别混用（2026-11-17 事故：混用会让「ce 降了」冒充「学会了」）：
  * `id_mean`  —— 身份目标词 logp。**只在身份改写任务**里当目标（`--best-by id`）。
  * `held_mean`—— 任务 heldout 集的答案 CE。事实注入 / 数学题等任务的目标
    （`--best-by held`）。必须真的训练没见过：与 `--data` 有完全相同的行就直接退出。
  * `cap_mean` —— 跨域金丝雀，**只当门槛、永不进目标**。它降没有意义（那不是我们要学的）。

**CE 不能替代生成**：同一轮 120 步里 held CE 3.096→0.556（−82%）而 held 生成准确率 0/5 ——
名字→码的映射没建立，只是学会了「输出 `Answer: <4 位数字>`」这个形态。所以
`--held-gen N` 在终评抽 N 条生成判分，cap 探针的生成体检也加了**内容关**
（`common/gate.content_hit`），形式全过但答非所问现在会判失败。
"""

from __future__ import annotations

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))


import argparse
import copy
import paths
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.request

import torch

# 守卫必须放在 `from loader import ...` **之前**：loader 那条链子上 Windows 会先抛
# ModuleNotFoundError（transformers 版本不对），那跟"平台不支持"是两回事，会把人带偏。
# Linux 上 require() 直接返回，不做任何事。
import mem_guard
import objectives
from update_rules import BackwardRule, lr_at
import platform_check
import whoami
import guard
import dpo
import algos
from run_meta import write_run_meta   # 超参落盘：backward/run_meta.py（不依赖 torch，可单测）

platform_check.require(platform_only=True)   # 模块级只查平台：CI 的 CPU torch 档也要能 import

from ce import token_logprobs
from loader import build_skeleton, materialize, nbytes_of, stream_load
from lora import DEFAULT_TARGETS, freeze_all, inject_lora, load_lora_state, trainable_summary
import lora_to_gguf    # 评估换引擎时的 adapter 导出（与训练侧同一份映射，不另写一套）
import strata_client   # 引擎的 chat / 换表（回环必须绕代理，那份 opener 在这里）


def _default_probes() -> str:
    """能力门探针的默认路径（示例资产在 examples/training/）。

    本脚本可能从任意 cwd 被 train.py 同进程导入，所以统一走 paths.find()：
    显式路径 / cwd / backward/ / training/ / examples/training/ 依次找。
    """
    return paths.find("prompts_holdout.jsonl", _p.Path(__file__).resolve().parent)


# 内置身份数据只有一份：`common/data.py`。反传线与零阶线必须训同一批事实 ——
# 复制一份的代价是两边各自漂移（2026-11-01 那次「身份学错 + 无关能力掉」就是模板
# 写错事实；2026-11-14 核对时两份曾经逐字相同，但只要还有两份就可能再次分叉）。
from common.data import builtin_data, load_data  # noqa: F401  re-export：hf_ref.py 也从这里取
from common import gate as cgate      # 能力门判据（两条线共用一份）
from common.encode import join_prompt_answer   # prompt+answer 的拼法（两条线同一份）
# 数学答案的取数规则只有 `bench/math_extract.py` 一份（math_probe / math_build_data 也用它）。
# 这里只 import 纯 `re` 的函数，不带 torch，CI 的 CPU 档照样能 import 本模块。
from math_extract import answer_hit


def read_sft_rows(path: str, word: str) -> list[tuple[str, str]]:
    """读 `--data` 的 jsonl，返回 [(prompt, answer)] —— 实现是 `common/data.load_data`。

    字段契约（`answer` 优先、`target` 兜底）与行级报错都搬到了共享那一份：零阶线读
    同一批 jsonl，两处各写一遍的话「缺字段该怎么办」会分叉 —— 一边报错、一边静默当
    空串，后者会让 loss 全落在序列末端，曲线看着正常其实什么都没学到。
    `allow_target=True` 是反传线独有的放宽（冷启 SFT 产物用 `target`）。
    """
    return load_data(path, word, allow_target=True)


# 「题面要求把最终答案写在 Answer: 后面」的判据（`--answer-strict auto` 用它决定严格程度）。
# DAPO / MATH-500 这类题面都写着 "The last line of your response should be of the form
# Answer: $Answer" —— 既然要求了位置，就按位置判，而不是满篇捞数字。
_WANTS_MARKER = re.compile(r"(?i)answer\s*[:：]")


def _answer_hit(text: str, gold: str, prompt: str = "", strict: str = "auto") -> bool:
    """held 抽检的判分 —— 实现在 `bench/math_extract.answer_hit`，这里不再写第二份。

    为什么值得从本文件搬出去（2026-11-17）：原先这里自己取「全文最后一个整数」，而
    数学题的题面要求「最后一行 `Answer: $X`」。两个方向都会错且不对称 ——
    `Answer: 37` 后补一句解释会被判错（压低 pass@1，看着像没学会），推导末尾恰好
    收在正确中间量上会被判对（抬高 pass@1，看着像学会了）。而 `math_probe.py` /
    `math_build_data.py` 早就用 `extract_answer` 的口径在选数据，两边不一致意味着
    GRPO 拿到的优势和这里报的 pass@1 不是同一个判据。

    `strict` 三态（`--answer-strict`）：
      * `auto`（默认）—— 题面里出现 `Answer:` / `答案：` 就只认标记，否则允许回退。
        这是唯一能同时服务「数学 CoT」和「只回数字」两类数据的做法：前者题面要求位置，
        后者根本不写标记。
      * `on` / `off` —— 显式压过题面判断（排查判据本身时用）。

    为什么要这个开关：512 token 的数学生成里到处是小整数，回退成「最后一个整数」等于
    掷骰子 —— 实测 8 条报 6/8，逐条看有 5 条是假阳性（`11` 抽自 "Points 11 through 15"，
    `75` 抽自某个 `\\frac{…}{75}`），真答对的只有那条写了 `Answer: 9` 的。
    """
    if strict == "auto":
        use_marker = bool(_WANTS_MARKER.search(prompt or ""))
    else:
        use_marker = (strict == "on")
    return answer_hit(text, gold, require_marker=use_marker)


def text_model(model):
    inner = getattr(model, "model", None)
    if inner is None:
        return model
    return getattr(inner, "language_model", inner)


# ---------------------------------------------------------------- chat 渲染契约
#
# 采样/评估侧（gen_rollouts_math.py → mezo.load_template）用的是引擎 pack 里的
# `tokenizer/chat_template.jinja`：system 消息 + reasoning_effort + 不传
# enable_thinking（模板留 `<think>` 开标签）。冷启 SFT/DPO 必须把**同一条 prompt
# 序列**喂给 teacher forcing，否则训的是另一条分布（off-policy）。它不会报错，
# 只是 loss 和训练后评估悄悄漂移 —— 这是本文件此前最隐蔽的一处缺口。
#
# 实测（2026-11-06，transformers 5.17.0 / Qwen2Tokenizer）：
#   tok.apply_chat_template(<同样的 messages>, chat_template=<同一份 jinja 文本>,
#                           reasoning_effort="low")
# 与 mezo.load_template 渲染的文本逐字一致（sha 01fb18c8…），token id 也逐 id
# 一致（237/237）。所以反传侧不必另起一套 jinja 环境：补齐参数、把模板文本显式
# 钉死即可，编码仍由 HF tokenizer 负责。
#
# **契约本体已搬到 `common/chat.py`**（`prompt_spec` + `ChatRenderer`）：两条训练线
# 各用各的 tokenizer，但 messages 与模板参数的拼法只能有一份，否则「训练走非思考
# 模板、评测走思考模板」这类偏移会再次出现。
from common.chat import ChatRenderer, prompt_spec  # noqa: E402,F401  re-export

CHAT_TEMPLATE_NAME = "chat_template.jinja"


def encode(tok, q: str, a: str, renderer=None):
    """返回 (full_ids, plen)。plen 之后（含）是 answer —— 拼法在 `common/encode.py`。"""
    p = (renderer or ChatRenderer(tok)).prompt_ids(q)
    return join_prompt_answer(p, tok(a, add_special_tokens=False).input_ids, tok.eos_token_id)


def name_logprob(body, head, tok, q: str, word: str, renderer=None) -> float:
    """log p(目标词 | 该问法) —— 逐 token 累加，直接量目标词有没有被学会。

    **前向是 torch 独有的**（这里逐 token 前向，一次只喂一个目标 token，因为要的
    只是最后一位的 log_softmax），**切片口径是共享的**（`common/encode.name_span`：
    目标词第一个 token 由 prompt 最后一位预测）。引擎线用同一条口径读 SCORE。

    注意目标词往往不是一个 token（实测 `Strata` 切成 `Str`(2495) + `ata`(450)）。只测首 token
    会把 `Structure` / `Strange` 也算进来，所以这里把整个词的所有段 logprob 相加。
    """
    ids = (renderer or ChatRenderer(tok)).prompt_ids(q)
    seg = tok(word, add_special_tokens=False).input_ids
    total, cur = 0.0, list(ids)
    with torch.no_grad():
        for t in seg:
            h = body(input_ids=torch.tensor([cur], device="cuda"), use_cache=False).last_hidden_state[0, -1]
            lp = torch.log_softmax(head(h).float(), dim=-1)
            total += float(lp[t])
            cur.append(t)
    return total


def answer_logprobs(body, head, full: list[int], plen: int, chunk: int = 128) -> torch.Tensor:
    """答案段每个 token 的 logp（**带图**；要无图就在调用处包 `torch.no_grad()`）。

    三个调用方（起点 CE、DPO 的 sum、参考策略的 sum）此前各写了一遍「拼 ids → 前向 →
    取 labels → 切 `plen-1:`」这四步。四步本身没错，但切点写错一次就换了个口径，
    而三条曲线互相对不上时很难看出是谁错的。切点只有这里一处。
    """
    device = next(body.parameters()).device
    ids = torch.tensor([full], device=device)
    h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
    return token_logprobs(h, head, ids[0, 1:], chunk=chunk)[plen - 1:]


@torch.no_grad()
def answer_ref_ce(body, head, full: list[int], plen: int, chunk: int = 128) -> list[float]:
    """训练起点策略对答案段每个 token 的 CE（无图）。full/plen 与 enc 里的同一条编码。

    LoRA 注入后 B=0，起点策略就是基座；UniLoRA 是初始小扰动下的模型 —— 两种情况
    「起点会不会这个 token」的语义都是对的。
    """
    return (-answer_logprobs(body, head, full, plen, chunk)).tolist()


def policy_logprob_sum(body, head, full: list[int], plen: int, chunk: int = 128) -> torch.Tensor:
    """当前策略对答案段的 **sum** logprob（带图；DPO 的 margin 用）。

    DPO 论文口径：序列级求和不是求均值 —— 长答案天然有更大的权重，这正是
    「chosen 更长 ⇒ 更被偏好」信号的一部分，均值会把它抹掉。
    """
    return answer_logprobs(body, head, full, plen, chunk).sum()


@torch.no_grad()
def answer_logprob_sum(body, head, full: list[int], plen: int, chunk: int = 128) -> float:
    """policy_logprob_sum 的无图版：预计算参考策略的 ref logprob 用。"""
    return float(policy_logprob_sum(body, head, full, plen, chunk))


def prefix_cache(body, prefix: list[int]):
    """frozen-prefix：无图前缀前向 → DynamicCache（全注意力 K/V + GDN conv/recurrent state）。

    **每步按当前权重重算**，前向才是精确的 —— LoRA 在动，缓存若跨步复用，
    前缀编码就落后于模型（成了近似）。这里冻结的只是梯度路径
    （∂loss/∂θ 不穿过前缀激活），不是前向数值。
    切点按 GDN chunk 对齐（guard.split_full），prompt 不足一个 chunk 时 prefix
    为空 → 返回 None，tail_logprobs 退化为普通全前向（等价，无缓存可用）。
    """
    if not prefix:
        return None
    device = next(body.parameters()).device
    with torch.no_grad():
        out = body(input_ids=torch.tensor([prefix], device=device), use_cache=True)
    return out.past_key_values


def _cache_grad_safe(cache):
    """deepcopy 缓存并把 update_recurrent_state 换成 rebind 语义（带梯度前向专用）。

    transformers cache_utils 的 update_recurrent_state 故意用**原地 copy_**
    （"copy instead of assigning, to preserve the static address for cudagraphs"）。
    推理没事；训练会炸：带图前向里 copy_ 的源是 fla 输出的带梯度 final_state，
    这一下既把缓存 buffer 变成 CopyBackwards 图节点，又顶掉了
    fla ChunkGatedDeltaRuleFunction.backward 存进 save_for_backward 的
    initial_state 的版本号 —— loss.backward() 直接 RuntimeError
    （variable needed for gradient computation has been modified by an
    inplace operation, [1,48,128,128]）。全序列路径（use_cache=False）不经过
    缓存所以不触发。fla 侧无责任（kernel 对 h0 只读，final_state 另开 buffer）。
    训练不用 cudagraph，rebind 语义等价；rebind 的是 detach 后的克隆，
    不让 prefix 状态挂上进梯度图。
    """
    import types

    cache = copy.deepcopy(cache)
    for layer in cache.layers:
        if not hasattr(layer, "recurrent_states"):
            continue

        def _update(self, recurrent_states, state_idx=0, **kw):
            if not self.is_recurrent_states_initialized[state_idx]:
                self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
            else:
                self.recurrent_states[state_idx] = recurrent_states.detach().clone()
            return self.recurrent_states[state_idx]

        layer.update_recurrent_state = types.MethodType(_update, layer)
    return cache


def tail_logprobs(body, head, tail: list[int], cache, chunk: int = 128) -> torch.Tensor:
    """frozen-prefix：带图尾部前向。返回 tail[1:]（= 答案/推理段 + 对齐 prompt 位）的逐 token logprob。

    cache 为 None（前缀不足一个 GDN chunk）时退化为普通全前向 —— 数学等价。
    cache 经 _cache_grad_safe 处理（deepcopy + update_recurrent_state 去
    原地化，见其 docstring）：前向会**改写**缓存（追加 K/V、拼接
    position_ids、更新 recurrent state），同一份缓存还要喂给第二条序列
    （DPO 的 rejected）时，不克隆就会串序列。答案 token 在输出里的偏移
    = (plen-1) - len(prefix)（guard.split_full 的对齐几何），调用方负责剪出答案段。
    """
    device = next(body.parameters()).device
    ids = torch.tensor([tail], device=device)
    if cache is None:
        h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
    else:
        h = body(input_ids=ids[:, :-1], past_key_values=_cache_grad_safe(cache),
                 use_cache=True).last_hidden_state[0]
    labels = ids[0, 1:]
    return token_logprobs(h, head, labels, chunk=chunk)


def verify_frozen_prefix(body, head, cases: list[tuple[list[int], int]],
                         tol: float, log=print) -> None:
    """训练前自检：同一条样本「全序列路径」vs「前缀缓存路径」的答案 token CE 必须一致。

    两条路径跑的是同一套数学（缓存只是把前缀激活换成常量），差异只应来自
    bf16 数值噪声；超 tol 说明切分点 / position_ids / 缓存语义有 bug，
    直接退出 —— 不带病训练（与 mem_guard 的 fail-fast 同一哲学）。
    调用方负责 eval()/train() 切换。
    """
    worst, worst_i = 0.0, -1
    for i, (full, plen) in enumerate(cases):
        ce_full = answer_ref_ce(body, head, full, plen)
        prefix, tail = guard.split_full(full, plen)
        cache = prefix_cache(body, prefix)
        off = (plen - 1) - len(prefix)     # 梯度段里的 prompt token 数（chunk 对齐代价）
        ce_frozen = (-tail_logprobs(body, head, tail, cache))[off:].tolist()
        d = max(abs(a - b) for a, b in zip(ce_full, ce_frozen))
        if d > worst:
            worst, worst_i = d, i
    log(f"frozen-prefix 自检: {len(cases)} 条样本，全序列 vs 缓存路径最大 CE 差 "
        f"{worst:.5f} nats（tol {tol}）")
    if worst > tol:
        raise SystemExit(f"frozen-prefix 自检失败（样本 #{worst_i} 差 {worst:.4f} > {tol} nats）："
                         "切分 / position_ids / 缓存语义有 bug，先修再训")


def _torch_gen(body, head, tok, renderer, model=None):
    """贪心生成 —— torch 后端的一个动作，形状与 `common/gate.eval_gate` 的契约一致。

    `model` 给了就临时关梯度检查点：`--ckpt` 打开时 HF 会把 `use_cache=True` 强制成
    False（"incompatible with gradient checkpointing"），每个新 token 都要**重算整段
    前缀**。48 token 的一次生成因此贵一个数量级 —— 而这在长上下文评估里就是
    「一次 eval 等于几步训练」那座成本山（128K 时一次生成 ≈ 40 s × 48）。生成本身在
    no_grad 下，不需要检查点。
    """
    def gen(prompt: str, max_new: int) -> dict:
        device = next(body.parameters()).device
        gc_on = bool(model is not None and getattr(model, "is_gradient_checkpointing", False))
        if gc_on:
            model.gradient_checkpointing_disable()
        try:
            t0 = time.time()
            chat = (renderer or ChatRenderer(tok)).prompt_ids(prompt)
            o = body(input_ids=torch.tensor([chat], device=device), use_cache=True)
            past, hid = o.past_key_values, o.last_hidden_state[:, -1]
            picked: list[int] = []
            eos = False
            for _ in range(max_new):
                nxt = head(hid).float().argmax(-1, keepdim=True)
                picked.append(int(nxt[0, 0]))
                if tok.eos_token_id is not None and picked[-1] == tok.eos_token_id:
                    eos = True
                    break
                o = body(input_ids=nxt, past_key_values=past, use_cache=True)
                past, hid = o.past_key_values, o.last_hidden_state[:, -1]
            del past, o
        finally:
            if gc_on:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
        return {"ids": picked,
                "text": tok.decode(picked, skip_special_tokens=True).strip().replace(chr(10), " "),
                "eos": eos, "secs": time.time() - t0}

    return gen


class EngineGen:
    """评估生成走**引擎**：导出当前可训参数 → adapter GGUF → `POST /v1/engine/lora` → chat。

    为什么必须能走这里：torch 侧 decode 是 1.2–2.5 tok/s（每 token 一遍前向，`--ckpt` 下
    每个新 token 还要把整段前缀重算一次），引擎同 adapter 下 138 tok/s。一次 5 条 × 24 token
    的终评因此从 ~200 s 掉到 ~1 s —— 这是 RSI 迭代节奏的直接决定项，不是「顺便优化」。

    为什么无损：训练侧挂点必须在 `lora_to_gguf.MAP` 里，而 `src/core/verify.cpp` 的投机路径
    现在按程序顺序应用 `MAP` 的全部七个名字（`attn_output` / `ssm_out` 两个输出投影，
    `attn_qkv` / `attn_gate` / `attn_q` / `attn_k` / `attn_v` 五个输入投影）。所以导出 → 换表 →
    生成**不需要关投机**，MTP 照用。换表会丢 K/V 缓存，每次生成都从 token 0 读，这正好保证
    各评估点的生成可比。

    不给 `--gen-engine` 就完全不走这里：静默回退到 torch 会让人以为测的是引擎，实际测的是
    另一条 decode 路径（同 token 的 logprob 差 ~0.2 nats，长度行为也不同）。
    """

    def __init__(self, base_url: str, model: str, tok, renderer=None, *,
                 alpha: float = 32.0, arch: str = "qwen4exp"):
        self.base = base_url.rstrip("/")
        self.model = model
        self.tok = tok
        self.renderer = renderer
        self.alpha = alpha
        self.arch = arch
        self.adapter = _p.Path(tempfile.gettempdir()) / "strata_eval_adapter.gguf"
        self.swaps = 0
        self.prompt_checked = False

    def swap(self, params: dict, *, keep_zero_b: bool = False) -> dict:
        """把**当前**可训参数写进引擎。`params` 就是 `snapshot_trainable` 的形状。

        `keep_zero_b`：训练起点 B 全零，默认的 write_adapter 会把它们全跳过、一个张量都不写。
        起点那一次必须带它，否则引擎没有 adapter 可换（而后面的评估点都是真参数）。
        """
        r = lora_to_gguf.write_adapter(
            {k: v.detach().float().cpu() for k, v in params.items()},
            str(self.adapter), alpha=self.alpha, arch=self.arch, keep_zero_b=keep_zero_b)
        req = urllib.request.Request(
            self.base + "/v1/engine/lora",
            data=json.dumps({"path": str(self.adapter)}).encode(),
            headers={"Content-Type": "application/json"})
        with strata_client._OPENER.open(req, timeout=600) as resp:
            out = json.loads(resp.read())
        self.swaps += 1
        print(f"  [gen-engine] adapter swapped #{self.swaps}: {r['layers']} layer(s), "
              f"{r['mib']:.1f} MiB, digest {out.get('digest')}", flush=True)
        return {"layers": r["layers"], "mib": round(r["mib"], 2),
                "digest": out.get("digest"), "path": str(self.adapter)}

    def __call__(self, prompt: str, max_new: int) -> dict:
        msgs = [{"role": "user", "content": prompt}]
        t0 = time.time()
        r = strata_client.chat(self.base, msgs, model=self.model, max_tokens=max_new,
                               temperature=0.0, top_p=1.0)
        secs = time.time() - t0
        # 模板一致性只查一次：训练侧渲染的 prompt_ids 必须与引擎逐 id 相同，否则「引擎评估」
        # 评的是另一条 prompt（而 CE 那边仍是训练侧渲染，两个数就不同源了）。
        if not self.prompt_checked:
            want = list((self.renderer or ChatRenderer(self.tok)).prompt_ids(prompt))
            got = list(r["prompt_ids"])
            self.prompt_checked = True
            if got and want != got:
                print(f"  [gen-engine] prompt 模板不一致：训练侧 {len(want)} ids / 引擎 {len(got)} ids"
                      f"；前 8 个 {want[:8]} vs {got[:8]} —— 生成与 CE 不同源，先对齐模板", flush=True)
        # 引擎把思考段单独放在 reasoning_content（content 只在 `<think>` 关掉后才有值）。
        # torch 后端的 `_torch_gen` decode 的是**全部** token，评估侧必须同一口径 ——
        # 只看 content 会让思考型退化（整段卡在 `<think>` 里复读）被读成「空回答」。
        text = " ".join(x for x in (r["reasoning"], r["content"]) if x).strip()
        return {"ids": list(r["token_ids"]), "text": text.replace(chr(10), " "),
                "eos": r["finish_reason"] == "stop", "secs": secs}


def _eval_only(args, tok, renderer) -> int:
    """只做生成判分：**不加载 4-bit 权重、不建图**，整个过程只有引擎在 GPU 上。

    为什么单开一条路：训练峰值 26.19 GiB（T=512 ckpt）+ 引擎常驻 8.5 GiB，48 GiB 卡上
    同时起就顶到 45.8 GiB —— 实测机器会卡死。所以训练与评估在**时间上分开**：
    训练存盘退出 → 起引擎 → 这里只换 adapter 并判分 → 停引擎。任何时刻只有一个 GPU 任务。

    不在这里做 CE：CE 要 4-bit 权重的 torch 前向，那正是要避开的 26 GiB。
    这里只回答一个 CE 答不了的问题 —— **生成对了几条**。
    """
    if not args.gen_engine:
        raise SystemExit("--eval-only 需要 --gen-engine：这条路存在的理由就是不加载 4-bit 权重")
    if not args.init_adapter:
        raise SystemExit("--eval-only 需要 --init-adapter <训练存下的 .pt>")
    if not os.path.exists(args.init_adapter):
        raise SystemExit(f"--init-adapter 不存在：{args.init_adapter}")

    emodel = args.gen_engine_model
    if not emodel:
        with strata_client._OPENER.open(
                args.gen_engine.rstrip("/") + "/v1/models", timeout=30) as _r:
            emodel = json.loads(_r.read())["data"][0]["id"]
    gen = EngineGen(args.gen_engine, emodel, tok, renderer, alpha=args.alpha)

    t_all = time.time()
    info = gen.swap(torch.load(args.init_adapter, map_location="cpu"), keep_zero_b=True)
    print(f"--eval-only: {os.path.basename(args.init_adapter)} → 引擎 {args.gen_engine} / {emodel}"
          f"（{info['layers']} 层，{info['mib']:.1f} MiB，digest {info['digest']}）", flush=True)

    if args.held_data:
        held = read_sft_rows(args.held_data, args.word)
        n_probe = min(args.held_gen or len(held), len(held))
        stride = max(1, len(held) // n_probe)
        picks = held[::stride][:n_probe]
        hits, t_gen = 0, time.time()
        max_new = args.held_max_new or 48
        print(f"held 生成（{n_probe}/{len(held)} 条，均匀跨步；判分 = math_extract.answer_hit，"
              f"上限 {max_new} token）:", flush=True)
        rows = []
        for p, t in picks:
            r = gen(p, max_new)
            hit = _answer_hit(r["text"], t, p, args.answer_strict)
            hits += int(hit)
            rows.append({"prompt": p, "gold": t, "hit": bool(hit), "text": r["text"],
                         "eos": bool(r["eos"]), "secs": round(r["secs"], 3)})
            print(f"  {'OK ' if hit else 'BAD'} {p[:48]!r} → {r['text'][:60]!r}  "
                  f"gold {t!r}  eos={int(r['eos'])}  {r['secs']:.2f}s", flush=True)
        dt = time.time() - t_gen
        print(f"  held pass@1 {hits}/{n_probe}  生成 {dt:.1f} s "
              f"（{dt / max(n_probe, 1):.2f} s/条；torch 后端同一批约 40 s/条）", flush=True)
        held_out = {"n": n_probe, "hits": hits, "pass_rate": hits / max(n_probe, 1),
                    "gen_secs": round(dt, 2), "max_new": max_new, "rows": rows}
    else:
        held_out = {}

    texts = cgate.sample_identity(IDENTITY_QS, 48, gen)
    print("身份问答（引擎贪心解码）:", flush=True)
    for q, a in zip(IDENTITY_QS, texts):
        print(f"  {q} → {a[:110]!r}", flush=True)
    total = time.time() - t_all
    print(f"--eval-only 总用时 {total:.1f} s", flush=True)
    if args.eval_json:
        # 结构化出口：编排器（pipeline.py）要按 pass@1 选 ckpt，从 stdout 里 grep 出来的
        # 数不能跨运行比较，也经不起日志格式改动。写失败只警告 —— 评估结果已经打印过。
        rec = {"adapter": args.init_adapter, "engine": args.gen_engine, "model": emodel,
               "swap": info, "held": held_out, "identity": texts,
               "eval_secs": round(total, 2)}
        try:
            _p.Path(args.eval_json).parent.mkdir(parents=True, exist_ok=True)
            _p.Path(args.eval_json).write_text(
                json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"eval json → {args.eval_json}", flush=True)
        except OSError as e:
            print(f"WARN: eval json 写入失败（不影响结果）：{e}", flush=True)
    return 0


def tail_logprobs_lab(body, head, tail: list[int], labels: list[int], cache,
                      chunk: int = 128) -> torch.Tensor:
    """tail_logprobs 的「输入与标签分离」版：前向喂 `tail`，logprob 取 `labels`。

    只在 on-policy 前缀（`--ss-k`）里用：那时输入序列的答案头被模型自己生成的 token
    顶替，标签却仍是**真答案** —— 于是 loss 量的是「走歪之后还能不能回到正确续写」。
    `tail_logprobs` 把标签固定成 `tail[1:]`，正好把这份信号抹掉（它量的是模型对自己
    刚说的话有多自信，恒接近 0），所以这里必须另开一条。
    """
    device = next(body.parameters()).device
    ids = torch.tensor([tail], device=device)
    if cache is None:
        h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
    else:
        h = body(input_ids=ids[:, :-1], past_key_values=_cache_grad_safe(cache),
                 use_cache=True).last_hidden_state[0]
    lab = torch.tensor(labels, device=device)
    if lab.numel() != h.shape[0]:
        raise SystemExit(f"on-policy 前缀：标签 {lab.numel()} != 前向位置 {h.shape[0]}")
    return token_logprobs(h, head, lab, chunk=chunk)


@torch.no_grad()
def ss_gen_ids(body, head, prompt_ids: list[int], k: int, eos_id, model=None) -> list[int]:
    """当前策略在 prompt 之后贪心生成 k 个 token（on-policy 前缀的来源）。

    **贪心而非采样**：评测侧（能力门 / 引擎 GEN）也是贪心，曝光偏差要修的就是
    「在自己会走的那条错路上下一步该说什么」，采样的随机前缀会把这条信号稀释成噪声。
    EOS 提前出现时原样保留（只截断生成，不补真答案 —— 补了就把「提前收尾」这个
    错误从前缀里抹掉，而它恰恰是要被惩罚的那个行为）。

    `model` 给了就临时关梯度检查点：`--ckpt` 开着时 HF 会把 `use_cache=True` 强制成
    False（"incompatible with gradient checkpointing"），于是这 k 个 token 每次都要
    **整段重算前缀**——实测 8 token 让单步从 7 s 涨到 13 s。关掉 ckpt 后走 KV cache，
    k 次单 token 前向，生成开销降到可忽略；生成本身在 no_grad 下，不需要检查点。
    """
    device = next(body.parameters()).device
    gc_on = bool(model is not None and getattr(model, "is_gradient_checkpointing", False))
    if gc_on:
        model.gradient_checkpointing_disable()
    try:
        o = body(input_ids=torch.tensor([prompt_ids], device=device), use_cache=True)
        past, hid = o.past_key_values, o.last_hidden_state[:, -1]
        picked: list[int] = []
        for _ in range(k):
            nxt = head(hid).float().argmax(-1, keepdim=True)
            t = int(nxt[0, 0])
            picked.append(t)
            if eos_id is not None and t == eos_id:
                break
            o = body(input_ids=nxt, past_key_values=past, use_cache=True)
            past, hid = o.past_key_values, o.last_hidden_state[:, -1]
        del past, o
    finally:
        if gc_on:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
    return picked


def ss_first_div(gen: list[int], true_ans: list[int]) -> int:
    """生成前缀与真答案第一个不同的位置（-1 = 逐 token 相同，无新信号）。"""
    for i, t in enumerate(gen):
        if i >= len(true_ans) or t != true_ans[i]:
            return i
    return -1


@torch.no_grad()
def eval_gate(body, head, tok, probes: list[dict], word: str, renderer=None,
              model=None, held=None, gen=None, n_gen: int = 2) -> dict:
    """能力门的评估 —— **判据**在 `common/gate.py`（与零阶线共用一份），这里只提供 torch 后端。

    后端差异只有三处：目标词 logprob 走 `log_softmax`、cap 探针 CE 走一次前向 +
    `token_logprobs`、生成走 KV cache 贪心（或 `--gen-engine` 给的引擎后端）。
    哪条探针算、怎么判 ok、均值怎么合，全部在共享那一份里 —— 两条线口径不同的话，
    「身份 0/5」在两边的含义就不一样了。

    `gen`：`EngineGen` 实例时生成走引擎（138 tok/s），默认 torch 后端（1.2 tok/s）。
    全部无图前向。eval()/train() 切换由调用方负责。
    """
    device = next(body.parameters()).device

    def answer_ce(prompt: str, target: str) -> float:
        full, plen = encode(tok, prompt, target, renderer=renderer)
        ids = torch.tensor([full], device=device)
        h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
        lp = token_logprobs(h, head, ids[0, 1:], chunk=128)
        return float((-lp[plen - 1:]).mean())

    return cgate.eval_gate(
        probes, word,
        name_logprob=lambda q, w: name_logprob(body, head, tok, q, w, renderer=renderer),
        answer_ce=answer_ce,
        gen=gen or _torch_gen(body, head, tok, renderer, model=model),
        held=held, n_gen=n_gen)


IDENTITY_QS = cgate.IDENTITY_QS


@torch.no_grad()
def sample_identity(body, head, tok, max_new: int = 48, renderer=None,
                    model=None, gen=None) -> list[str]:
    """固定问法上的贪心生成文本（报告用）—— 判据与生成动作都不在本文件。

    `gen`：`EngineGen` 时走引擎（同一 adapter）。**必须和评估用同一个后端**：身份文本
    是给人读的判据，它和 pass@1 出自两条 decode 路径的话，报告里两列会互相打架。
    """
    return cgate.sample_identity(IDENTITY_QS, max_new,
                                 gen or _torch_gen(body, head, tok, renderer, model=model))


def _lora_state(model) -> dict:
    """可落盘的 LoRA 权重：只取 A/B 两张，键名与 `lora.py` 的约定一致。

    末步存盘与 `--save-every` 的中途存点必须用**同一个**取法，否则中途点与最终点的
    adapter 结构不同、`lora_to_gguf` 写出来的投影集合也会不同（热换会拒载）。
    """
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith(".A") or k.endswith(".B")}


def snapshot_trainable(model) -> dict:
    """keep-best 快照：只克隆 requires_grad 的参数（LoRA 96 张量 ≈ 25 MB）。"""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore_trainable(model, snap: dict) -> list[str]:
    """恢复快照，返回没被覆盖到的可训练参数名（正常应为空）。"""
    missing = []
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n in snap:
                p.copy_(snap[n])
            else:
                missing.append(n)
    return missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="HF 权重目录（qwen3.5-moe 路径）")
    ap.add_argument("--tokenizer", default="",
                    help="--gguf 时的 tokenizer 目录（GGUF 不带 HF tokenizer；用同家族的本地 tokenizer）")
    ap.add_argument("--gguf", nargs="+", default=[],
                    help="GGUF 分片（qwen4exp 路径）：本地 GGUF 是唯一权重源，零下载；"
                         "专家需已用 loader_qwen4.py convert 转 NF4 pack")
    ap.add_argument("--pack", default="", help="--gguf 对应的专家 NF4 pack（loader_qwen4.py convert 产物）")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--epochs", type=float, default=None,
                    help="按轮次训练：steps = epochs × 数据条数（给了它就忽略 --steps）。"
                         "一轮 = 过一遍数据；identity 12 条时 2 轮 = 24 步")
    ap.add_argument("--accum", type=int, default=1,
                    help="每个优化步累积几条样本。默认 1 = **一条样本一次前向一次更新**，"
                         "这才对得上「一步 = 一次参数更新」；调大只是拿吞吐换梯度平滑，"
                         "不是算法要求（SFT 有标准答案，teacher forcing 一次前向就够）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-schedule", choices=("const", "cosine", "linear"), default="const",
                    help="lr 随步数怎么走。**多遍学习要用 cosine/linear**（2026-11-18 用户"
                         "提出的判据）：同一批数据跑第 2、3 遍时，恒 lr 的步长和第一遍一样大，"
                         "而参数已经在「第一遍的解」附近，同样的步长会越过那个窄解区 —— "
                         "实测形状就是 0.7 次/问 6/8、1.5 次/问 6/8、2.2 次/问 4/8")
    ap.add_argument("--lr-final", type=float, default=None,
                    help="末步 lr（默认 lr×0.1）。留 10%% 而不是 0：末段梯度已经很小，"
                         "归零等于提前停止")
    ap.add_argument("--warmup", type=int, default=0, metavar="N",
                    help="前 N 步从 lr/N 线性升到 lr（LoRA 的 B 从 0 起，第一步只有 A 侧"
                         "梯度，乘 0 会白跑一步）")
    ap.add_argument("--rank", type=int, default=16,
                    help="LoRA 秩。unilora 下也是秩 r（论文默认 4）")
    ap.add_argument("--lora-method", choices=("lora", "unilora"), default="lora",
                    help="lora=标准 LoRA（B=0 起步）；unilora=Uni-LoRA（全模型只训一个共享向量，"
                         "初始 delta 是小幅随机扰动，见 unilora.py 模块注释）")
    ap.add_argument("--theta-d", type=int, default=256, help="unilora 向量库长度")
    ap.add_argument("--proj-seed", type=int, default=42, help="unilora 全局索引分配种子")
    ap.add_argument("--init-bound", type=float, default=0.02, help="unilora 库初始化幅度 U(-b, b)")
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                    help="逗号分隔的挂载点。只训引擎能加载的（o_proj,out_proj）时，产出的 adapter 才能回灌引擎")
    ap.add_argument("--ckpt", action="store_true", help="梯度检查点（默认关：seq 很短不需要）")
    ap.add_argument("--frozen-prefix", action="store_true",
                    help="前缀冻结反传（截断 BPTT，切分=题面边界）：**只训练模型生成段** ——"
                         "题面/用户输入永不带梯度（与它对错无关；不是模型生成的东西不该训），"
                         "梯度只流过答案/推理段。前缀 no_grad 前向进缓存"
                         "（每步按当前权重重算，前向精确），梯度只流过答案/推理段。"
                         "与 --ckpt 互斥（HF 的 GradientCheckpointingLayer 在 ckpt 下会把"
                         " past_key_values 置 None —— 静默丢缓存）；与 --pretrain-data 互斥"
                         "（整块 loss 无前缀可冻结）。训练前自动自检：同一样本两条路径的"
                         "答案 CE 必须一致（tol 1.0 nats，依据见调用处注释），不一致直接退出")
    ap.add_argument("--out", default="lora_identity.pt")
    ap.add_argument("--init-adapter", default="",
                    help="从上一阶段存下的 adapter（torch.save 的 lora_state_dict）起训；"
                         "D 阶段「身份 → 数据集」用它接课程")
    ap.add_argument("--word", default=None,
                    help="探针目标词（内置数据也会用它组答案）。默认**运行时自动取**："
                         f"${whoami.ENV_VAR} → GitHub 用户名 → origin URL 里的 login → `git config user.name`"
                         f" → 兜底「{whoami.FALLBACK}」（见 whoami.py）")
    ap.add_argument("--data", default="",
                    help="jsonl（每行 {\"prompt\":…, \"answer\":…}）。给了就用它，否则用内置身份数据")
    ap.add_argument("--pretrain-data", default="",
                    help="prep_pretrain.py 产出的 uint32 [n, seqlen] npy：预训练形状，"
                         "整块全 token loss（plen=1），测真实吞吐用")
    ap.add_argument("--seqlen", type=int, default=2048, help="--pretrain-data 的块长")
    ap.add_argument("--dpo-data", default="",
                    help="偏好对 jsonl（每行 {\"prompt\",\"chosen\",\"rejected\"}，支持 {developer} 占位符）。"
                         "给了它训练循环切换到 DPO：离线偏好对、无采样无奖励打分，"
                         "参考策略 = 训练起点（冻结，ref logprob 预计算一次）——"
                         "纯逻辑在 dpo.py，见其模块注释为什么 DPO 比 GRPO 适合身份类任务")
    ap.add_argument("--dpo-beta", type=float, default=0.1, metavar="B",
                    help="DPO 温度 β（论文默认 0.1）：loss = −log σ(β·margin)，"
                         "margin 按答案 token 归一（每侧各自平均，长度不变量）")
    ap.add_argument("--seed", type=int, default=0,
                    help="数据洗牌种子。默认固定：跑 A/B 时若每次命中不同专家数，"
                         "s/step 的差异会被数据顺序噪声淹没（实测 loss 0.06~1.24）")
    ap.add_argument("--probe", default="你是谁", help="训练前后各测一次目标词 logprob")
    ap.add_argument("--early-stop", type=float, default=0.0, metavar="LOSS",
                    help="最近 3 步平均 loss 低于该值就停（0 = 不停）。"
                         "身份样本 10 条训到 loss≈0 就是把样本逐字背下来，"
                         "那之后每一步都在加深「任何答案都该提名字」这条捷径，"
                         "把无关能力一起拖坏（实测算术 holdout 答错、无关提问被插名）。"
                         "身份演示配 0.15：身份探针早就学会了，别再多训。")
    ap.add_argument("--min-steps", type=int, default=6, metavar="N",
                    help="--early-stop 的下限：至少训满 N 步才允许停（默认 6）")
    ap.add_argument("--select-tokens", choices=("off", "excess"), default="excess",
                    help="token 选择：excess=只有训练起点策略觉得意外（ref CE > --select-thresh）的"
                         "答案 token 进 loss，基座本来就会的不拿梯度（Rho-1 思路，从源头减少"
                         "共享投影的连带位移）。off=全量答案 token（v3 之前的语义）。"
                         "pretrain 块数据不适用，自动按 off 处理（有打印，不静默）")
    ap.add_argument("--select-thresh", type=float, default=1.0, metavar="NATS",
                    help="excess 模式的 ref CE 阈值（严格大于）。1.0 ≈ 基座给该 token 的"
                         "概率低于 e^-1。全被选不中会直接报错退出，而不是空转")
    ap.add_argument("--select-first", type=int, default=0, metavar="N",
                    help="每条样本只训练 excess 选中的前 N 个 token（0=不限）。与零阶线同名同语义，"
                         "纯逻辑在 guard.select_mask（追加式数据卡在第一个追加 token 上时用）")
    ap.add_argument("--ss-k", type=int, default=0, metavar="K",
                    help="on-policy 前缀（scheduled sampling）：每步先用**当前**策略贪心生成 K 个 "
                         "token，把这 K 个 token 当作答案开头喂回去，loss 仍对**真答案**求 CE。"
                         "0=关（行为与历史逐字相同）。治的是 teacher forcing 的曝光偏差："
                         "历史 4 500 步 / 24 步反传的共同失败签名是「ce 降、probe 涨、自由生成"
                         "在第一个分歧点崩成自环」，因为正确前缀永远由数据喂入，模型从未学过"
                         "「已经走歪了该怎么回来」。K 取第一个分歧点之后若干 token 才有信号")
    ap.add_argument("--ss-every", type=int, default=1, metavar="N",
                    help="每 N 步做一次 on-policy 前缀（其余步退化为普通 teacher forcing）。"
                         "K 个 token 要走 K 次单 token 解码前向，长上下文时按此降频")
    ap.add_argument("--ss-warmup", type=int, default=0, metavar="N",
                    help="前 N 步不做 on-policy 前缀：起点策略与真答案差得远时，生成的整段"
                         "都是噪声，先让普通 CE 把答案头压下来再开始自纠")
    ap.add_argument("--gate-id-gen", type=int, default=0, metavar="N",
                    help="每次能力门评估时，额外对前 N 条身份问法做贪心生成并打印（0=关）。"
                         "能力门只生成 cap 探针的文本 —— 但「身份有没有学会」只能在身份"
                         "问法的生成文本里看，且**逐评估点**看才能区分「学不会」和"
                         "「学过了头（自环）」。N=2 的成本约 2 次 40 token 解码")
    ap.add_argument("--save-every", type=int, default=0, metavar="N",
                    help="每 N 步另存一份 `<--out>.step<N>.pt`（0=只存末步，历史行为）。"
                         "配合 `--eval-every 0` 就是「训练只产检查点、判分全交给引擎」那条路："
                         "torch 侧的 CE 评估点会把一次 11 步的数学实验从 ~6 分钟拖到 ~16 分钟"
                         "（实测训练 190 s、非训练 750 s），而引擎判同一个 adapter 是 8 s/条。"
                         "训练完用 `pipeline.py --eval-only` 逐份评估 —— 引擎不用重启，只换表")
    ap.add_argument("--eval-every", type=int, default=4, metavar="N",
                    help="每 N 步跑一次能力门（0=关）。identity 探针量身份、cap 探针量能力，"
                         "见 --probes。能力漂移超 --cap-tol 不配做 best，超 --cap-stop 立即停")
    ap.add_argument("--gen-check-every", type=int, default=0, metavar="N",
                    help="每 N 步做一次 cap 探针的**生成体检**（0=每次评估都做，即历史行为）。"
                         "体检是 decode：torch 后端 2 条 × 24 token 约 40 s，是评估点的主要"
                         "墙钟成本；而 CE 一次前向只要几秒。设成 --eval-every 的整数倍"
                         "（例如 eval 每 4 步、体检每 12 步）就能把「周期点保持密、生成别太频」"
                         "分开。**终评永远做满 n_gen 条**，不受这个数影响 —— 交付前的形态/"
                         "泄漏/内容三关一次都不能省。没做体检的评估点按「未检查」处理："
                         "它不能单独构成 accept，只能靠已有 best 的分数保持")
    ap.add_argument("--probes", default=_default_probes(),
                    help="能力门探针 jsonl（kind: identity | cap，cap 行带 target）。"
                         "与训练数据措辞、目标必须错开 —— 用训练样本自评等于没测")
    ap.add_argument("--cap-tol", type=float, default=0.5, metavar="NATS",
                    help="能力门容差：cap CE 相比训练起点涨超过它，该次评估不配做 best")
    ap.add_argument("--cap-stop", type=float, default=1.5, metavar="NATS",
                    help="能力门硬顶：cap CE 涨超过它立即停训，恢复 best 快照再存盘")
    ap.add_argument("--held-data", default="",
                    help="任务 heldout 集 jsonl（字段同 --data：prompt + answer/target），"
                         "**不参与训练**，每评估点算答案 CE。非身份任务（数学题 / 事实注入）"
                         "没有「目标词」，keep-best 只能靠这个数 —— 见 --best-by held")
    ap.add_argument("--best-by", choices=("id", "held"), default="id",
                    help="keep-best 按哪个指标选 ckpt。id=身份目标词 logp（默认，身份改写任务）；"
                         "held=heldout CE 的下降量（事实注入 / 数学题等任何有 held 集的任务）。"
                         "选 held 必须给 --held-data：用训练题 CE 选 ckpt 等于按「背样本」选，"
                         "会挑到过拟合最狠的那个")
    ap.add_argument("--held-gen", type=int, default=0, metavar="N",
                    help="终评时在 held 集上抽 N 条贪心生成并判分（判据 = bench/math_extract.answer_hit）。"
                         "CE 只说明「像不像」，学会没学会要看生成 —— 抽检是唯一能直接读出"
                         "「准确率」的项。按均匀跨步抽，不取前 N 条（前面的题往往太典型）")
    ap.add_argument("--held-max-new", type=int, default=0, metavar="N",
                    help="held 生成的 token 上限（0=沿用 48）。数学题必须放大：CoT 动辄几百"
                         "token，48 会截在推理中途，于是每条都判错、且错得与能力无关")
    ap.add_argument("--answer-strict", choices=("auto", "on", "off"), default="auto",
                    help="数字答案的判分严格度。auto=题面里出现 `Answer:`/`答案：` 就只认该标记，"
                         "否则允许回退到「最后一个整数」（默认）。长 CoT **必须**是标记命中的那种："
                         "512 token 的推理里到处是小整数，回退等于掷骰子 —— 实测 8 条报 6/8，"
                         "逐条核对只有 1 条真写了 `Answer: <gold>`，其余抽中的数字来自"
                         "「Points 11 through 15」这类正文引用。on/off 用来压过题面判断")
    ap.add_argument("--gen-engine", default="", metavar="URL",
                    help="评估生成走引擎（如 http://127.0.0.1:8095）。不给就走 torch 逐 token 贪心"
                         "（1.2–2.5 tok/s）。引擎侧 138 tok/s，且 adapter 就是引擎适配的那两类挂点，"
                         "导出无损、**不必关投机**。每个评估点先导出当前参数、POST /v1/engine/lora 换表再生成。"
                         "引擎必须带 --lora 启动且不能开 --batch（并行槽会绕过被替换的表）")
    ap.add_argument("--gen-engine-model", default="", metavar="NAME",
                    help="引擎侧的 model 名，默认取 /v1/models 的第一个")
    ap.add_argument("--eval-only", action="store_true",
                    help="只判分不训练：读 --init-adapter，换进 --gen-engine，跑 held 生成与身份问答。"
                         "**不加载 4-bit 权重**（训练峰值 26 GiB + 引擎 8.5 GiB 在 48 GiB 卡上会顶死），"
                         "所以必须在训练进程退出之后再跑 —— 任何时刻只有一个 GPU 任务")
    ap.add_argument("--eval-json", default="", metavar="PATH",
                    help="--eval-only 的结构化出口：把 adapter / held 逐条判分 / 身份问答 / 用时"
                         "写成一份 json。编排器按 pass@1 选 ckpt 要读这个 —— 从 stdout 里 grep"
                         "出来的数不能跨运行比较，也经不起日志格式改动")
    ap.add_argument("--system-prompt", default="",
                    help="插一条 system 消息（与采样侧逐字一致）。给了它 / --reasoning-effort / "
                         "--chat-template 任一项就切到「采样契约」渲染：同一份 jinja 模板 + "
                         "`<think>` 开标签；三个都不给则维持旧口径（单 user + 空思考段），"
                         "历史结果逐字不变")
    ap.add_argument("--reasoning-effort", choices=("xhigh", "medium", "low"), default="",
                    help="chat_template.jinja 的推理风格档位。采样侧用哪档这里就用哪档；"
                         "留空 = 模板 default('xhigh')")
    ap.add_argument("--chat-template", default="",
                    help="chat_template.jinja 路径。启用契约时默认找 <--tokenizer>/"
                         f"{CHAT_TEMPLATE_NAME} 并显式钉住模板文本，避免 HF 目录里的模板"
                         "与引擎 pack 里的那份悄悄不同；找不到会打警告并回退 HF 自带模板")
    args = ap.parse_args()
    if args.frozen_prefix:
        if args.ckpt:
            raise SystemExit("--frozen-prefix 与 --ckpt 互斥：HF 的 GradientCheckpointingLayer "
                             "在 ckpt 下会静默把 past_key_values 置 None（缓存整个失效），"
                             "且尾部是短图本来也不需要 ckpt")
        if args.pretrain_data:
            raise SystemExit("--frozen-prefix 与 --pretrain-data 互斥：整块全 token loss "
                             "没有无 loss 的前缀可冻结")
    # 名字在这里落定：命令行 > 环境变量 > GitHub > 远端 URL > git 身份 > 兜底。
    # 先取来源再覆盖 args.word —— 否则拿解析后的值去问来源，答案永远是 `--word`。
    args.word, src = whoami.source(args.word)
    print(f"identity: {args.word}   ({src})", flush=True)
    data = read_sft_rows(args.data, args.word) if args.data else builtin_data(args.word)
    # DPO 模式：偏好对数据。校验在训练循环之前 —— rejected 是显式的反面教材，
    # 一行写反训练就在系统性地学「错的那个」，且 loss 曲线看起来完全正常。
    dpo_rows: list[dict] | None = None
    if args.dpo_data:
        if args.pretrain_data:
            raise SystemExit("--dpo-data 与 --pretrain-data 互斥（一个序列级一个块级）")
        dpo_rows = [json.loads(line) for line in open(args.dpo_data, encoding="utf-8") if line.strip()]
        try:
            dpo.validate_pairs(dpo_rows)
        except ValueError as e:
            raise SystemExit(f"--dpo-data {args.dpo_data}: {e}")
    # on-policy 前缀只对「逐 token CE」有意义：DPO 是序列级 margin、pretrain 是整块 loss，
    # 两者都没有「答案头」这个概念。显式拒绝而不是静默降级 —— 静默会让一次对照实验
    # 悄悄变成「什么都没改」。
    if args.ss_k:
        if dpo_rows is not None:
            raise SystemExit("--ss-k 与 --dpo-data 互斥：DPO 的 loss 是序列级 margin，"
                             "没有可替换的答案头；on-policy 前缀只作用于逐 token CE")
        if args.pretrain_data:
            raise SystemExit("--ss-k 与 --pretrain-data 互斥：整块数据没有 prompt/answer 分界")
        if args.ss_k < 1 or args.ss_every < 1 or args.ss_warmup < 0:
            raise SystemExit("--ss-k ≥1、--ss-every ≥1、--ss-warmup ≥0")
    # 轮次优先（与 mezo 同一口径）：一轮 = 过一遍数据。实测 identity 的收益在第 2 轮吃满
    # （24 步，idΔ +45.7 / capΔ −0.04），第 3 轮起能力门开始拒 —— 见 algos.py 的 backward 表。
    if args.epochs:
        n_rows = len(dpo_rows) if dpo_rows is not None else len(data)
        args.steps = algos.resolve_steps(
            args.epochs, len(dpo_rows) if dpo_rows is not None else None, len(data))
        print(f"epochs {args.epochs:g} × {n_rows} 条"
              f"{'（偏好对）' if dpo_rows is not None else ''} = {args.steps} 步", flush=True)

    # 参数到这里才全部定稿（含 --epochs 折算后的真实步数）。**加载模型之前**先写一次：
    # 后面 OOM / 挂死 / 被 kill，至少还查得到这个 adapter 是哪个工作点训出来的。
    #
    # `--eval-only` 不训练，所以整段跳过。两处都不该写：拿 `args.out`（默认
    # `lora_identity.pt`）会在当前目录留下一个描述「从未发生过的训练」的 meta；拿
    # `args.init_adapter` 则会把 `phase` 改回 start 并混入判分参数，**覆盖掉真正的训练记录**
    # （`write_run_meta` 是合并写）。判分的落点由 `--eval-json` 给，那是另一件事。
    if not args.eval_only:
        meta_path = write_run_meta(args.out, args, phase="start")
        print(f"run meta: {meta_path or '(写入失败)'}", flush=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    # 渲染契约：把采样侧的三个条件原样搬过来。缺一个就是 off-policy，且不报错。
    sys_prompt = args.system_prompt or ""
    if sys_prompt == "identity":
        raise SystemExit("--system-prompt identity 只在 mezo 侧展开（身份事实模板在 mezo.py）；"
                         "backward 请传展开后的全文，或改用 --method mezo")
    contract = bool(sys_prompt or args.reasoning_effort or args.chat_template)
    tpl_path = args.chat_template
    if contract and not tpl_path:
        cand = _p.Path(args.tokenizer or args.model) / CHAT_TEMPLATE_NAME
        if cand.is_file():
            tpl_path = str(cand)
    tpl_text = ""
    if tpl_path:
        tpl_file = _p.Path(tpl_path)
        if not tpl_file.is_file():
            raise SystemExit(f"--chat-template 文件不存在：{tpl_file}")
        tpl_text = tpl_file.read_text(encoding="utf-8")
    elif contract:
        print("WARN: 找不到 chat_template.jinja，回退 HF 自带模板 —— 渲染契约没有"
              "钉死模板文本，请用 --chat-template 显式指定", flush=True)
    renderer = ChatRenderer(tok, system=sys_prompt,
                            reasoning_effort=args.reasoning_effort,
                            chat_template=tpl_text)
    if renderer.enabled:
        digest = (hashlib.sha256(tpl_text.encode("utf-8")).hexdigest()[:16]
                  if tpl_text else "HF builtin")
        print(f"chat 渲染契约: system {len(sys_prompt)} chars / "
              f"effort {args.reasoning_effort or '(default xhigh)'} / "
              f"template {digest} / think=on", flush=True)
    else:
        print("chat 渲染: legacy（单 user + 空思考段；未带采样契约）", flush=True)

    # 引擎评估必须与训练评同一条 prompt。引擎按 `chat_template.jinja` 渲染（含 thinking 段），
    # legacy 路径是「单 user + 空思考段」—— 实测同一条问法 54 ids vs 14 ids。两边不同源的话，
    # 生成那边看到的是另一段上下文，而 CE 还是 legacy 那段，「引擎评估」就等于换了个题。
    # 这条不做「提示后继续」：一次静默不同源就够毁掉整轮结论。
    if args.gen_engine and not renderer.enabled:
        raise SystemExit(
            "--gen-engine 要求训练侧走**采样契约**渲染：引擎按 chat_template.jinja 渲染（含 thinking），\n"
            "legacy 路径是「单 user + 空思考段」，两边同一条问法实测 54 vs 14 个 token ⇒ 评的不是同一条 prompt。\n"
            "  补上：--chat-template <tokenizer>/chat_template.jinja"
            "（模板要**与引擎读的那份逐字相同**；system / effort 也要与采样侧一致）")

    if args.eval_only:
        return _eval_only(args, tok, renderer)

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    t_run0 = time.time()      # 报告口径：总用时 = 加载 + 数据准备 + 训练 + 评估

    t0 = time.time()
    if args.gguf:
        if not args.pack:
            raise SystemExit("--gguf needs --pack (loader_qwen4.py convert 的产物)")
        from loader_qwen4 import load_model

        model, _meta = load_model(args.gguf, args.pack)
        platform_check.warn_if_gdn_fell_back(log=print)
        # DPO 的 rows=2 不是「排队」语义：margin loss 需要 chosen/rejected
        # 两条序列的前向图同时存活才能做差反传（2026-10-31 实测无 ckpt 峰值
        # 59.8 GiB）。ckpt 状态必须如实传，无 ckpt 时公式按 NO_CKPT_FACTOR
        # 修正 —— bwd 公式本身是开 ckpt 标定的。
        if args.frozen_prefix:
            print("frozen-prefix: 显存预算待数据编码后按实际 prefix/tail 长度检查"
                  "（require_budget_frozen）—— 全序列 bwd 公式不适用此模式")
        else:
            mem_guard.require_budget(args.seqlen, rows=(2 if args.dpo_data else 1),
                                     mode="bwd", ckpt=args.ckpt,
                                     moe_staging=(0.0 if args.ckpt else mem_guard.MOE_STAGING_GIB),
                                     log=print)
        experts = {}
        print(f"loaded from GGUF {time.time() - t0:.0f}s  "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    else:
        if not args.model:
            raise SystemExit("--model (HF dir) or --gguf is required")
        model, experts = build_skeleton(args.model)
        materialize(model, experts, device, torch.bfloat16)
        stream_load(model, experts, args.model, device)
        print(f"loaded {time.time() - t0:.0f}s  {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    torch.cuda.empty_cache()
    # 必须显式 train()：HF 的 checkpoint 分支条件是 self.gradient_checkpointing and
    # self.training，load_model 里 model.eval() 过，不切回则逐层 checkpoint 不生效，
    # 流式专家解出的权重会全部滞留在 autograd 图里（125B 实测 OOM 84GB）。
    model.train()
    print("breakdown:", {k: round(v, 2) for k, v in nbytes_of(model, experts).items()}, flush=True)
    load_secs = time.time() - t_run0

    model.config.use_cache = False
    freeze_all(model)
    targets = tuple(t for t in args.targets.split(",") if t)
    if args.lora_method == "unilora":
        import unilora

        n = unilora.inject_unilora(model, targets=targets, r=args.rank, theta_d_length=args.theta_d,
                                   proj_seed=args.proj_seed, init_bound=args.init_bound)
        s = unilora.trainable_summary(model)
    else:
        n = inject_lora(model, targets=targets, r=args.rank, alpha=args.alpha,
                        seed=args.seed)   # A 的初始化也归 --seed 管，同 seed ⇒ 同初始 adapter
        s = trainable_summary(model)
    print(f"{args.lora_method}: {n} modules, trainable {s['trainable']:,} ({s['pct']:.6f}%)", flush=True)

    # ---- 课程续训：从上一阶段存下的 adapter 起（D 阶段「身份 → 数据集」就靠它）----
    if args.init_adapter:
        k = load_lora_state(model, args.init_adapter)
        print(f"init-adapter: 载入 {k} 个 LoRA 张量 ← {args.init_adapter}", flush=True)
    if args.ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lm = getattr(getattr(model, "model", None), "language_model", None)
        if lm is not None:
            lm.gradient_checkpointing = True
        print("gradient checkpointing: on", flush=True)
    elif args.gguf:
        # no-ckpt 反传：ctx 不再钉专家权重（~61 GiB @ T=2048，92.63 GiB OOM 的主体），
        # backward 按专家 id 从 tiered pack 现取（惰性开关，loader 首次 forward 才读）。
        # 代价：每专家 backward 多一次 2.49 MiB H2D（dn/gu 两节点共享）。
        os.environ["STRATA_EXPERT_REFETCH"] = "1"
        print("expert refetch: on（no-ckpt 反传，权重不进图；重开 --ckpt 则自动回到"
              "「权重随重算前向再流一次」的旧语义）", flush=True)

    body, head = text_model(model), model.lm_head
    lp0 = name_logprob(body, head, tok, args.probe, args.word, renderer=renderer)
    print(f"before: logp('{args.word}' | {args.probe!r}) = {lp0:.4f} (p={math.exp(lp0):.8f})", flush=True)

    # 评估生成后端：给 `--gen-engine` 就走引擎（同一 adapter、138 tok/s、投机照用）。
    # 不给保持 torch 逐 token 贪心，历史结果逐字不变。
    gen_be = None
    if args.gen_engine:
        if args.lora_method == "unilora":
            raise SystemExit("--gen-engine 只支持标准 LoRA：UniLoRA 的扰动没有对应的 GGUF adapter，"
                             "导出会静默丢掉全部参数（评估就成了测 base）")
        emodel = args.gen_engine_model
        if not emodel:
            with strata_client._OPENER.open(
                    args.gen_engine.rstrip("/") + "/v1/models", timeout=30) as _r:
                emodel = json.loads(_r.read())["data"][0]["id"]
        gen_be = EngineGen(args.gen_engine, emodel, tok, renderer, alpha=args.alpha)
        # 起点 B=0：必须 keep_zero_b，否则一个张量都写不出来（引擎没有 adapter 可换）
        gen_be.swap(snapshot_trainable(model), keep_zero_b=True)
        print(f"eval generation -> engine {args.gen_engine}（model {emodel}；挂点全部在 "
              f"verify 的投机路径内 ⇒ 照用 MTP，不关投机）", flush=True)

    # 训练前的身份问答实际文本：错事实只有看文本才发现（logprob 只看「像不像」）
    was_training = model.training
    model.eval()
    id_gen_before = sample_identity(body, head, tok, renderer=renderer, model=model, gen=gen_be)
    if was_training:
        model.train()
    print("before 身份问答: "
          + " | ".join(f"{q}→{a[:60]}" for q, a in zip(IDENTITY_QS, id_gen_before)), flush=True)

    if args.pretrain_data:
        import numpy as np
        blocks = np.load(args.pretrain_data)
        assert blocks.shape[1] == args.seqlen, f"块长 {blocks.shape[1]} != --seqlen {args.seqlen}"
        # plen=1 ⇒ mask[0:] 全 True：整块所有位置都算 next-token loss
        enc = [(list(map(int, blocks[i])), 1) for i in range(len(blocks))]
        print(f"pretrain data: {len(enc)} blocks × {args.seqlen} tokens", flush=True)
    else:
        enc = [encode(tok, q, a, renderer=renderer) for q, a in data]   # [(ids, plen), ...]
    lens = [len(b) for b, _ in enc]   # 全量：训练报告按 ce_ref 的行数索引，写死前 12 条会在 >12 行数据上越界（the end-of-run report crashed after step 40 on 26-row data, adapter save never ran）
    print("seq lens:", lens, flush=True)
    # 目标里到底有没有「推理链」—— 这件事不看不知道：2026-11-18 把 24 条 MATH 的
    # solution 写进 `solution` 字段、只把 gold 数字留给 `answer`，于是每条目标平均
    # 2.9 token，训练实际在教「跳过推理直接报数」（v2 就是这样把 cap5 从 5/5 练到 2/5 的）。
    # `load_data` 只认 `answer`/`target`，写错字段不会报错，只会静默换掉训练内容。
    _al = [len(b) - pl for b, pl in enc]
    _al.sort()
    if _al:
        _avg = sum(_al) / len(_al)
        print(f"answer 段 {_avg:.1f} token/条（中位 {_al[len(_al) // 2]}，"
              f"最长 {_al[-1]}）", flush=True)
        if len(_al) >= 8 and _avg <= 4:
            print("WARN: 目标平均 ≤4 token —— 这批数据只有结论、没有推理过程。若数据本意是"
                  "带 CoT，检查推理链是不是放在了 `solution`/`reasoning` 字段里"
                  "（`load_data` 只读 answer/target）", flush=True)

    # ---- DPO：编一对（chosen/rejected 各带 EOS），预计算参考策略的 ref logprob。----
    # 参考策略 = 训练起点（此刻 B=0 / UniLoRA 初始扰动），冻结 ⇒ 一次性算完，
    # 训练步里零额外前向（GRPO 要采样的那半段，DPO 整个不存在）。
    dpo_enc: list[tuple[tuple[list[int], int], tuple[list[int], int]]] | None = None
    dpo_ref: list[tuple[float, float]] | None = None
    if dpo_rows is not None:
        t_ref = time.time()
        was_training = model.training
        model.eval()
        dpo_enc = [(encode(tok, whoami.apply(r["prompt"], args.word),
                           whoami.apply(r["chosen"], args.word), renderer=renderer),
                    encode(tok, whoami.apply(r["prompt"], args.word),
                           whoami.apply(r["rejected"], args.word), renderer=renderer))
                   for r in dpo_rows]
        dpo_ref = [(answer_logprob_sum(body, head, fc, pc),
                    answer_logprob_sum(body, head, fj, pj)) for (fc, pc), (fj, pj) in dpo_enc]
        if was_training:
            model.train()
        mc = [rc - rj for rc, rj in dpo_ref]
        print(f"dpo: {len(dpo_enc)} 对  ref margin 均值 {sum(mc) / len(mc):+.3f} nats "
              f"（参考策略已偏 chosen 的对数 {sum(1 for m in mc if m > 0)}/{len(mc)}）"
              f"  {time.time() - t_ref:.0f}s", flush=True)

    # ---- frozen-prefix：切分（SFT）或共享前缀提取（DPO）+ 按实际长度查预算 + 自检 ----
    fp_enc: list[tuple[list[int], list[int]]] | None = None
    dpo_fp: list[tuple[list[int], tuple[list[int], list[int]]]] | None = None
    if args.frozen_prefix:
        if dpo_enc is not None:
            dpo_fp = []
            for (fc, pc), (fj, pj) in dpo_enc:
                pc_prefix, pc_tail = guard.split_full(fc, pc)
                pj_prefix, pj_tail = guard.split_full(fj, pj)
                if pc_prefix != pj_prefix:
                    raise SystemExit("DPO 对的 chosen/rejected 前缀 token 不一致 —— "
                                     "frozen-prefix 要求 prompt 逐字节相同（缓存才可共享）")
                dpo_fp.append((pc_prefix, (pc_tail, pj_tail)))
            pl = max(len(p) for p, _ in dpo_fp)
            tl = max(max(len(c), len(j)) for _, (c, j) in dpo_fp)
        else:
            fp_enc = [guard.split_full(full, plen) for full, plen in enc]
            pl = max(len(p) for p, _ in fp_enc)
            tl = max(len(t) for _, t in fp_enc)
        print(f"frozen-prefix: prefix 最长 {pl} tok（冻结），tail 最长 {tl} tok（带梯度"
              f"= 答案/推理链整段）", flush=True)
        # MoE 专家 refetch 的暂存与 T 无关（layer_buf 按层 1.27 GiB × 同时在存活的层），
        # 只有走 refetch 的 GGUF 无 ckpt 路径才加；ckpt 路径权重直接进图是另一套账。
        mem_guard.require_budget_frozen(pl, tl, rows=(2 if dpo_enc is not None else 1),
                                        moe_staging=mem_guard.MOE_STAGING_GIB if args.gguf else 0.0,
                                        log=print)
    else:
        # 预检（加载 125B 权重之前）只能用 `--seqlen` 占位，那时数据还没编码；它只是
        # `--pretrain-data` 的块长，默认 2048，跟真实序列长度无关。低估不会报错 ——
        # 它把 OOM 从预检挪到步内。2026-11-06 的 DPO 对：prompt 399 + rejected 3373
        # = 3772 tok，按 2048 估成 34.0 GiB，实测峰值 38.2 GiB，中间这 3.5 GiB 就是
        # 低估的量。编码完成后按真实最长序列再查一次，超了就重算。
        real_t = (max(max(len(fc), len(fj)) for (fc, _), (fj, _) in dpo_enc)
                  if dpo_enc is not None else max(len(full) for full, _ in enc))
        if real_t > args.seqlen:
            print(f"实际最长序列 {real_t} tok > 预检占位 --seqlen {args.seqlen}"
                  f"（预检按占位算，这里按真实长度重查）", flush=True)
            mem_guard.require_budget(real_t, rows=(2 if dpo_enc is not None else 1),
                                     mode="bwd", ckpt=args.ckpt,
                                     moe_staging=(0.0 if args.ckpt else mem_guard.MOE_STAGING_GIB),
                                     log=print)

    # ---- 护栏 1：token 选择（excess）。参照策略 = 训练起点（此刻 B=0 / UniLoRA 初始扰动）。----
    # 训练三判据（决定「哪些 token 配拿到梯度」，越靠前是越硬的前提）：
    #   1) 来源：不是模型自己生成的（题面 / 用户输入）→ 永不训练（--frozen-prefix 冻前缀）；
    #   2) 可判别：模型生成的内容若**无法判断对错**，也不该训练 —— SFT 靠 gold answer、
    #      DPO 靠 chosen/rejected 偏好对，都是「已知对错」才给 loss；这里 excess 选择器
    #      再窄一层：只训「模型当前确实答错」（ref CE > 阈值）的答案 token，已经会的 token
    #      给梯度等于拿背样本换无关能力；
    #   3) 相对奖励：GRPO 这类不需要逐 token 绝对标签，靠「整组相对变好」（group-relative
    #      advantage）—— 数学原理不同，代价是组内必须有方差（std=0 ⇒ advantage 全零无信号，
    #      所以训练日志必须打 adv0 / uniq）。
    sel_masks: list[list[bool]] | None = None
    ref_ce_lists: list[list[float]] | None = None   # 起点答案段逐 token CE（selection 与报告共用一次前向）
    if dpo_rows is not None:
        if args.select_tokens != "off":
            print("DPO：--select-tokens 不适用（loss 是序列级 margin，不是逐 token CE），按 off 处理")
    elif args.pretrain_data:
        if args.select_tokens != "off":
            print("pretrain 块数据：--select-tokens 不适用（整块 loss 语义），按 off 处理")
    elif args.select_tokens == "excess":
        t_sel = time.time()
        was_training = model.training
        model.eval()
        # 一次前向同时产出两样东西：选择掩码（护栏 1）与报告基线（训练题起点 CE）。
        # 这两处原先各算一遍 `answer_ref_ce`，40 条数据白跑 ~1 min —— 训练总共才 1 min。
        ref_ce_lists = [answer_ref_ce(body, head, full, plen) for full, plen in enc]
        sel_masks = [guard.select_mask(c, args.select_thresh, args.select_first)
                     for c in ref_ce_lists]
        if was_training:
            model.train()
        tot_ans = sum(len(m) for m in sel_masks)
        tot_sel = sum(sum(m) for m in sel_masks)
        per = [sum(m) for m in sel_masks]
        print(f"token selection (ref CE > {args.select_thresh}): {tot_sel}/{tot_ans} 答案 token 进 loss"
              f"  每条 {per}  {time.time() - t_sel:.0f}s", flush=True)
        if tot_sel == 0:
            raise SystemExit(f"--select-thresh {args.select_thresh} 下一个 token 都没选中："
                             "阈值过高，或数据与训练起点无差异。先降阈值看分布再跑，不要空转")

    # ---- 报告基线：训练题在「训练起点」的答案 CE（全序列口径，无图）。
    # 训练末会用同一口径再算一遍 → 训练报告里给出「训练题前后变化」。
    # 未训练部分的对照走能力门探针（id/cap/生成体检），两边都要有数才叫完整账目。
    ce_ref: list[float] | None = None            # SFT：每条样本答案段平均 CE
    dpo_ref_ce: list[tuple[float, float]] | None = None   # DPO：每对 (chosen, rejected) 均值
    if not args.pretrain_data:
        was_training = model.training
        model.eval()
        if dpo_enc is None:
            if ref_ce_lists is not None:
                ce_ref = [sum(c) / max(1, len(c)) for c in ref_ce_lists]
            else:
                ce_ref = [sum(c) / max(1, len(c)) for c in
                          (answer_ref_ce(body, head, full, plen) for full, plen in enc)]
        else:
            dpo_ref_ce = [(sum(answer_ref_ce(body, head, fc, pc)) / max(1, len(fc) - pc),
                           sum(answer_ref_ce(body, head, fj, pj)) / max(1, len(fj) - pj))
                          for (fc, pc), (fj, pj) in dpo_enc]
        if was_training:
            model.train()

    # ---- 护栏 2：能力门（训练前先记起点：cap 的基线、id 的基准都从这里出）。----
    probes: list[dict] = []
    base_eval: dict | None = None
    final_ev: dict | None = None     # 终评结果（训练报告里「未训练部分」用）
    held: list[tuple[str, str]] = []
    if args.best_by == "held" and not args.held_data:
        raise SystemExit("--best-by held 需要 --held-data：没有 task heldout 集就没有可选的"
                         "目标指标（拿训练题 CE 选 ckpt = 按过拟合程度选）")
    if args.held_data:
        if args.eval_every <= 0:
            # `--eval-every 0` = 训练期完全不评估（「只产检查点，判分全交给引擎」那条路）。
            # 此时 held CE 无处可算，但 --held-data 仍要接受：它是给下游 `--eval-only`
            # 用的路标（pipeline 从训练参数里读它决定判哪一批）。在这里拒绝等于逼调用方
            # 把同一个路径写两遍，其中一份迟早会忘。
            print(f"--eval-every 0：训练期不评估，--held-data {args.held_data} 只给评估阶段用",
                  flush=True)
        held = read_sft_rows(args.held_data, args.word)
        if not held:
            raise SystemExit(f"--held-data {args.held_data} 是空的")
        # held 与训练集重叠 = 自评，等于把「背样本」当成「学到规律」。直接拒绝，
        # 不做「指出后继续跑」——历史里一次静默自评就够毁掉整轮结论了。
        overlap = set(held) & set(data)
        if overlap:
            raise SystemExit(f"--held-data 与 --data 有 {len(overlap)} 条完全相同的 (prompt, answer)，"
                             f"例如 {overlap.pop()!r} —— heldout 必须训练没见过")
    if args.eval_every > 0:
        if not os.path.exists(args.probes):
            raise SystemExit(f"--eval-every {args.eval_every} 需要探针文件 {args.probes}，不存在")
        probes = [json.loads(line) for line in open(args.probes, encoding="utf-8") if line.strip()]
        need_id = args.best_by == "id"
        if (need_id and not any(p["kind"] == "identity" for p in probes)) or \
           not any(p["kind"] == "cap" and p.get("target") for p in probes):
            raise SystemExit(f"{args.probes} 里 "
                             + ("identity / " if need_id else "")
                             + "cap(带 target) 探针至少各要有一条")
        was_training = model.training
        model.eval()
        base_eval = eval_gate(body, head, tok, probes, args.word, renderer=renderer,
                              model=model, held=held, gen=gen_be)
        if was_training:
            model.train()
        held_txt = (f"  held_mean {base_eval['held_mean']:.3f} nats/token"
                    f"（{base_eval['n_held']} 条，best-by {args.best_by}）"
                    if held else "")
        print(f"capability gate: 起点 id_mean {base_eval['id_mean']:.3f} nats  "
              f"cap_mean {base_eval['cap_mean']:.3f} nats/token{held_txt}  "
              f"({base_eval['n_id']} id + {base_eval['n_cap']} cap 探针；tol {args.cap_tol}, "
              f"stop {args.cap_stop}, every {args.eval_every})", flush=True)

    if args.frozen_prefix:
        cases = list(enc) if fp_enc is not None else [seq for pair in dpo_enc for seq in pair]
        was_training = model.training
        model.eval()
        # tol 1.0 nats 的依据（2026-10-31 diag_frozen.py 全样本实测）：切分点已对齐
        # GDN chunk 边界（消除 chunked delta rule 的块错位差）。残余差异来自「前缀/
        # 尾段与全序列 kernel 形状不同 ⇒ bf16 重结合不同」+「GDN triton atomics 的
        # run-to-run 累加顺序抖动」：逐层 rel ~8e-3（≈2 ulp bf16）从 layer 0 平滑
        # 放大，无单层突刺；逐 token worst CE 差 0.10~0.55 nats（同一样本跨进程可差
        # 4 倍，如 #2: 0.130 vs 0.548），且最大差总落在高熵 token（相对变化 ~12%）。
        # 逻辑错误（切错位/position 错）的量级是 5~30 nats —— 1.0 留一个数量级余量。
        verify_frozen_prefix(body, head, cases, tol=1.0)
        if was_training:
            model.train()

    torch.cuda.reset_peak_memory_stats()  # 训练段口径：加载期的临时缓冲不计入 peak
    # 显存探针（STRATA_MEM_PROBE=/path.pkl）：记录分配栈历史，step 2 结束时落盘。
    # 分析用 memanalyze.py（无 torch 依赖）。探针本身有开销，只用于定位，别开着训练。
    mem_probe = os.environ.get("STRATA_MEM_PROBE", "")
    mem_probe_step = int(os.environ.get("STRATA_MEM_PROBE_STEP", "2"))
    mem_probe_stage = os.environ.get("STRATA_MEM_PROBE_STAGE", "end")   # end=步尾；fwd=backward 前（图活着）
    mem_probe_done = False
    if mem_probe:
        torch.cuda.memory._record_memory_history(context="all", stacks="python",
                                                 max_entries=120000)
        print(f"mem probe: recording (python stacks) → {mem_probe}"
              f"（step {mem_probe_step} stage={mem_probe_stage} 落盘后停止）", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    rule = BackwardRule(params, lr=args.lr)   # 与零阶共用 step(loss_fn)/backward+finish
    shuffle = torch.Generator().manual_seed(args.seed)   # 固定数据顺序，A/B 才可比
    order, cursor = [], 0
    recent: list[float] = []          # 最近几步的 loss，给 --early-stop 用（跳步不计入）
    stopped_at = None
    hard_stopped = None
    last_step = 0
    best: dict | None = None          # keep-best：{"step", "objective", "id_delta", "cap_drift", "sd"}

    def _objective(ev: dict) -> float:
        """keep-best 的目标增量（越大越好）——`--best-by` 决定取哪个数。

        id  = 身份目标词 logp 相对起点的增量（身份改写任务）。
        held = 起点 held CE − 当前 held CE（任何有 heldout 集的任务）。
        cap 只当门槛，永不进目标：它是「跨域金丝雀」，降它没有意义（那不是我们要学的）。
        """
        if args.best_by == "held":
            return base_eval["held_mean"] - ev["held_mean"]
        return ev["id_mean"] - base_eval["id_mean"]

    def _obj_txt(v: float) -> str:
        # 两个任务的方向都统一成「正数 = 好」：id 是 logp 增量，held 是 CE **降幅**。
        # 不要在这行写「heldΔ 为负才好」之类 —— 目标量与 CE 本身符号相反，混着写必看错。
        return (f"heldCE↓ {v:+.3f} nats/tok" if args.best_by == "held"
                else f"idΔ {v:+.3f} nats")
    train_secs, n_ans_total, n_steps_done = 0.0, 0, 0     # 训练报告口径（步耗时之和 / 答案 token 数）
    prep_secs = time.time() - t_run0 - load_secs          # 编码 + token 选择 + 自检 + 起点评估
    # on-policy 前缀（--ss-k）的起点说明：把 K、降频、预热一次打印清楚，
    # 否则「这一步为什么多了一次解码」在日志里看不出来。
    if args.ss_k:
        print(f"on-policy 前缀: K={args.ss_k} 每 {args.ss_every} 步一次"
              f"{f'，前 {args.ss_warmup} 步跳过' if args.ss_warmup else ''}"
              f"（答案开头换成**当前策略**生成的 K 个 token，loss 仍对真答案求 CE）", flush=True)
    ss_any = 0        # 累计：真的发生了替换（生成前缀与真答案有分歧）的样本数
    ss_same = 0       # 累计：生成的 K 个 token 与真答案逐 token 相同（无新信号）
    for step in range(1, args.steps + 1):
        last_step = step
        t0 = time.time()
        # lr 调度在**每一步开头**定：AdamW 的 step() 用的就是此刻 param_group 里的值。
        # const + 无 warmup 时算出来就是 args.lr（`set_lr` 只写字段，不碰矩估计）。
        cur_lr = lr_at(step, args.steps, args.lr, args.lr_final, args.lr_schedule, args.warmup)
        rule.set_lr(cur_lr)
        lr_txt = f"  lr {cur_lr:.2e}" if (args.lr_schedule != "const" or args.warmup) else ""
        rule.zero_grad()
        tot, n_ans, n_sel, gn = 0.0, 0, 0, 0.0
        n_skip, n_contrib = 0, 0
        ss_step_seen = ss_step_div = 0        # 本步：参与的样本数 / 真的分歧的样本数
        ss_step_pos: list[int] = []           # 本步：第一个分歧位置（诊断用）
        margins: list[float] = []       # DPO：本步每个偏好对的 margin（报告 + acc 用）
        pool = dpo_enc if dpo_enc is not None else enc
        for _ in range(args.accum):
            if cursor >= len(order):
                order = torch.randperm(len(pool), generator=shuffle).tolist()
                cursor = 0
            idx = order[cursor]
            cursor += 1
            if dpo_enc is not None:
                # DPO 步：一对 = chosen + rejected 两次前向（各走一遍专家流），
                # ref 已预计算 ⇒ 每对恰 2 次前向 + 1 次反传，无采样。
                (fc, pc), (fj, pj) = dpo_enc[idx]
                rc, rj = dpo_ref[idx]
                if dpo_fp is not None:
                    # frozen-prefix：prompt 缓存算一次，chosen/rejected 各 deepcopy 一份
                    # —— 前向原地追加 K/V，不克隆就串序列（tail_logprobs 内部克隆）。
                    # margin 只吃答案段：剪掉 chunk 对齐多出来的 prompt 位（两端同前缀
                    # ⇒ 同一 prefix，off 相同）。
                    prefix, (tc, tj) = dpo_fp[idx]
                    cache = prefix_cache(body, prefix)
                    off = (pc - 1) - len(prefix)
                    lc = tail_logprobs(body, head, tc, cache)[off:].sum()
                    lj = tail_logprobs(body, head, tj, cache)[off:].sum()
                else:
                    lc = policy_logprob_sum(body, head, fc, pc)
                    lj = policy_logprob_sum(body, head, fj, pj)
                # margin 按答案 token 归一（与引擎侧 mezo 的 DpoMezo 同一语义，objectives
                # 唯一出处）：量级与序列长度无关，lr 不用随长度重标定。
                nc_tok, nr_tok = int(len(fc) - pc), int(len(fj) - pj)
                margin = objectives.dpo_margin(lc, lj, rc, rj, nc_tok, nr_tok)
                # β 必须跟着归一化换算，否则 KL 约束被除以 n̄（长度越长越松，
                # 见 objectives.dpo_beta_effective）：这是 2026-11-06 冷启 DPO 崩坏
                # 的根因，不是 lr 或能力门的问题。
                loss = objectives.dpo_loss(
                    margin, objectives.dpo_beta_effective(args.dpo_beta, nc_tok, nr_tok)
                ) / args.accum
                tot += rule.backward(loss) * args.accum
                margins.append(float(margin))
                n_ans += (len(fc) - pc) + (len(fj) - pj)
                n_contrib += 1
                continue
            full, plen = enc[idx]
            # ---- on-policy 前缀（--ss-k）：输入换成「自己生成的答案头」，标签仍是真答案 ----
            # 这一处就是治曝光偏差的全部：teacher forcing 下答案的每个 token 都由数据喂入，
            # 模型从没见过「自己已经走歪」的那种上下文；一旦评测时第 3 个 token 偏到基座
            # 先验上，它在这条新分支上没有学过任何东西（ce 0.2 而生成崩，就是这件事）。
            # 替换后，位置 0..K-1 的 loss 变成「在自己的错前缀之后说对下一个词」。
            ss_head: list[int] | None = None
            if (args.ss_k and (step > args.ss_warmup)
                    and (step - args.ss_warmup - 1) % args.ss_every == 0 and plen > 0):
                true_ans_now = full[plen:]
                kk = min(args.ss_k, max(0, len(true_ans_now) - 1))   # 至少留一个 token 作标签
                if kk > 0:
                    gen = ss_gen_ids(body, head, full[:plen], kk, tok.eos_token_id, model=model)
                    if len(gen) < kk:
                        # 提前 EOS：**保留**这个错误行为（不拿真答案把它填掉），只补足长度
                        gen = gen + true_ans_now[len(gen):kk]
                    d = ss_first_div(gen, true_ans_now)
                    ss_step_seen += 1
                    if d < 0:
                        ss_same += 1
                    else:
                        ss_head = gen
                        ss_any += 1
                        ss_step_div += 1
                        ss_step_pos.append(d)
            if fp_enc is not None:
                # frozen-prefix：前缀缓存（本步权重下的精确前向，no_grad）+ 尾段带图。
                # lp 剪掉 chunk 对齐多出的 prompt 位后恰是全部答案 token。
                prefix, tail = fp_enc[idx]
                cache = prefix_cache(body, prefix)
                off = (plen - 1) - len(prefix)
                if ss_head is None:
                    lp = tail_logprobs(body, head, tail, cache)[off:]
                else:
                    # 答案在 tail 内的起点是 off+1（lp[off] 预测 tail[off+1]）—— 替换从这个
                    # 下标开始，不是 off。标签取自**原始** tail（真答案），长度恰好等于
                    # 前向位置数：off 个 prompt 位 + 答案 token 数。
                    lab = tail[1:off + 1] + list(full[plen:])
                    tail = tail[:off + 1] + ss_head + tail[off + 1 + len(ss_head):]
                    lp = tail_logprobs_lab(body, head, tail, lab, cache)[off:]
                mask = torch.ones_like(lp, dtype=torch.bool)
                if sel_masks is not None:
                    if len(sel_masks[idx]) != lp.numel():
                        raise SystemExit(f"token 选择长度 {len(sel_masks[idx])} != "
                                         f"尾段 token 数 {lp.numel()}（切分与选择不是同一份编码）")
                    mask = mask & torch.tensor(sel_masks[idx], device=device, dtype=torch.bool)
                    if not mask.any():
                        n_skip += 1     # 这条样本的答案基座全都会：没有要学的东西，整条跳过
                        continue
                loss = objectives.sft_loss(lp[mask]) / args.accum
            else:
                # 输入序列（ss 时答案头被替换）与标签序列（恒为真答案）分开构造：
                # ss 关掉时 seq == full、lab == full[1:]，与历史逐 token 相同。
                seq = full if ss_head is None else (full[:plen] + ss_head
                                                    + full[plen + len(ss_head):])
                lab_ids = full[1:plen] + list(full[plen:])
                ids = torch.tensor([seq], device=device)
                h = body(input_ids=ids[:, :-1], use_cache=False).last_hidden_state[0]
                labels = torch.tensor(lab_ids, device=device)
                mask = torch.zeros_like(labels, dtype=torch.bool)
                mask[plen - 1:] = True
                if sel_masks is not None:
                    mask = mask & torch.tensor(guard.expand_to_labels(sel_masks[idx], labels.shape[0], plen),
                                               device=device, dtype=torch.bool)
                    if not mask.any():
                        n_skip += 1     # 这条样本的答案基座全都会：没有要学的东西，整条跳过
                        continue
                lp = token_logprobs(h, head, labels, chunk=128)
                loss = objectives.sft_loss(lp[mask]) / args.accum
            if mem_probe and mem_probe_stage == "fwd" and step == mem_probe_step and not mem_probe_done:
                # 快照打在 backward 之前：无 ckpt 时整张激活图都活着 —— 找「T 无关的
                # 激活大头」就要在这一刻看活块构成（步尾快照里图已经释放，看不到）。
                torch.cuda.memory._dump_snapshot(mem_probe)
                torch.cuda.memory._record_memory_history(None)
                mem_probe_done = True
                print(f"mem probe: snapshot dumped at step {step} (fwd, graph alive) → {mem_probe}",
                      flush=True)
            tot += rule.backward(loss) * args.accum
            # 答案 token（选择前口径，tok/s 与历史可比）：frozen 路径 lp 已是答案段
            n_ans += lp.numel() if fp_enc is not None else labels.numel() - (plen - 1)
            n_sel += int(mask.sum())
            n_contrib += 1
        if n_contrib == 0:
            print(f"step {step}: {args.accum} 条样本全被 token 选择跳过 —— 不更新参数、"
                  f"不计入 early-stop（loss=0 会假触发）", flush=True)
            continue
        _fin = rule.finish()
        gn = _fin["grad_norm"]
        dt = time.time() - t0
        train_secs += dt
        n_ans_total += n_ans
        n_steps_done += 1
        peak = torch.cuda.max_memory_allocated() / 2**30
        if dpo_enc is not None:
            m_loss, m_acc = dpo.dpo_batch(margins, args.dpo_beta)
            print(f"step {step}: dpo loss {m_loss:.4f}  acc {m_acc:.0%}  "
                  f"margin均值 {sum(margins) / len(margins):+.3f}  pairs {len(margins)}  "
                  f"tok/s {n_ans / dt:.1f}  grad_norm {gn:.4f}{lr_txt}  {dt:.1f}s  "
                  f"peak {peak:.1f} GiB",
                  flush=True)
            recent.append(m_loss)
        else:
            sel_txt = f"  sel {n_sel}/{n_ans}" if sel_masks is not None else ""
            skip_txt = f"  skip {n_skip}" if n_skip else ""
            # ss_txt：本步有几条样本出现了真实分歧、分歧平均发生在答案第几个 token。
            # 全是 same（分歧=0）说明策略已经能复现答案头，这一步的替换是空转 ——
            # 这个数字掉到 0 就是「该收工了」的直接证据，比 loss 曲线灵敏。
            ss_txt = ""
            if args.ss_k and ss_step_seen:
                pos = f"@{sum(ss_step_pos) / len(ss_step_pos):.1f}" if ss_step_pos else ""
                ss_txt = f"  ss {ss_step_div}/{ss_step_seen}{pos}"
            print(f"step {step}: loss {tot / args.accum:.4f}  loss_tokens {n_ans}{sel_txt}{skip_txt}{ss_txt}  "
                  f"tok/s {n_ans / dt:.1f}  grad_norm {gn:.4f}{lr_txt}  "
                  f"{dt:.1f}s  peak {peak:.1f} GiB",
                  flush=True)
            recent.append(tot / args.accum)
        if mem_probe and mem_probe_stage == "end" and step == mem_probe_step:
            torch.cuda.memory._dump_snapshot(mem_probe)
            torch.cuda.memory._record_memory_history(None)   # 停止记录，后续步零开销
            print(f"mem probe: snapshot dumped at step {step} → {mem_probe}", flush=True)
        # 中途存点（--save-every）：落盘的是**这一步**的参数，不是 keep-best 快照。
        # 存完训练继续往下走，所以这些文件可以之后用引擎逐个判分（引擎只换表、不重启），
        # 得到「步数 → pass@1」的曲线。这条路比在训练里插 torch 评估点便宜一个数量级：
        # 一次 CE 评估点要 30 次前向 + 可选 40 s 慢解码，而引擎判一份 adapter 是 8 s/条。
        if (args.save_every > 0 and step % args.save_every == 0 and step < args.steps
                and dpo_enc is None):
            ck = f"{args.out}.step{step}.pt"
            try:
                torch.save(_lora_state(model), ck)
                print(f"  saved {ck}", flush=True)
            except OSError as e:
                print(f"  WARN: 中途存点失败（训练继续）：{e}", flush=True)
        if (args.early_stop > 0 and step >= args.min_steps and len(recent) >= 3
                and sum(recent[-3:]) / 3 < args.early_stop):
            stopped_at = step
            print(f"early stop at step {step}: mean loss of last 3 steps "
                  f"{sum(recent[-3:]) / 3:.4f} < --early-stop {args.early_stop}  "
                  f"(样本已收敛；再训就是在背样本，无关能力会跟着坏)", flush=True)
            break
        # ---- 护栏 2：能力门（周期性、无图；id 好且能力没越界才配做 best）----
        if base_eval is not None and step % args.eval_every == 0:
            # 生成体检降频（--gen-check-every）：CE 是一次前向（几秒），体检是 decode
            # （torch 2 条 × 24 token ≈ 40 s）—— 评估点的墙钟几乎全在体检上。
            # 没到体检周期就只算 CE，但**这一点的 verdict 不许 accept**：n_gen=0 时
            # `all([])` 是空真，照原样放行等于把「没检查」当成「检查通过」。
            did_gen = not (args.gen_check_every > 0
                           and step % args.gen_check_every != 0)
            was_training = model.training
            model.eval()
            if gen_be is not None and did_gen:
                gen_be.swap(snapshot_trainable(model))   # 生成要评的是**这一步**的参数
            ev = eval_gate(body, head, tok, probes, args.word, renderer=renderer,
                           model=model, held=held, gen=gen_be,
                           n_gen=2 if did_gen else 0)
            # 身份文本的逐评估点轨迹（--gate-id-gen）：能力门自带的 gen 只生成 cap 探针，
            # 而「身份学没学会 / 是不是已经自环」只在身份问法上看得出来。逐点看才能把
            # 「还没学会」和「学会了但已经重复到自环」分开 —— 这两者的处方完全相反
            # （前者加步数，后者立刻停）。
            if args.gate_id_gen:
                # 必须显式 no_grad：torch 后端的 `_torch_gen` 自己不带装饰（它的调用方都在
                # no_grad 里），裸调会在 40 步解码上建出整张图 —— 实测 rank64 下一次就 58.8 GiB OOM。
                with torch.no_grad():
                    gg = gen_be or _torch_gen(body, head, tok, renderer, model=model)
                    for q in IDENTITY_QS[:args.gate_id_gen]:
                        r = gg(q, 40)
                        print(f"    id-gen[{step}] {q} → {r['text'][:110]!r} eos={int(r['eos'])}",
                              flush=True)
            if was_training:
                model.train()
            id_delta = ev["id_mean"] - base_eval["id_mean"]
            cap_drift = ev["cap_mean"] - base_eval["cap_mean"]
            objective = _objective(ev)
            v = guard.gate_verdict(objective, cap_drift,
                                   best["objective"] if best else None, args.cap_tol,
                                   gen_ok=ev["gen_ok"])
            if not did_gen and v == "accept":
                # 没做生成体检的点不配做 best：`gen_ok` 在 n_gen=0 时是空真 True，
                # 直接 accept 就把「没检查」当成了「检查通过」。降到 hold，等下一个
                # 体检点（目标更高时会重新 accept），终评永远做满 2 条兜底。
                v = "hold"
            gen_txt = ("(本点未做生成体检：--gen-check-every "
                       f"{args.gen_check_every})" if not did_gen else
                       "  ".join(f"{r['text'][:22]!r}/d3={r['distinct3']}/eos={int(r['eos'])}"
                                 for r in ev["gen"]))
            held_txt = (f"  heldCE {ev['held_mean']:.3f}" if held else "")
            print(f"gate step {step}: {_obj_txt(objective)}  capΔ {cap_drift:+.3f} nats/tok"
                  f"{held_txt}  → {v}"
                  f"\n    gen: {gen_txt}", flush=True)
            if guard.should_hard_stop(cap_drift, args.cap_stop):
                hard_stopped = step
                print(f"capability hard stop at step {step}: capΔ {cap_drift:.3f} > "
                      f"--cap-stop {args.cap_stop}（能力在坏，立刻停；存盘前恢复 best 快照）", flush=True)
                break
            if v == "accept":
                best = {"step": step, "objective": objective, "id_delta": id_delta,
                        "cap_drift": cap_drift, "sd": snapshot_trainable(model)}

    # ---- 能力门收尾：终评 + 恢复 keep-best，再走存盘 ----
    if base_eval is not None:
        was_training = model.training
        model.eval()
        if gen_be is not None:
            gen_be.swap(snapshot_trainable(model))       # 终评量的是**末参数**（best 可能更早）
        ev = eval_gate(body, head, tok, probes, args.word, renderer=renderer,
                       model=model, held=held, gen=gen_be)
        final_ev = ev
        if was_training:
            model.train()
        id_delta = ev["id_mean"] - base_eval["id_mean"]
        cap_drift = ev["cap_mean"] - base_eval["cap_mean"]
        objective = _objective(ev)
        v = guard.gate_verdict(objective, cap_drift,
                               best["objective"] if best else None, args.cap_tol,
                               gen_ok=ev["gen_ok"])
        gen_txt = "  ".join(f"{r['text'][:22]!r}/d3={r['distinct3']}/eos={int(r['eos'])}"
                            for r in ev["gen"])
        held_txt = (f"  heldCE {ev['held_mean']:.3f}" if held else "")
        print(f"gate final (step {last_step}"
              + (f", early stop" if stopped_at else "")
              + (f", hard stop" if hard_stopped else "")
              + f"): {_obj_txt(objective)}  capΔ {cap_drift:+.3f} nats/tok{held_txt}  → {v}"
              f"\n    gen: {gen_txt}", flush=True)
        if v == "accept":
            best = {"step": last_step, "objective": objective, "id_delta": id_delta,
                    "cap_drift": cap_drift, "sd": snapshot_trainable(model)}
        if best is not None and best["step"] != last_step:
            missing = restore_trainable(model, best["sd"])
            print(f"restored keep-best snapshot: step {best['step']} "
                  f"({_obj_txt(best['objective'])}, capΔ {best['cap_drift']:+.3f}, "
                  f"best-by {args.best_by})"
                  + (f"  未命中 {missing}" if missing else ""), flush=True)
        elif best is None:
            print("capability gate 从未 accept：没有 best 可恢复，产物按当前参数保存 —— "
                  "身份没学会或能力已坏，请人工核对验收", flush=True)

    lp1 = name_logprob(body, head, tok, args.probe, args.word, renderer=renderer)
    print(f"after:  logp('{args.word}' | {args.probe!r}) = {lp1:.4f} (p={math.exp(lp1):.8f})   "
          f"Δ={lp1 - lp0:+.4f} nats", flush=True)

    # ---- 训练报告：速度 / 用时 / 训练前后效果（训练题 + 未训练探针）----
    # 每次训练都必须留下这三样，否则「练了有没有用、快不快、有没有伤到别的能力」无从判断。
    run_secs = time.time() - t_run0
    post_secs = max(0.0, run_secs - load_secs - prep_secs - train_secs)
    print("\n===== 训练报告 =====", flush=True)
    print(f"训练: {n_steps_done} 步 / 影响答案 token {n_ans_total} / 用时 {train_secs:.1f} s / "
          f"tok/s {n_ans_total / train_secs if train_secs else 0:.1f} / "
          f"峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    # 每问被训练多少次：这是「同一批样本看几遍」的直接口径，也是「训练两次能力就掉」
    # 那个现象的量化轴。设计值按 --steps 推，实际值按真正跑完的步数算 —— early stop /
    # hard stop 之后两者会不同，只报设计值会把「跑了 5 步就停」说成「每问 1 遍」。
    # 一个 step 看 --accum 条样本，所以每问次数 = 步数 × accum ÷ 不同问法数。
    # DPO 的一行本身就是一条偏好对（同一问题的 chosen/rejected），按对数计。
    n_uniq = len(dpo_enc) if dpo_enc is not None else len({p for p, _ in data})
    per_q_design = (args.steps * args.accum / n_uniq) if n_uniq else 0.0
    per_q_real = (n_steps_done * args.accum / n_uniq) if n_uniq else 0.0
    n_rows = len(dpo_enc) if dpo_enc is not None else len(data)
    print(f"数据: {n_rows} 行 / {n_uniq} 个不同问法  "
          f"每问训练 {per_q_real:.2f} 次"
          + (f"（设计 {per_q_design:.2f} 次，实际按 {n_steps_done} 步算）"
             if abs(per_q_real - per_q_design) > 1e-9 else ""), flush=True)
    if args.ss_k:
        # 收尾口径：真实分歧率。如果训练全程都是 same，等于没开 ss（成本白付），
        # 必须显式说明 —— 否则「改了参数没效果」会被误读成「这条路不通」。
        tot_ss = ss_any + ss_same
        print(f"on-policy 前缀: {args.ss_k} tok/样本  分歧 {ss_any}/{tot_ss} 次"
              f"（{100 * ss_any / tot_ss if tot_ss else 0:.0f}%）  "
              f"逐字相同 {ss_same} 次", flush=True)
    if final_ev is not None and final_ev["gen_tokens"]:
        print(f"生成: {final_ev['gen_tokens']} tokens / {final_ev['gen_secs']:.1f} s / "
              f"tok/s {final_ev['gen_tok_s']:.2f}（终评贪心解码，含 prefill）", flush=True)
    else:
        print("生成: 未测（--eval-every 0 时没有探针评估）", flush=True)
    print(f"总用时 {run_secs:.0f} s = 加载 {load_secs:.0f} + 数据准备/自检 {prep_secs:.0f} + "
          f"训练 {train_secs:.0f} + 终评/存盘 {post_secs:.0f}", flush=True)

    def _ce_now(full, plen):
        return answer_ref_ce(body, head, full, plen)

    was_training = model.training
    model.eval()
    if ce_ref is not None:
        ce_now = [sum(c) / max(1, len(c)) for c in (_ce_now(f, p) for f, p in enc)]
        print("训练题效果（答案 CE nats/token，越小越好；全序列口径 = 与评估同尺）:", flush=True)
        for i, (a, b) in enumerate(zip(ce_ref, ce_now)):
            print(f"  #{i}  len {lens[i]:3d}  {a:6.3f} → {b:6.3f}  ({b - a:+.3f})", flush=True)
        m0, m1 = sum(ce_ref) / len(ce_ref), sum(ce_now) / len(ce_now)
        print(f"  均值         {m0:6.3f} → {m1:6.3f}  ({m1 - m0:+.3f})", flush=True)
    elif dpo_ref_ce is not None:
        now = [ (sum(_ce_now(fc, pc)) / max(1, len(fc) - pc),
                 sum(_ce_now(fj, pj)) / max(1, len(fj) - pj))
                for (fc, pc), (fj, pj) in dpo_enc]
        print("偏好对效果（CE nats/token；chosen 该降、rejected 该升）:", flush=True)
        for i, ((c0, j0), (c1, j1)) in enumerate(zip(dpo_ref_ce, now)):
            print(f"  pair#{i}  chosen {c0:6.3f} → {c1:6.3f} ({c1 - c0:+.3f})   "
                  f"rejected {j0:6.3f} → {j1:6.3f} ({j1 - j0:+.3f})", flush=True)
    if was_training:
        model.train()

    if base_eval is not None and final_ev is not None:
        print(f"未训练部分（能力门探针：{base_eval['n_id']} id + {base_eval['n_cap']} cap，"
              f"跨域金丝雀，不该坏）:", flush=True)
        if base_eval["n_id"]:
            print(f"  id_mean  {base_eval['id_mean']:+.3f} → {final_ev['id_mean']:+.3f} nats  "
                  f"(Δ{final_ev['id_mean'] - base_eval['id_mean']:+.3f}，目标词概率应上升)", flush=True)
        else:
            print("  id_mean  未测（探针文件里没有 identity 行：非身份任务，目标指标走 --held-data）",
                  flush=True)
        print(f"  cap_mean {base_eval['cap_mean']:.3f} → {final_ev['cap_mean']:.3f} nats/tok  "
              f"(Δ{final_ev['cap_mean'] - base_eval['cap_mean']:+.3f}，越界阈值 {args.cap_tol})", flush=True)
        if held:
            # 这一行是「学到规律 vs 背样本」的分界：训练题 CE 一定降，held CE 不降就是过拟合。
            print(f"  held_mean {base_eval['held_mean']:.3f} → {final_ev['held_mean']:.3f} nats/tok  "
                  f"(Δ{final_ev['held_mean'] - base_eval['held_mean']:+.3f}，"
                  f"{base_eval['n_held']} 条训练没见过；keep-best 按它选 = {args.best_by == 'held'}）",
                  flush=True)
        print(f"  生成体检 {sum(1 for r in final_ev['gen'] if r['ok'])}/{len(final_ev['gen'])} ok"
              f"（distinct-3gram + EOS + 无答案串泄漏）", flush=True)

    # keep-best 已恢复（若与末参数不同源）：引擎侧必须跟着换成**要存盘的**那一份，
    # 否则报告里「生成抽检」「身份问答」量的是被丢掉的末参数。
    if gen_be is not None:
        gen_be.swap(snapshot_trainable(model), keep_zero_b=True)

    # ---- held 生成抽检：CE 只说「像」，这里直接读准确率（唯一能算 pass@1 的项）----
    if held and args.held_gen > 0:
        n_probe = min(args.held_gen, len(held))
        stride = max(1, len(held) // n_probe)
        picks = held[::stride][:n_probe]
        was_training = model.training
        model.eval()
        with torch.no_grad():
            gg = gen_be or _torch_gen(body, head, tok, renderer, model=model)
            hits = 0
            # `--held-max-new` 对数学题是必须的：48 token 会截在推理中途，于是每条都判错、
            # 而且错得与能力无关（模型不是答错，是还没答完）。
            max_new = args.held_max_new or 48
            print(f"held 生成抽检（{n_probe}/{len(held)} 条，均匀跨步；"
                  f"判分 = math_extract.answer_hit，上限 {max_new} token）:", flush=True)
            for p, t in picks:
                r = gg(p, max_new)
                hit = _answer_hit(r["text"], t, p, args.answer_strict)
                hits += int(hit)
                print(f"  {'OK ' if hit else 'BAD'} {p[:52]!r} → {r['text'][:52]!r}  "
                      f"gold {t!r}  eos={int(r['eos'])}", flush=True)
            print(f"  held pass@1 抽检 {hits}/{n_probe}  "
                  f"（这是「学到了没有」的直接读数，CE 降低不能代替它）", flush=True)
        if was_training:
            model.train()

    # ---- 身份问答的实际文本（训练后）：事实对不对只能靠眼看 ----
    was_training = model.training
    model.eval()
    id_gen_after = sample_identity(body, head, tok, renderer=renderer, model=model, gen=gen_be)
    if was_training:
        model.train()
    print("身份问答（训练后，贪心解码；与训练前逐条对照）:", flush=True)
    for q, b, a in zip(IDENTITY_QS, id_gen_before, id_gen_after):
        print(f"  {q}\n    before: {b[:150]}\n    after:  {a[:150]}", flush=True)

    if args.lora_method == "unilora":
        import unilora

        # 物化成稠密 A/B：键名与 lora.py 约定一致，lora_to_gguf.py 直接可读。
        # 引擎侧缩放 s = alpha/rank，导出时传 --alpha <rank> 才能逐值还原训练时的 delta。
        sd = unilora.materialize_ab(model)
        ck = unilora.unilora_ckpt(model, proj_seed=args.proj_seed, theta_d_length=args.theta_d,
                                  init_bound=args.init_bound)
        torch.save(ck, args.out + ".unilora.pt")
    else:
        sd = _lora_state(model)
    torch.save(sd, args.out)
    # 完成态：把实际步数、keep-best、能力门终评、耗时与峰值并进 start 那份，不另开文件。
    # 注意不要往 extra 里放 `out`：它在 vars(args) 里已有；而且 write_run_meta 的
    # 首参就叫 out_path，重复的 out 只会覆盖同一个值、没有信息量。
    done_extra = {"phase": "done", "steps_done": n_steps_done,
                  "train_secs": round(train_secs, 3), "run_secs": round(run_secs, 3),
                  # 每问训练次数：报告里也印，但 meta 里必须有 —— 「训两次就退化」这类
                  # 结论要靠它和其他运行对齐，事后从日志里 grep 出来的数不能跨运行比较。
                  "rows": n_rows, "unique_prompts": n_uniq,
                  "per_question_design": round(per_q_design, 4),
                  "per_question_done": round(per_q_real, 4),
                  "peak_gib": round(float(torch.cuda.max_memory_allocated()) / 2**30, 3)}
    if best is not None:
        done_extra["best_step"] = best["step"]
        done_extra["best_objective"] = round(float(best["objective"]), 6)
        done_extra["best_by"] = args.best_by
        done_extra["best_cap_drift"] = round(float(best["cap_drift"]), 6)
        done_extra["best_id_delta"] = round(float(best["id_delta"]), 6)
    if base_eval is not None and final_ev is not None:
        done_extra["cap_drift_final"] = round(float(final_ev["cap_mean"] - base_eval["cap_mean"]), 6)
        done_extra["id_delta_final"] = round(float(final_ev["id_mean"] - base_eval["id_mean"]), 6)
        if held:
            done_extra["held_ce_base"] = round(float(base_eval["held_mean"]), 6)
            done_extra["held_ce_final"] = round(float(final_ev["held_mean"]), 6)
            done_extra["held_n"] = int(base_eval["n_held"])
        done_extra["gen_ok"] = sum(1 for r in final_ev["gen"] if r["ok"])
        done_extra["gen_n"] = len(final_ev["gen"])
    write_run_meta(args.out, args, **done_extra)
    nz = sum(1 for v in sd.values() if float(v.abs().max()) > 0)
    print(f"saved {args.out}: {len(sd)} tensors, non-zero {nz}, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as _exc:                  # noqa: BLE001
        if mem_guard.is_oom(_exc):
            raise mem_guard.die_on_oom(_exc, where=__file__.rsplit("/", 1)[-1])
        raise
