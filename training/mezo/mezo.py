"""mezo.py —— 引擎侧零阶（MeZO）训练循环：引擎自己当 forward，三种算法共用。

分工（2026-11-14 重划 —— 算法与训练器分家）：

  zo/optim.py     零阶估计器：采方向、±ε 对称差分、动量、步长预算。只认四样东西 ——
                  θ、`loss_fn(th) -> float`、lr、rng。不认识引擎、SCORE、batch、数据、评测。
  common/         与「怎么算 Δθ」无关的一切，两条训练线共用同一份：
                  data.py 装载 / chat.py 模板与 tokenizer / encode.py 编码与探针口径 /
                  lora.py adapter GGUF 读写 / runtime.py 引擎会话 / schedule.py lr 曲线。
  本文件          引擎侧的**编排**：四个目标类（各写一条 `loss_at`）、引擎换表与评测、
                  循环、CLI。算法实现一行都不在这里（`_zo_step` 只剩 3 行转发）。

  为什么要这么分（用户 2026-11-14 指出）：MeZO 只是「替换反传的那个方法」，之前却有
  2000 行里有 1900 行是数据/模板/评测/调度 —— 结果任何一个评测口径问题都会被误判成
  算法问题，反之亦然。现在算法有 `tests/test_zo_optim.py` 八条离线判据（含与旧实现
  3 步逐位比对的回归锁），读数口径问题一律落到 common/ 或本文件的编排里。

方向采样（--zo-subspace，默认 0 = MeZO 原文均匀方向）：

    默认是 MeZO 原文的各向同性 z ~ N(0, I)（2026-11-16 实测它是唯一到过 4/5 的
    路径；LoZO 在本设定 d=69k 下反而更慢）。可选 InferZO 的低秩方向
    （随机正交基 + 每 ν=50 步懒刷新，`--zo-subspace 8 --zo-subspace-mode lozo`）。
    引用：

    [LoZO]   Chen, Zhang, Cao, Yuan, Wen. "Enhancing Zeroth-Order Fine-Tuning
             for Language Models with Low-Rank Structures." ICLR 2025.
             https://arxiv.org/abs/2410.07698
    [InferZO] Li, Ding. "Inference-Native Zeroth-Order Optimization for LLMs."
             arXiv:2605.28760, 2026. https://arxiv.org/abs/2605.28760
             （官方实现 github.com/playeriv65/InferZO；其 LOZODirectionProvider
             是本实现 `training/zo/optim.py` lozo 模式的参照。）
    [MeZO]   Malladi et al. "Fine-Tuning Language Models by Zeroth-Order
             Optimization." NeurIPS 2023. https://arxiv.org/abs/2305.17333
             （--zo-subspace 0 退回此行为）

算法（--objective，与 train.py --algo 一一对应）：
    ce    SFT：answer 段逐 token logp → objectives.sft_loss（mean 口径）
    dpo   偏好对：chosen/rejected 各一次 SCORE → 长度归一 margin（objectives.dpo_margin，
          每侧按答案 token 平均）→ objectives.dpo_loss（ref 在 θ₀ 预计算一次，训练中不重算）
    grpo  回放轨迹：逐条 SCORE 得 per-token logp → objectives.grpo_loss
          （old logp 在 θ₀ 缓存，ratio = exp(logp(θ±ε) − old)；组内优势共用
          objectives.group_advantages）
    luspo 同一条回放链路，损失换成 objectives.luspo_loss：GSPO 的序列级 ratio，
          分母用序列数 G 而不是 token 数 ⇒ 每条轨迹的梯度不随长度衰减
          （参考 github.com/murphy4122/LUSPO，与 grpo 只差归一化这一处）

**损失数学只有一份**：四个目标全部调 `objectives.py`（numpy 路径），反传线
（sft.py/grpo.py）调同一份函数的 torch 路径 —— 「同算法换方法」共享同一个数，
test_objectives.py 的 numpy↔torch 对拍守着这件事。

每步（ZO-SGD，Malladi et al. 2023 的对称差分 + 种子再生；--momentum/--dirs/--batch-rows 是方差三杠杆）：
    z   = rng(step)                          方向由 seed 再生，不存储
    l+  = L(theta + eps*z)  —— LORA plus.gguf  → SCORE（batch-rows 条样本的均值）
    l-  = L(theta - eps*z)  —— LORA minus.gguf → SCORE
    pj  = (l+ - l-) / (2*eps)                方向导数估计（L = -answer 段 **per-token 平均** logp）
    g   = mean_dirs(pj * z)                  --dirs 个方向取平均（方差 ~1/q）
    v   = momentum * v + g                   --momentum 动量（零均值噪声抵消）
    theta -= lr * v                          fp32 参数，bf16 吃不动 1e-6 量级更新

单方向 pj 的信噪比实测 ~0.15（_snr.py：std 3.0e+02 vs |mean| 4.6e+01）⇒ 单步更新
本质是「带漂移的随机游走」，这就是 MeZO 难收敛的结构性原因；三个杠杆都在压方向方差。

**loss 口径必须与 HF 线一致（2026-11-01 起默认 `--loss-reduction mean`）**。旧实现
按「answer 段 sum logp」算，pj ∝ 序列长度：同一份 73569 token 的数据 pj = -4.0e+04，
lr 1e-5 两步把 ‖θ‖ 从 736 推到 1419（步长等于参数范数的 40%，直接发散），而 HF 侧
mean 口径的 pj 只有 ~0.55 —— 同一个 lr 在两条路上的实际步长差 T 倍。现在两口子都
按 token 平均，`step` 行新增 `rel = ‖lr·v‖/‖θ‖`：**>0.05/步 就会发散**，看这条比看
lr 的绝对值可靠（identity 上 lr 1e-4 → rel 0.09，已经偏大；lr 3e-5 → 0.03 合适）。

与引擎的三个接缝：
  LORA <path>   把 adapter GGUF 的表换进引擎（lora_update：形状必须不变 ⇒ rank 全程固定）。
  SCORE <ids>   逐位置 log p(ids[t+1] | ids[0..t])；answer 段从 t = plen-1 起求和
                （位置 t 的 SL 是「下一个 token」的 logprob，answer 的第一个 token 由
                prompt 的最后一位预测）。末位没有 next，引擎回 NaN，不进任何和。
  极性          训练全程 +1。SCORE 的 `+`/`-` 前缀翻的是同一份表的符号，只有
                「固定 theta、只翻扰动」的读法才用得到；这里每步换表，用不上。

chat 渲染与 serve 同一条模板（pack 的 tokenizer/chat_template.jinja，jinja2 同参数），
训练分布 = 推理分布；评测探针（--probe）也走同一渲染。

用法：
    python mezo.py --pack <pack目录> --gguf <基座shard> \
        --engine ../../build-mezo/strata --data identity --word Strata --steps 40
    python mezo.py ... --data jsonl:math_prompts.jsonl
    python mezo.py ... --objective dpo --dpo-data examples/training/data_dpo.jsonl
    python mezo.py ... --objective grpo --rollouts rollouts.jsonl --group-size 4
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
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

import objectives  # noqa: E402  共享目标函数（numpy 路径；与反传线同一份数学）
import guard        # noqa: E402  token 选择（select_mask）与反传线共用同一份纯逻辑
import metrics as mtr  # noqa: E402  训练指标出口（JSONL + 本地 HTTP，RSI 控制器轮询用）
import report  # noqa: E402  回环内的评测与记录（判据展示只在这里改）
from zo.optim import (Mezo, theta_norms, rel_cap_would_selflock,  # noqa: E402
                      theta_guard_exceeded, theta_ema_init, theta_ema_update)
from common.chat import (load_template, load_tokenizer, load_transcript_template,
                         template_thinks)
from common.data import (IDENTITY_ROWS, SYSTEM_IDENTITY, THINK_EN, THINK_GEN_EN, THINK_GEN_ZH,
                         THINK_ZH,
                         builtin_data, dataset_thinks, load_conversations, load_data, load_dpo,
                         load_rollouts, with_thinking)
from common.encode import answer_seg, enc_answer, enc_transcript, engine_keep, eos_id, name_logprob
from common.lora import (adapter_layout, init_theta, read_adapter, read_adapter_alpha,
                         read_adapter_rank, read_geometry, write_adapter)
from common.schedule import lr_at
from common.runtime import (PREFILL_CHUNK_CAP, Engine, EngineApi, EngineHttp, engine_argv,
                            launch_engine, resolve_prefill_chunk)

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools"))
from gguf_writer import GGUFWriter, _Tensor  # noqa: E402




class EngineMezo:
    """引擎零阶（MeZO）训练的公共骨架。

    继承就是「算法 × 方法」里方法轴的落法：方法（ZO 估计循环、引擎会话、adapter
    换表、动量/rel 日志、eval、存档）在这里写一次；算法 = 子类，只实现三个钩子：

      load_items      数据装载与编码。统一 dict；answer 段的切片下界放 "start"。
      theta0_cache    θ₀（训练起点的 adapter 状态）处的一次性缓存。dpo 缓存 ref、
                      grpo 缓存 old logp —— 它们不随 θ±ε 变，训练中不重算。
      loss_at         损失。**必须调 objectives.py（numpy 路径）**，与反传线共用
                      同一份数学；这里不出现损失公式 —— 公式写两遍就等于没共享。

    加新算法 = 加一个子类 + 在 TRAINER_FOR 登记一行，main() 与 train.py 不动。
    """

    objective = ""        # 子类填：ce / dpo / grpo（与 algos.ALGOS[algo].objective 对应）
    # 回合结束 token：`__init__` 按 pack 的 eos 覆盖。类默认 None 只是给测试桩
    # （`object.__new__` 不经 `__init__`）一个确定的取值，不是「不收尾」的意思。
    eos: int | None = None

    def __init__(self, args, tok, render, eng, theta, layout):
        self.args, self.tok, self.render = args, tok, render
        self.eng, self.theta, self.layout = eng, theta, layout
        self.items: list[dict] = []
        self.use_rows_n = 1
        self.plus_path = self.minus_path = self.final_path = ""
        self.lp0 = float("nan")
        self.ce0: float | None = None
        self.swap = lambda th, path: None      # run() 里替换成真实现
        # 量 CE 用非思考模板：与「答案就是一句话」的训练/采样口径一致
        system = getattr(args, "system_prompt", "") or ""
        effort = getattr(args, "reasoning_effort", "") or None
        self.render_off = (load_template(args.pack, enable_thinking=False, system=system,
                                         reasoning_effort=effort)
                           if template_thinks(render) else render)
        # `--think-targets auto`：**按数据判**要不要让模型先写思考段，且必须在建 eval_render
        # 之前定下来 —— 评测与训练必须走同一条模板，这条判错两组读数就不是同一个位置。
        self.think_mode = self._resolve_think_mode(args)
        # 评测/探针走**哪条通道**必须与训练目标一致，否则量的是模型从没被监督过的位置。
        # 2026-11-14 实测：`--think-targets off`（只训纯答案）时，评测仍用思考模板
        # （prompt 以 `<think>` 收尾，模板默认 reasoning_effort='xhigh'），而身份答案在
        # `</think>` 之后 —— 64 token 的预算连思考段都写不完，`success` 恒 0/5，
        # 与学没学会无关；`logp(word|probe)` 也测在 `<think>` 紧随的位置，训练从未优化那里。
        # auto = 跟着判定结果（off → 非思考模板；on/raw → 思考模板）。
        _er = (getattr(args, "eval_render", "") or "auto").strip()
        if _er == "off":
            self.eval_render = self.render_off
        elif _er == "on":
            self.eval_render = render
        else:
            self.eval_render = self.render_off if self.think_mode == "off" else render
        # 回合结束 token：答案末尾必须带上它，否则模型从没学过「在哪收尾」（2026-11-15 定位）。
        # `--no-answer-eos` 只为把历史 run 复现出来做对照，不是正常用法。
        _tok_eos = eos_id(tok)
        if getattr(args, "answer_eos", True) and _tok_eos is not None:
            self.eos: int | None = _tok_eos
        else:
            self.eos = None
            if getattr(args, "answer_eos", True):
                # 不硬退：换 pack 的人不该因为打包漏了 eos 就跑不起来。但也不能静默 ——
                # 这一档就是 2026-11-15 那个「生成自环复读」的成因，必须让日志自己带出来。
                print("WARNING: 这个 pack 没有 tokenizer.ggml.eos_token_id，答案末尾接不上回合结束"
                      " token ⇒ 模型不会学「说到哪停」，生成可能自环复读。"
                      "换一份带 eos 的 pack，或显式 --no-answer-eos 表示你知道。", flush=True)
        self._ce_pairs: list[tuple[str, str]] | None = None
        # 指标出口：空路径 + 0 端口 = 什么都不做（旧行为逐字不变）。run() 里发事件，
        # main() 的 finally 负责关掉 HTTP 线程 —— 训练崩了也不能留下一个监听端口。
        self.sink = mtr.MetricSink(
            getattr(args, "metrics", "") or "",
            int(getattr(args, "metrics_port", 0) or 0),
            run=getattr(args, "metrics_run", "") or
            f"{args.objective}-{getattr(args, 'method', 'mezo')}")

    # ---- 三个算法钩子（子类覆写） -------------------------------------------
    def _resolve_think_mode(self, args) -> str:
        """把 `--think-targets` 定成一个具体档（off / on / raw / both）。

        `auto`（默认）看**详细答案本身**：里面有 `<think>`/`</think>` 就开思考，没有就关。
        理由写在 `common/data.dataset_thinks` 里 —— 目标里有什么，推理就允许写什么；
        目标里没有的，不该让模型在推理时凭空多写一段没人监督的文本。

        读不到监督目标（例如纯多轮转录自带思考段）时按 `off` 并说明原因。
        """
        mode = getattr(args, "think_targets", "auto") or "auto"
        if mode != "auto":
            return mode
        targets: list[str] = []
        why = ""
        try:
            obj = self.objective
            if obj == "ce" and getattr(args, "data", ""):
                targets = [a for _, a in load_data(args.data, args.word)]
            elif obj == "dpo":
                spec = getattr(args, "dpo_data", "") or getattr(args, "data", "")
                targets = [r["chosen"] for r in load_dpo(spec) if r.get("chosen")]
            elif getattr(args, "rollouts", ""):
                targets = [r.get("completion") or "" for r in load_rollouts(args.rollouts)]
        except (SystemExit, OSError, ValueError, KeyError) as exc:
            targets, why = [], f"读不到监督目标（{type(exc).__name__}）"
        on, why2 = dataset_thinks(targets) if targets else (False, why or "没有监督目标文本")
        mode = "on" if on else "off"
        print(f"think-targets auto → {mode}：{why2}", flush=True)
        return mode

    def load_items(self) -> list[dict]:
        raise NotImplementedError

    def theta0_cache(self) -> None:
        pass                                   # ce 没有常数侧

    def anchor_weight(self) -> float:
        return float(getattr(self.args, "anchor_weight", 0.0) or 0.0)

    def prepare_selection(self) -> None:
        """标记「起点答错」的答案 token（`--select-tokens excess`）。只有 ce 用得上。

        反向线 `backward/sft.py` 里这是一个 tensor 掩码；零阶线没有反传，掩码的作用是
        **把零阶差分限制在选中的 token 上** —— 估计量只对这几个 token 的 CE 求导。
        位置与 θ₀ 绑定：掩码是数据属性（起点会不会），训练中不重算，否则每个候选点
        的掩码不同，θ+ε 与 θ-ε 差分的就不再是同一个量。
        """
        sel = getattr(self.args, "select_tokens", "excess")
        if self.objective != "ce":
            if sel != "off":
                print(f"{self.objective}：--select-tokens 不适用（不是逐 token CE），按 off 处理",
                      flush=True)
            return
        if sel == "off":
            if float(getattr(self.args, "select_mass", 0.0) or 0.0) > 0:
                raise SystemExit("--select-mass 自己就决定选哪些 token，与 --select-tokens off 互斥")
            if float(getattr(self.args, "span_weight", 0.0) or 0.0) > 0:
                raise SystemExit("--select-tokens off 没有任何位置可选，--span-weight 无从加权；"
                                 "要「整段进 loss + 难位置加权」请用 --select-tokens all")
            print("token selection: off（全答案段平均）", flush=True)
            for it in self.items:
                # 全答案段进 loss ⇒ 引擎只需要算这些行（prompt 那些位置一个都不进 loss）
                it["keep"] = engine_keep(it, None)
            return
        all_on = sel == "all"
        if all_on:
            # 「全段留 + 难位置加权」：excess 的两条已知失败都在这里被同时避开 ——
            #   ① 只监督少数 token（--select-first / excess thresh 1.0 都只留 5~14%）⇒
            #      其余位置无人约束，生成走到那里 argmax 就是它自己，塌成自环；
            #   ② 全段等权平均 ⇒ 530 个 token 里 34 个贵位置（闸门 CE 2~4、其余 0.05~0.2）
            #      只占 1/530 的零阶差分权重，闸门压不动（sft 12000→20000 步实测：
            #      句号后首 token 的 logp 只从 −1.93 挪到 −1.90，而 EOS 从 −2.24 抬到 −1.75）。
            if float(getattr(self.args, "span_weight", 0.0) or 0.0) <= 1.0:
                raise SystemExit("--select-tokens all 需要 --span-weight W（W>1）说明难位置加权多少；"
                                 "等同于不加权时请直接用 --select-tokens off")
            for bad, why in (("select_first", "--select-first 是截断"),):
                if int(getattr(self.args, bad, 0) or 0) > 0:
                    raise SystemExit(f"--select-tokens all 与 {why}互斥：all 的要点是「每一位都留着」")
            if float(getattr(self.args, "gate_weight", 0) or 0) > 0:
                raise SystemExit("--select-tokens all 的加权由 --span-weight 负责（按 ref CE 挑位置），"
                                 "不再叠 --gate-weight（那是按位置下标挑）")
        thresh = self.args.select_thresh
        first = int(getattr(self.args, "select_first", 0) or 0)
        gw = float(getattr(self.args, "gate_weight", 0) or 0)
        gn = int(getattr(self.args, "gate_n", 2) or 2)
        span_word = (getattr(self.args, "select_word", "") or "").strip()
        lead = int(getattr(self.args, "select_lead", 2) or 0)
        mass = float(getattr(self.args, "select_mass", 0.0) or 0.0)
        grow = int(getattr(self.args, "select_grow", 0) or 0)
        fill = int(getattr(self.args, "select_fill", 0) or 0)
        cluster = bool(getattr(self.args, "select_cluster", False))
        extra = int(getattr(self.args, "select_extra", 1) or 0)
        sw = float(getattr(self.args, "span_weight", 0.0) or 0.0)
        if sw > 0 and not (span_word or cluster or mass > 0 or all_on):
            raise SystemExit("--span-weight 要知道「给哪一段加权」：用 --select-mass 或 "
                             "--select-cluster（通用，由 surprise 自己定是哪一段）、"
                             "--select-tokens all（全段留、按 ref CE 定难位置），"
                             "或 --select-word（诊断用，指定词）")
        if sw > 0 and (first or gw > 0):
            raise SystemExit("--span-weight 本身就是「整句留在 loss + 局部加权」，"
                             "与 --select-first 的截断、--gate-weight 的另一种加权互斥")
        if first and gw > 0:
            raise SystemExit("--select-first 是截断、--gate-weight 是整句加权，两者互斥："
                             "前者正是复读环的成因，不要叠加")
        if span_word and gw > 0:
            raise SystemExit("--select-word 已经把范围裁到目标词那一段，--gate-weight 的「前 N 个」"
                             "指的就不再是闸门了：两者互斥，闸门用 --select-first 1 单独给")
        if cluster and (first or span_word or mass > 0 or gw > 0):
            raise SystemExit("--select-cluster 自己就决定「哪一段 + 哪几个孤立高点」，"
                             "与 --select-first / --select-word / --select-mass / --gate-weight 互斥")
        if mass > 0 and (first or span_word):
            raise SystemExit("--select-mass 是通用口径（按 surprise 质量占比排），"
                             "--select-first / --select-word 是指定位置口径，三者只留一个")
        if mass > 0 and gw > 0:
            raise SystemExit("--select-mass 已经保证闸门（CE 最高的那几个）在 loss 里，"
                             "再叠 --gate-weight 是重复计权")
        if mass > 0 and cluster:
            raise SystemExit("--select-mass 与 --select-cluster 都在决定「哪一段最意外」，"
                             "只留一个：先用 ``--select-mass 0.45 --span-weight 20`` 那种有界口径，"
                             "cluster 在「整段都意外」的数据上会退化成一整段")
        if all_on and (cluster or mass > 0 or span_word):
            raise SystemExit("--select-tokens all 自己就用 --select-thresh 定「哪些位置加权」，"
                             "与 --select-cluster / --select-mass / --select-word 互斥")
        tot_sel = tot_ans = 0
        span_sel = span_all = 0
        hot_all = 0
        skipped = 0
        tot_w = gate_w = 0.0
        for it in self.items:
            slp, _ = self.eng.score(it["seq"])
            ce = [-float(v) for v in answer_seg(it, slp)]
            hot = None
            if all_on:
                m, hot = guard.select_all(ce, thresh)
                mask = np.asarray(m, dtype=bool)
                hot_all += len(hot)
                span_all += len(hot)
            else:
                mask = np.asarray(guard.select_mask(ce, thresh, first), dtype=bool)
            if cluster:
                # 通用口径二：中心化后求最大子段（最大的那一件事），再补孤立高点。
                # 配 --span-weight 时它**不截断**：整句留在 loss 防自环，这一段只是加权 ——
                # 与 --select-word 那条路同一种语义，区别只是「哪一段」由 surprise 自己定，
                # 调用方不需要知道内容是名字、数字还是代码。
                cmask = np.asarray(guard.select_cluster(ce, extra=extra, ctx=1), dtype=bool)
                if sw > 0:
                    hot = [i for i, v in enumerate(cmask) if v]
                    hot_all += sum(1 for i in hot if i < mask.size and mask[i])
                else:
                    mask = cmask
                    span_all += int(mask.sum())
            if mass > 0:
                # 通用口径：按 ref CE 降序取到 surprise 总量的 `mass` 比例，与内容是
                # 名字、数字还是代码无关。`--select-tokens off` 在这里也不生效（那是「不选」）。
                # 配 --span-weight 时同样**不截断**：这个有界的选段只当「加权段」，
                # mask 保持整句（截断会把没监督的位置变成 argmax 自环，g2/g3 就是这么失败的）。
                nmask = np.asarray(guard.select_novel(ce, mass=mass, grow=grow,
                                                      fill=fill, thresh=0.0), dtype=bool)
                if sw > 0:
                    hot = [i for i, v in enumerate(nmask) if v]
                    hot_all += sum(1 for i in hot if i < mask.size and mask[i])
                    span_all += len(hot)
                else:
                    mask = nmask
                    span_all += int(mask.sum())
                    span_sel += int(mask.sum())
            if span_word:
                if it.get("region") is not None:
                    raise SystemExit("--select-word 只接了单轮路（--conv-data 的多段 region 还没接："
                                     "每段的答案边界不同，需要按段找 span）")
                texts = [self.tok.decode([i]) for i in it["seq"][it["start"]:]]
                sp = guard.find_token_span(texts, span_word)
                if sp is None:
                    raise SystemExit(
                        f"--select-word {span_word!r} 在答案段里找不到。答案段 = "
                        f"{''.join(texts)[:200]!r}")
                keep = np.asarray(guard.span_mask(mask.size, sp, lead), dtype=bool)
                hot = [i for i, v in enumerate(keep) if v]
                span_all += int(keep.sum())
                hot_all += sum(1 for i in hot if mask[i])
                if sw > 0:
                    # 整句留在 loss（每位都有梯度 ⇒ 不会自环），目标词那几位权重高。
                    pass
                elif first:
                    # 闸门课程 + 目标词：两个瓶颈一起推。中间的「前缀同义词」token
                    # （独立开发→自主研发 那一类）不进 loss —— 它们只是把预算花掉。
                    mask = mask | keep
                else:
                    # 只留目标词这一段：excess 不再二次筛选，这几个位置就是「必须学会的」。
                    mask = keep
                span_sel += int(mask.sum())
                it["span"] = sp
            it["sel"] = mask
            # 引擎的位置口径掩码（长度 n−1，见 encode.engine_keep）：SCORE 只算这些行。掩码一旦定下
            # 就不再变（锚在 θ₀），所以在这里算一次、之后每一步都用它。
            it["keep"] = engine_keep(it, mask)
            if hot is not None and sw > 0:
                w = guard.span_weights(mask, hot, sw)
            else:
                w = guard.gate_weights(int(mask.sum()), gw, gn)
            it["w"] = w
            if w is not None:
                tot_w += sum(w)
                gate_w += sum(w[:gn])
            tot_ans += int(mask.size)
            tot_sel += int(mask.sum())
            if mask.size and not mask.any():
                skipped += 1
        if tot_sel == 0:
            raise SystemExit(
                f"--select-thresh {thresh} 下一个 token 都没选中：阈值过高，或数据与训练起点"
                "无差异。先降阈值看分布再跑，不要空转")
        msg = f"token selection (ref CE > {thresh}): {tot_sel}/{tot_ans} 答案 token 进 loss"
        if cluster:
            msg = (f"token selection (max surprise cluster + {extra} isolated peak): "
                   f"{tot_sel}/{tot_ans} 答案 token 进 loss"
                   f"（逐行减去均值后求最大子段 = 最大的那一件事，再按 ref CE 补 {extra} 个"
                   f"段外高点；与 token 内容无关）")
        if mass > 0:
            msg = (f"token selection (surprise mass {mass:g}, grow {grow}: "
                   f"{tot_sel}/{tot_ans} 答案 token 进 loss"
                   f"（逐行按 ref CE 降序取到总 surprise 的 {100 * mass:.0f}%，"
                   f"；相邻选中位置的空隙 ≤{fill + 1} 就填满，不向簇外扩"
                   f"（与 token 内容无关）")
        if all_on:
            msg = (f"token selection: all —— {tot_sel}/{tot_ans} 个答案 token **全部**进 loss"
                   f"（每位都有梯度 ⇒ 不会像 excess 那样漂走）")
        if sw > 0 and hot_all:
            label = ("--select-word {!r}".format(span_word) if span_word
                     else (f"--select-mass {mass:g}" if mass > 0
                           else ("--select-tokens all 的 ref CE > " + f"{thresh:g}" if all_on
                                 else "--select-cluster")))
            msg += (f"；{label} ×{sw:g}（整句 {tot_sel} 位全留在 loss 里防自环，"
                    f"加权段 {span_all} 位、其中 {hot_all} 位确实进了 loss ⇒ 占 loss 权重 "
                    f"{100.0 * (sw * hot_all) / max(tot_w, 1e-9):.0f}%）")
        elif span_word:
            if first:
                msg += (f"；--select-word {span_word!r} 并入「前 {lead} 个 token + 目标词」"
                        f"那 {span_all} 个位置（闸门 + 名字段一起推，中间的前缀同义词不进 loss）")
            else:
                msg += (f"；--select-word {span_word!r} 把范围裁到「前 {lead} 个 token + 目标词」"
                        f"那 {span_all} 个位置，共 {span_sel} 个 token 进 loss")
        if first:
            msg += f"（--select-first {first}：每条只留前 {first} 个，闸门课程）"
        if tot_w and gw > 0 and not all_on:
            msg += (f"；--gate-weight {gw:g}×前 {gn} 个 ⇒ 闸门占 loss 权重 "
                    f"{100.0 * gate_w / tot_w:.0f}%（其余 token 全部保留在 loss 里）")
        if skipped:
            msg += f"；{skipped}/{len(self.items)} 条起点全对，整条跳过"
        print(msg, flush=True)

    def steps_per_epoch(self) -> int:
        """一轮 = 过一遍数据的步数。ce/dpo 一条样本一步；grpo/luspo 覆写成 items//group-size。"""
        return max(1, len(self.items))

    def loss_at(self, th: dict, batch: list[dict]) -> float:
        raise NotImplementedError

    def batch_at(self, step: int) -> list[dict]:
        """默认逐条轮转（--batch-rows 控制窗口）。grpo 覆写：一步一组。"""
        items, k = self.items, self.use_rows_n
        start = step % len(items)
        return [items[(start + i) % len(items)] for i in range(k)]

    def trainable_facs(self) -> tuple[str, ...]:
        """参与更新的 LoRA 因子。--zo-factors b（默认）时 A 是冻结的随机基：B=0 处
        ∂L/∂A 恒为 0，它只是把梯度投到 rank-r 子空间的随机基，不需要学。"""
        return ("a", "b") if getattr(self.args, "zo_factors", "b") == "ab" else ("b",)

    def _zo_step(self, theta: dict, vel: dict, rng, batch: list[dict], layers: list,
                 lr: float, slot: int | None = None,
                 persist_state: bool = True) -> tuple[list[float], float, float, float | None, float]:
        """一步零阶更新 —— 算法在 `zo/optim.py`，这里只负责把 batch 绑成标量损失。

        本方法曾经的 86 行实现（采方向、±ε 差分、动量、步长上限、rel 读数）已整体搬到
        `training/zo/optim.Mezo`：那里只认 θ、一个 `loss_fn(th) -> float`、lr，不认识
        引擎、SCORE、batch、数据或评测。数值逐位不变 ——
        `tests/test_zo_optim.py::TestBitwiseParityWithOldImpl` 用同 seed 同损失桩对旧
        实现做 3 步逐位比对，4 组超参（默认 / rel-cap / step-max / 无动量）全部相等。

        留这个转发而不是让调用点直接调 `Mezo`，是为了让 `--lr-finder` 与主循环继续
        共用**同一个**入口：两边一旦各写一遍，扫出来的 lr 就不适用于真实训练
        （2026-11-04 踩过）。

        slot / persist_state（--zo-subspace 专用）：
          slot    batch-rows 1 时传 `step % len(items)`。结构化方向的「创新低」判据
                  需要 loss 跨步可比，但逐条轮转下不同样本的 loss 量级差 ~10×
                  （2026-11-16 实测 between/within 方差比 69.6）——创新低的步几乎
                  必然是「轮到简单样本」的步。这里给 loss_fn 包一层 per-item EMA
                  基线（`raw − base[slot]`）：基线在 ±ε 两点同减一个常数 ⇒ **pj 逐位
                  不变**，只有 mids（= 基的收集判据）恢复可比性。
          persist_state False = lr-finder 路径：不读不写基/槽位基线（扫完即弃，
                  否则扫描期的方向会灌进主循环的基里）。
        """
        args = self.args
        use_sub = bool(getattr(args, "zo_subspace", 0)) and persist_state
        opt = Mezo(self.layout, layers,
                   eps=args.eps,
                   dirs=args.dirs,
                   momentum=args.momentum,
                   facs=self.trainable_facs(),
                   rel_cap=getattr(args, "zo_rel", 0.0) or 0.0,
                   step_max=getattr(args, "zo_step_max", 0.0) or 0.0,
                   one_point=bool(getattr(args, "zo_one_point", False)),
                   subspace=getattr(args, "zo_subspace", 0) or 0,
                   subspace_noise=float(getattr(args, "zo_subspace_noise", 0.2) or 0.2),
                   subspace_mode=getattr(args, "zo_subspace_mode", "lozo"),
                   subspace_refresh=getattr(args, "zo_subspace_refresh", 50) or 50)
        # residual-feedback 的 `prev_loss` 是**跨步状态**，但 Mezo 实例在这里每步重建
        # （lr 每步变、lr-finder 每档重置，所以不能简单缓存实例）。状态改由训练器持有：
        # 不这么做的话每步都被当成「首步」，会重新算一次中心基线 ⇒ 仍然是每步 2 次
        # 前向，还白白背上单点估计的偏差（2026-11-15 实测 800 步 260 s vs 两点 276 s，
        # CE 降幅却腰斩，就是这个 bug）。
        if opt.one_point:
            opt.prev_loss = getattr(self, "_zo_prev_loss", None)
        # 结构化方向基同样是跨步状态（正交基 + 最低 loss + 形状缓存），同一机理：
        # 每步重建实例，状态从训练器取，跑完存回。首步训练器还没有这些属性，
        # 缺省值必须与 Mezo.__init__ 的初始值同型（None 会把 list/dict 换坏）。
        _zo_state_defaults = {"_basis": [], "_best_loss": None, "_seg": None,
                              "_shape_cache": {}, "_d": None,
                              "_lozo_since_refresh": 0}
        if use_sub:
            for _k, _dv in _zo_state_defaults.items():
                setattr(opt, _k, getattr(self, f"_zo_{_k}", None)
                        if hasattr(self, f"_zo_{_k}") else _dv)
            # per-item 基线：该槽位还没访问过（首访问）时不收基也不记 best ——
            # 否则首个简单样本把 _best_loss 钉在假低位。
            base = (getattr(self, "_zo_slot_base", None) or {}).get(slot) \
                if slot is not None else None
            opt.allow_push = base is not None or slot is None
            raw_fn = lambda th: self.loss_at(th, batch)  # noqa: E731
            loss_fn = raw_fn if base is None else (lambda th: raw_fn(th) - base)
        else:
            loss_fn = lambda th: self.loss_at(th, batch)  # noqa: E731
        if getattr(self, "_prof", False):
            _inner = loss_fn

            def loss_fn(th):                     # noqa: F811 —— 只包一层计时，语义不变
                t0 = time.perf_counter()
                v = _inner(th)
                self._prof_acc["forward"] += time.perf_counter() - t0
                self._prof_acc["n"] += 1
                return v
        out = opt.step(theta, vel, lr, loss_fn, rng, step_max=self._step_max_at(lr))
        # 上限钉死 = lr 曲线被重归一化吃掉。要计数，否则「调了 --lr-schedule 却没变化」
        # 在日志里完全看不出来（2026-11-21：v4 冷启动 1500 步把 12 题数学打成 1/12，
        # 长锚漂移 −0.73 nats/token = 噪声底 150×，而 lr 一直显示 1.00e-03）。
        self._zo_clamped_n = getattr(self, "_zo_clamped_n", 0) + (1 if opt.clamped else 0)
        self._zo_step_n = getattr(self, "_zo_step_n", 0) + 1
        if opt.one_point:
            self._zo_prev_loss = opt.prev_loss
        if use_sub:
            for _k in _zo_state_defaults:
                setattr(self, f"_zo_{_k}", getattr(opt, _k))
            # 槽位基线 EMA（β=0.9，窗口 ~120 步）。用 raw mid 更新：out[3] 是归一化值。
            if slot is not None and out[3] is not None:
                raw_mid = float(out[3]) + (base if base is not None else 0.0)
                bases = getattr(self, "_zo_slot_base", None)
                if bases is None:
                    bases = self._zo_slot_base = {}
                bases[slot] = raw_mid if base is None else 0.9 * base + 0.1 * raw_mid
        return out

    def lr_at(self, step: int) -> float:
        """本步学习率 —— 曲线在 `common/schedule.py`，两条训练线同一份。

        留这个转发是为了 `--lr-finder` 与主循环继续读同一个入口（run 里每步调一次；
        两条路各写一遍就会像 2026-11-04 那样，扫出来的 lr 不适用于真实训练）。
        """
        args = self.args
        return lr_at(args.lr_schedule, args.lr, args.lr_final, args.lr_warmup,
                     args.steps, step)

    def _step_max_at(self, lr: float) -> float | None:
        """本步的绝对位移上限 = `--zo-step-max` × lr 曲线的相对值（const 曲线不缩放）。

        为什么必须缩放：`zo/optim.py` 的裁剪作用在**施加的位移**上，
        `scale = step_max / ‖lr·v‖` 把范数重归一化回 step_max ⇒ 只要上限被钉死，
        `--lr-schedule` 与 `--lr-final` 对位移就完全没有影响，θ 以恒定速度随机游走。
        实测（2026-11-21，identity v4 冷启动，step-max 0.03 / lr 1e-3 / 1500 步）：
        θ 范数 2.82→6.95（2.46×），长锚 per-token logp 漂移 −0.728 nats（噪声底 0.0047
        的 155×），保持集 8/12→1/12（McNemar p=0.016，12 条全 `length` 自环）。
        上限跟着曲线收手后同一个 run 才有「先学事实、后停手」的形状。
        """
        sm = float(getattr(self.args, "zo_step_max", 0.0) or 0.0)
        if sm <= 0:
            return None
        if getattr(self.args, "lr_schedule", "const") == "const":
            return sm
        base = float(getattr(self.args, "lr", 0.0) or 0.0)
        if base <= 0:
            return sm
        return sm * max(0.0, min(1.0, lr / base))
    def lr_finder(self) -> int:
        """LR range test 的零阶改造版：每档从 θ₀ 重置，只扫一个短窗口。

        与原文的两处差别，都是零阶逼出来的：
        1. 原文连跑一条 lr 递增曲线看 loss 拐点。零阶单方向 pj 信噪比只有 ~0.15
           （见 `_snr.py`），连续爬升会把前一档的漂移灌进后一档 —— 所以每档重置。
        2. 判据不看 loss 最低点，改看 **ΔCE/步 与 rel**：ΔCE 要显著为负，且
           rel=‖lr·v‖/‖θ‖ 留在 0.03 以内（>0.05/步 必发散，见 run() 里的经验线）。
           每档用同一 seed 的随机方向，各档之间只有 lr 不同。
        """
        args, layers = self.args, sorted(self.layout)
        theta0 = {l: {w: (a.copy(), b.copy()) for w, (a, b) in self.theta[l].items()}
                  for l in layers}
        self.swap(theta0, self.final_path)
        self.prepare_selection()                 # 掩码锚在 θ₀：扫描的各档共用同一份
        ce0 = self.answer_ce()
        lrs = list(np.geomspace(args.lr_min, args.lr_max, args.lr_probes))
        print(f"lr-finder: theta0 answer-ce {ce0:.4f}; {len(lrs)} probes x "
              f"{args.lr_probe_steps} steps, dirs {args.dirs}", flush=True)
        rows = []
        for lr in lrs:
            # 每档从 θ₀ 重置：单点估计的上一扰动点属于另一档，不能带过去。
            self._zo_prev_loss = None
            theta = {l: {w: (a.copy(), b.copy()) for w, (a, b) in theta0[l].items()}
                     for l in layers}
            vel = {l: {w: (np.zeros_like(a), np.zeros_like(b))
                       for w, (a, b) in theta[l].items()} for l in layers}
            rng = np.random.default_rng(args.seed + 1)   # 每档重置：同一批方向
            rels, pjs = [], []
            for s in range(args.lr_probe_steps):
                pj_vals, _n, rel, _lm, _mv = self._zo_step(
                    theta, vel, rng, self.batch_at(s), layers, float(lr),
                    persist_state=False)   # 扫描期不读不写基/槽位基线，防灌进主循环
                rels.append(rel)
                pjs.append(float(np.mean(pj_vals)))
            self.swap(theta, self.final_path)
            per = (self.answer_ce() - ce0) / args.lr_probe_steps
            # rel 的**第一步**必然是 NaN（B 从 0 起，‖θ_能动‖=0，除零），普通 mean 会把整档
            # 都报成 nan —— 2026-11-14 实测五档全 nan，看上去像「rel 算不出来」。跳过它。
            rel_mean = float(np.nanmean(rels)) if np.any(np.isfinite(rels)) else float("nan")
            rows.append((float(lr), per, rel_mean, float(np.mean(pjs))))
            print(f"  lr {lr:9.2e}  dCE/step {per:+.5f}  rel {rel_mean:.2e}"
                  f"  pj {np.mean(pjs):+.2e}", flush=True)
        self.swap(theta0, self.final_path)               # 收敛后把引擎还原成 θ₀
        print("lr-finder: 选「dCE/step 显著为负 且 rel<0.03」的最大 lr 作 --lr 上限，"
              "再接 --lr-schedule cosine --lr-final 0", flush=True)
        ok = [r for r in rows if r[1] < 0 and r[2] == r[2] and r[2] < 0.03]
        if ok:
            b = max(ok, key=lambda r: r[0])
            print(f"  推荐上限 --lr {b[0]:.2e}（dCE/step {b[1]:+.5f}, rel {b[2]:.2e}）", flush=True)
        else:
            print("  没有一档同时满足「ΔCE 为负 + rel<0.03」：信号可能被方向方差盖住了，"
                  "加大 --dirs 或 --lr-probe-steps 重扫", flush=True)
        self.eng.close()
        return 0

    def ce_pairs(self) -> list[tuple[str, str]] | None:
        """健康度口径（--ce-data）：目标文本的 (prompt, gold)。

        RL 目标（dpo/grpo/luspo）的 loss 不是「某一句话的 CE」，但**学没学到目标文本**
        仍然要看目标文本的 CE —— 否则每轮只有一个词的 logp，判据太弱（2026-11-04：
        用 lr 1e-6 跑 luspo 时，唯一的主指标是 logp('Strata'|你是谁)，和训练目标无关）。
        "identity" 走内置身份集；其他按 jsonl 读 prompt/answer（{dev}/{word} 会替换）。
        """
        if self._ce_pairs is not None:
            return self._ce_pairs or None
        d = (getattr(self.args, "ce_data", "") or "").strip()
        if not d:
            self._ce_pairs = []
            return None
        w = self.args.word
        if d == "identity":
            pairs = [(q, a) for q, a in load_data("identity", w)]
        else:
            pairs = []
            for line in open(d, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                a = r.get("answer") or r.get("gold") or r.get("completion") or ""
                pairs.append((r["prompt"], a.replace("{dev}", w).replace("{word}", w)))
        self._ce_pairs = pairs
        return pairs or None

    def answer_ce(self) -> float | None:
        """当前 θ 下「训练目标本身」的全量度量（nats/token，越低越好）。

        2026-11-02 的教训：identity 数据重写后答案不再以 {word} 开头，probe 的
        logp(word|prompt) 与训练目标脱钩 —— 那条线只能当参考，**验收看这里**。
        ce 子类用它自己训的那份 items；dpo/grpo/luspo 要 --ce-data 才量得出来。
        """
        pairs = self.ce_pairs()
        if not pairs:
            return None
        total, ntok = 0.0, 0
        for q, gold in pairs:
            seq, plen = enc_answer(self.tok, self.render_off, q, gold, eos=self.eos)
            slp, _ = self.eng.score(seq)
            seg = np.asarray(slp[plen:], dtype=np.float32)
            total += -float(np.sum(seg))
            ntok += len(seg)
        return total / max(ntok, 1)

    # ---- 骨架：ZO 估计循环（写一次，三个算法共用） ---------------------------
    def seq_lens(self, it: dict) -> list[int]:
        """一条样本里要进 SCORE 的序列长度。DPO 是 chosen/rejected 两条，其余一条。

        `--prefill` 与 `--max-context` 的检查都按「最长的那条序列」定，所以这里返回列表
        而不是单个长度：dpo 的偏好对里 rejected 常常比 chosen 长（2026-11-12 踩到：
        主流程直接读 `it["seq"]`，`--algo dpo --method mezo` 起不来，KeyError: 'seq'）。
        """
        if "seq" in it:
            return [len(it["seq"])]
        if "c_seq" in it:
            return [len(it["c_seq"]), len(it["r_seq"])]
        return []

    def _check_context_fit(self) -> None:
        """最长样本超过 `--max-context` 就**停下**，不截断、不打折。

        为什么是中止而不是告警：引擎已经按 `--max-context` 起好了，事后改参数无效；
        而截断后的 logp 不是整条轨迹的值，loss 只差一小截，日志上看不出来 —— 但梯度
        已经把「被腰斩的轨迹」当成负例。2026-11-05 在评测侧踩过同型坑：1536 预算下
        5 条错例**全部**是截断，放到无上限后 12/12 全对。
        默认 4096 只够身份那种 2 token 答案；长 CoT 必须显式抬上下文。
        """
        if not self.items:
            return
        lens = [n for it in self.items for n in self.seq_lens(it)]
        mx, cap = max(lens), int(self.args.max_context)
        if mx > cap:
            need = ((mx + 1023) // 1024) * 1024
            raise SystemExit(
                f"最长样本 {mx} token > --max-context {cap}：SCORE/GEN 会把它截断，"
                f"loss 与 logp 都不是整条序列的值。\n"
                f"  加 --max-context {need}（或更大）重跑；显存会随上下文上升，"
                f"见 mem_guard 的估算与提示。")
        if mx > cap * 0.9:
            print(f"  提示：最长样本 {mx} token 已到 --max-context {cap} 的 90%，"
                  f"再长就会被截断（不会静默：超了会直接停）", flush=True)

    def probe_len(self) -> int:
        """自检 probe 的长度 = 引擎真实的 prefill chunk，**不是常数**。

        判据是「读数落在确定区」：`n >= chunk` 时 SCORE 逐位可重复，`n < chunk` 时抖 3–5 nats/序列
        （2026-11-16 `diag_score_repeat.py`：chunk 8192 下 n=256 抖、n=8192/32768/130514 全 +0.00）。
        所以 probe 至少一个 chunk —— 但**不能写死 8192**：identity 这类短数据的 chunk 是 256，
        拿 8192 去问等于让引擎切 32 块，native QSA indexer 的 batch 路径会直接报
        「bad geometry, positions, parameters or scaling」（2026-11-16 实测，写死 8192 时必现，
        训练在 `check_start_state` 就崩）。取 chunk 同时满足两件事：落在确定区、不跨块。
        """
        v = str(getattr(self.args, "prefill_chunk", "") or "")
        if v.isdigit() and int(v) >= 2:
            return int(v)
        return 2048   # 引擎在 --no-prefill-borrow 且没给 --prefill 时自己挑的 chunk

    def probe_ids(self, n: int = 0) -> list[int] | None:
        """自检用的 probe。取「任意目标都能拿到」的那条序列，截到 n 个 token。

        ce 的样本叫 `seq`，dpo 的是 `c_seq`/`r_seq`（偏好对没有单独的 `seq`），RL 是
        `seq`。原来两处自检都写死 `it["seq"]` ⇒ **DPO 下它们一次都没执行过**：
        2026-11-16 那次 DPO 从头到尾没有 `adapter:` / `start state:` 行，`--init-adapter`
        的 θ 不一致也就没被拦住。自检的覆盖面必须跟着 objective 走，不能只跟 ce 走。

        **长度必须是 8192，不是 256**（2026-11-16 实测，`diag_score_repeat.py`）：同一条序列、
        同一个 adapter 连打两次 SCORE ——

            n=256    Δ(sum) −15.98 / −0.28 / +9.04（max|Δtok| 3.0–4.7 nats）
            n=8192   Δ(sum) +0.00，逐 token 位相同
            n=32768  Δ(sum) +0.00
            n=130514 Δ(sum) +0.00

        与预热无关（先跑 8192 再跑 256 照样抖），与 adapter 也无关（B=0 时同样抖）；
        短序列落在 SCORE 的不可重复区，单 token 可差 3–5 nats。用 256 做 probe 时自检
        读到 −49.97 的假差异（128K 那次的 128KiB chunk 配置），而三个 gguf 逐位相同。

        代价：8192 token 一次 SCORE 约 1–2 s（CHUNK 8192），比 256 的 <1 s 贵但可忽略。
        短样本（identity 这类 ~100 token）**循环填充到 8192**：自检问的是「adapter 进没进前向」
        与「θ 与引擎是否同一份」，这两个问题的答案与序列内容无关，只要求读数落在确定区。
        填充是必需的 —— 不填就退回抖动区，`d > 1e-6` 这条判据会被 3–5 nats 的抖动假通过。
        """
        n = n or self.probe_len()
        for it in self.items:
            for key in ("seq", "c_seq", "r_seq"):
                s = it.get(key)
                if s is not None and len(s) >= 2:
                    base = list(s)
                    if len(base) >= n:
                        return base[:n]
                    reps = (n + len(base) - 1) // len(base)
                    return (base * reps)[:n]
        return None

    def check_start_state(self, swap, tmp_path: str) -> None:
        """证伪「引擎起点 == 本进程的 θ」：同一条序列，as-loaded 与 swap(θ) 后必须同一个读数。

        为什么需要（2026-11-16 实测）：`--init-adapter` 曾经只把 gguf 交给引擎 `--lora`，
        进程内 θ 还是 `init_theta()` 的 B=0 骨架。DPO 于是 **ref = SFT、policy = 基座 + εb**
        地跑了 2000 步：每一步 pj 正常、‖θ‖ 正常涨、rc=0，只有两个数字不对劲 ——
        `before:` 的 probe 是 −51.69（SFT 终点 −21.15）、answer-ce 3.19（终点 0.30）。
        训练器看不出异常，因为「引擎加载了什么」和「θ 是什么」从来没被比对过。

        阈值：两边都是同一份数值、同一条 SCORE 路径，差异只该来自浮点求和顺序（实测
        16k token 上是 0.004 nats/tok）。取 256 token ⇒ 绝对差应在 1 nat 以内；
        真出这个 bug 时差是几十上百 nats，判据不会含糊。
        """
        probe = self.probe_ids()
        if probe is None:
            return
        s0, _ = self.eng.score(probe)                    # 引擎当前加载的那份（init 文件）
        swap(self.theta, tmp_path)                       # 把进程内 θ 推给引擎
        s1, _ = self.eng.score(probe)
        d0, d1 = float(np.sum(s0)), float(np.sum(s1))
        tol = max(1.0, 1e-3 * abs(d0))
        if len(probe) < 8192:
            # 短样本落在 SCORE 的不可重复区（实测 256 token 连打两次可差 ±16 nats，
            # 与 adapter 无关、与预热无关）。这里只做「量级」判断：真不一致是几十上百 nats。
            tol = max(tol, 32.0)
        if not (abs(d1 - d0) <= tol):
            raise SystemExit(
                "起点不一致：引擎加载的 adapter 与进程内的 θ 不是同一份。\n"
                f"  同一条 {len(probe)} token 序列：as-loaded {d0:.2f}，swap(θ) 后 {d1:.2f}"
                f"（差 {d1 - d0:+.2f}，容忍 {tol:.2f}）。\n"
                "  含义：`before:` / θ₀ 缓存用的是引擎那份、每步 `swap(θ±εb)` 写的是另一份 ——\n"
                "  训练会在「起点不是它以为的 θ₀」的状态下跑完，且日志上没有别的异常。\n"
                "  排查：--init-adapter 是否被真的读回成 θ（`init theta ←` 那行），"
                "--resume 的 npz 层是否被恢复。")
        print(f"start state: as-loaded {d0:.2f} == swap(θ) {d1:.2f}"
              f"（差 {d1 - d0:+.3g}）⇒ θ 与引擎起点一致", flush=True)

    def check_adapter_effect(self, swap, tmp_path: str, layers: list[int]) -> None:
        """证伪一次「adapter 到底进没进前向」：B 随机化后同一段 token 的 logp 必须变。

        为什么要有这一步（2026-11-15 实测）：引擎的 prompt 有两条走法（chunk-major 的
        `run_impl` 与 layer-major 的 `lm_run`），LoRA 曾经只在前者里 apply。走后者时
        adapter 被**静默忽略**：同一序列、同一 adapter，chunk 走法 SSUM −222,017，
        layer-major 走法 −193,265 —— 正好等于不带 adapter 的基座 −193,330。对零阶训练，
        它的表现是「pj 恒等于 0、θ 一步没动」，与「学不动」在日志上无法区分。

        代价：两次 256 token 的 SCORE（<1 s），与上下文长度无关（只取序列前 256 个 token）。
        """
        probe = self.probe_ids()
        if probe is None:
            return
        theta = self.theta
        rng = np.random.default_rng(int(self.args.seed) + 977)
        bumped = {}
        for l in layers:
            bumped[l] = {}
            for w in theta[l]:
                a, b = theta[l][w]
                noise = (0.01 * rng.standard_normal(b.shape)).astype(np.float32)
                bumped[l][w] = (a, b + noise)
        swap(theta, tmp_path)
        s0, _ = self.eng.score(probe)
        swap(bumped, tmp_path)
        s1, _ = self.eng.score(probe)
        d = max((abs(x - y) for x, y in zip(s0, s1)), default=0.0)
        if not (d > 1e-6):
            raise SystemExit(
                "adapter 检查失败：把选中层的 B 换成随机数后，同一条 %d token 序列的 logp 一点没变"
                "（最大差 %.3g）。\n"
                "  含义：引擎的前向没有应用 LoRA —— 零阶差分恒为 0，训练会「跑完但什么都没发生」。\n"
                "  排查顺序：① 引擎是不是走了 layer-major prompt（STRATA_PREFILL_LAYER_MAJOR=1）；\n"
                "  ② LORA 命令是否成功（引擎日志里应有 'LoRA: ... r=' 行）；③ --lora 是否指向本步写出的文件。\n"
                "  确认这是已知限制而不是 bug 时才加 --no-adapter-check 跳过。"
                % (len(probe), d))
        swap(theta, tmp_path)
        print(f"adapter: B 随机化后 {len(probe)} token 的 logp 最大变化 {d:.3g}"
              f" ⇒ adapter 进了前向（层 {layers[0]}..{layers[-1]}）", flush=True)

    def run(self) -> int:
        args, eng, tok, render = self.args, self.eng, self.tok, self.render
        layers = sorted(self.layout)
        # `--zo-layers K` / `--zo-range L0,L1`：只把选中的层放进零阶更新的范围。
        # 理由是可算的：单方向 rank-1 探测的**信噪比是 1/√d**，d 是可训维数（= Σ r·(n_in+n_out)）。
        # 48 层 rank8 ⇒ d = 3.34e6；末 2 层 rank4 ⇒ d = 6.97e4（约差 48 倍）。
        # **但 d 只在「同层同矩阵」之间可比**：末 2 层全是输出投影，它与事实召回所在的中间层
        # MLP 根本不是同一个函数 —— 2026-11-15 之前每个 run 都在这个子空间里跑（见 guard.pick_layers）。
        _zl = int(getattr(args, "zo_layers", 0) or 0)
        _zr = str(getattr(args, "zo_range", "") or "")
        try:
            _keep, _desc = guard.pick_layers(layers, _zl, _zr)
        except ValueError as e:
            raise SystemExit(f"层选择参数有误：{e}") from None
        _d = sum(args.rank * (ni + no) for l in _keep for ni, no in self.layout[l].values())
        _d_all = sum(args.rank * (ni + no) for l in layers for ni, no in self.layout[l].values())
        if len(_keep) != len(layers):
            print(f"zo-layers: 只更新 {_desc}，d = {_d}"
                  f"（全层 {_d_all}，{100.0 * _d / max(1, _d_all):.1f}%）", flush=True)
        else:
            print(f"zo-layers: {_desc}，d = {_d}", flush=True)
        layers = _keep
        # 「能改什么」必须让日志自己带出来：全是 attn_output/ssm_out 时，模型只可能学到
        # 一个与位置无关的 logit 偏置 —— 那和「学会在哪个位置说什么」是两件事。
        _kinds = {}
        for l in layers:
            for w in self.layout[l]:
                _kinds[w] = _kinds.get(w, 0) + 1
        if layers:
            print("  矩阵种类: " + "、".join(f"{k}×{v}" for k, v in sorted(_kinds.items()))
                  + ("（都只投回残差流；MLP 未开放）" if set(_kinds) <= {"ssm_out", "attn_output"} else ""), flush=True)
        # 训练目标里到底有没有「说到哪停」，必须让日志自己带出来：2026-11-15 之前的每一个
        # run 都没有它，而失败签名（`由由由…` / `当前当前…` 的自环）看不出少了什么。
        print(f"answer-eos: {'on' if self.eos is not None else 'OFF'}"
              + (f"（答案末尾接 <|im_end|> = id {self.eos}，它同时进 loss）" if self.eos is not None
                 else "（--no-answer-eos：序列在答案最后一个字上结束，模型不会学收尾）"), flush=True)
        self.plus_path = os.path.join(args.work, "plus.gguf")
        self.minus_path = os.path.join(args.work, "minus.gguf")
        self.final_path = os.path.join(args.work, "final.gguf")

        # 步时去向：SCORE 只是其中一段。`STRATA_MEZO_PROFILE=1` 时按「写盘 / 热加载 / 前向」
        # 三段累计，每 200 步报一次 —— 否则「一步 0.3 s」无法判断该优化哪一段
        # （2026-11-22：`--zo-one-point` 把前向次数砍半，步时只从 310 ms 降到 273 ms）。
        self._prof = os.environ.get("STRATA_MEZO_PROFILE") == "1"
        self._prof_acc = {"write": 0.0, "lora": 0.0, "forward": 0.0, "n": 0}

        def swap(th: dict, path: str) -> None:
            # `--zo-layers` 下 th 可能只含被训的那几层（_zo_step 的 th_plus/th_minus 就是
            # 按 `layers` 拼的）。适配器文件必须写满全层 —— 引擎 lora_update 要求张量集合
            # 与形状不变 —— 未训层沿用 θ₀（B=0、A 仍是初始随机基），语义就是「没动」。
            full = th if len(th) == len(self.layout) else {**self.theta, **th}
            t0 = time.perf_counter() if self._prof else 0.0
            write_adapter(path, self.layout, full, args.alpha)
            t1 = time.perf_counter() if self._prof else 0.0
            eng.lora(os.path.abspath(path))    # 引擎 cwd 是它自己的目录：永不给相对路径
            if self._prof:
                self._prof_acc["write"] += t1 - t0
                self._prof_acc["lora"] += time.perf_counter() - t1
        self.swap = swap

        self.items = self.items or self.load_items()
        print(f"data: {len(self.items)} items ({self.objective})", flush=True)
        self._check_context_fit()
        # 轮次优先：步数会随「一条样本算几遍 / 数据几条」漂移，轮次不会。--epochs 给了就盖掉 --steps。
        # 一轮的步数走 steps_per_epoch()：ce/dpo 是「一条样本一步」，grpo/luspo 是「一组一步」，
        # 用 len(items) 统一口径会让 RL 的 --epochs 5 实际跑 20 轮（2026-11-04 修）。
        if getattr(args, "epochs", 0):
            per_epoch = self.steps_per_epoch()
            args.steps = max(1, int(round(args.epochs * per_epoch)))
            print(f"epochs {args.epochs:g} × {per_epoch} steps/epoch = {args.steps} steps", flush=True)
        self.use_rows_n = (len(self.items) if args.batch_rows in (0, len(self.items))
                           else args.batch_rows)
        _est = (f"estimator: dirs {args.dirs}, momentum {args.momentum}, "
                f"batch-rows {self.use_rows_n}/{len(self.items)}, "
                f"zo-factors {'ab' if args.zo_factors == 'ab' else 'b'}, "
                f"rel-cap {getattr(args, 'zo_rel', 0.0) or 0:g}, "
                f"step-max {getattr(args, 'zo_step_max', 0.0) or 0:g}, "
                f"loss {args.loss_reduction}, objective {self.objective}")
        if getattr(args, "zo_subspace", 0):
            _est += (f", subspace {args.zo_subspace}@{getattr(args, 'zo_subspace_mode', 'lozo')}"
                     f"(nu {getattr(args, 'zo_subspace_refresh', 50)}, "
                     f"noise {getattr(args, 'zo_subspace_noise', 0.2) or 0.2:g})")
        print(_est, flush=True)
        # rel-cap 是**相对**上限：每步 ≤ rel_cap × ‖θ_能动‖。B=0 起点上 ‖θ_能动‖=0，
        # 而它只能靠步长长大 —— 于是 步长 = rel_cap×‖B‖ 变成自锁（2026-11-14 idH 实测：
        # rel_cap 0.01 把每步钉死在 1.4e-4，比 idC 的 6.4e-3 小 45 倍，‖B‖ 17 步没长）。
        # 从零初始化起步只能用绝对 --zo-step-max。θ 加载后（run() 里）还有一道硬门。
        # r=1 且只训 B：ΔW = B·A 秩为 1（A 是 (r,n_in)、B 是 (n_out,r)），
        # Δlogits(h) = (U·B)·(A·h)·alpha/r，而 A·h 在 r=1 时是标量 ⇒ 任意隐藏态下
        # 的词表变化都沿 U·B 这一个方向。2026-11-14 bias_probe.py 实测（idM 4000 步）：
        # `当前` 在训练闸门位置 CE −6.09，在算术/写诗/讲笑话等无关上下文反而 −9.45。
        # 但**秩不是天花板**：idR4 只把 rank 改成 4，闸门 CE 中位 10.87→0.14，
        # 无关上下文 ΔCE 反而涨到 −13.91，生成仍是「当前版本版本版本…」复读环、
        # answer-ce 恶化 +0.985。病根是 --select-first N 只把前 N 个 token 放进
        # teacher-forced CE：它们被推到 ~0，第 N+1 位从来没有 loss，自由生成必然
        # 回落到刚学会的那个 token。提高 rank 只是更快、更彻底地过拟合这 N 个位置。
        if args.rank == 1 and getattr(args, "zo_factors", "b") == "b":
            print("提示：rank 1 且只训 B 时 Δlogits 只有一个固定方向（上下文无关偏置），"
                  "适合移动单个 token 的偏好；但把 rank 提到 4 也修不好多 token 目标"
                  "—— 实测闸门全开后仍复读，且无关上下文偏置更强", flush=True)
        if getattr(args, "select_first", 0):
            print(f"提示：--select-first {args.select_first} 只把每条前 "
                  f"{args.select_first} 个 excess token 放进 loss，是开门/诊断工具，"
                  "不是学整句的方法：第 N+1 个 token 无梯度，自由生成会在此复读。"
                  "要学完整追加句改用 --gate-weight（整句监督 + 闸门加权）", flush=True)
        if getattr(args, "gate_weight", 0):
            print(f"提示：--gate-weight {args.gate_weight:g} 保留整句 excess 监督，"
                  f"只给前 {args.gate_n} 个 token 加权；判据除了闸门 CE，"
                  "还要看第 N+1 位之后的 CE 是否跟着降（截断口径下那里会反弹）", flush=True)

        if getattr(args, "lr_finder", False):    # 扫描完直接退出，不进入训练
            return self.lr_finder()
        if args.lr_schedule != "const":
            print(f"lr schedule: {args.lr_schedule} {args.lr:.2e} -> {args.lr_final:.2e}"
                  f" over {args.steps} steps (warmup {args.lr_warmup:.0%})", flush=True)

        # before：probe word logp（参考线）+ answer-ce（验收线）。引擎此刻在 θ₀，
        # 显式 swap 一次保证度量的是 --resume 后的 θ₀ 而不是引擎启动时的状态。
        swap(self.theta, self.final_path)
        # 探针读数必须与训练/评测走同一条通道（--think-targets off 时 eval_render 是
        # 非思考模板，`render` 是思考模板）。2026-11-15 实测：stage1 的 eval 报 -47.2，
        # 而同一 θ 的 `after:` 报 -52.7 —— 差 5.5 nats 全来自模板，不是权重。
        self.lp0 = name_logprob(eng, tok, self.eval_render, args.probe, args.word)
        self.ce0 = self.answer_ce()
        msg = f"before: logp({args.word!r} | {args.probe!r}) = {self.lp0:.4f}"
        if self.ce0 is not None:
            msg += f"  answer-ce {self.ce0:.4f} nats/tok"
        print(msg, flush=True)

        # 引擎把 adapter 丢掉时，零阶差分**恒等于 0**，而日志只显示「每步 pj≈0、θ 不动」——
        # 与「学不动」无法区分。2026-11-15 实测到真事：引擎的 layer-major prompt 走法没有
        # lora_apply，同一序列同一 adapter 在两条走法下 SSUM 是 −222,017 与 −193,265（后者
        # 等于不带 adapter 的基座）⇒ 开那条走法就是「训练一整晚什么都没发生」。
        # 所以开跑前用一次 256 token 的 SCORE 证伪：B 填随机后读数必须变。
        if not getattr(args, "no_adapter_check", False):
            # 先证「引擎当前的起点就是进程内的 θ」，再证「LoRA 进了前向」。顺序不能反：
            # 后者（B 随机化后 logp 变）在「进程内 θ 是 B=0、引擎却加载了训好的 adapter」
            # 时照样能过，于是 2026-11-16 那次 DPO 一路跑到底都没报警。
            self.check_start_state(swap, tmp_path=self.plus_path)
            self.check_adapter_effect(swap, self.final_path, layers)

        # LAYER REPLAY（--replay）：同一条序列只全读一次，之后每次 SCORE 只走层 [L, 48)。
        # 放在 adapter 自检之后 —— 自检的判据是「B 随机化后读数必须变」，而重放同样吃新表
        # （2026-11-16 实测：同 rank 的 B×0.5 变体换上去，重放读数 Δsum +24827.89），
        # 顺序反了不会漏检，但先自检一遍更省事。
        # 门槛两条，任一条不成立就关掉而不是硬开：
        #   ① 层 L > 0（层 0 之前没有东西可复用）；
        #   ② L 就是 θ 动的第一层（否则层 < L 上还有非零 B，重放读到的是旧残差）。
        # （曾经的门槛③「最长样本 >= 一个 chunk」2026-11-16 撤销，见下方 gate 处的实测记录。）
        # 安全网恒在：首次捕获后引擎侧自动比对一次「全读 vs 重放」逐位。
        _longest = 0
        for _it in self.items:
            for _k in ("c_seq", "seq", "r_seq"):
                if _it.get(_k):
                    _longest = max(_longest, len(_it[_k]))
                    break
        _want = str(getattr(args, "replay", "auto") or "auto").strip().lower()
        _rl = -1
        if _want not in ("off", "0", "none", ""):
            _rl = int(layers[0]) if _want == "auto" else int(_want)
            # 门槛③（最长样本 >= chunk）2026-11-16 撤销：实测推翻了它的前提——
            # 同一条 64-token 序列连续 4 次 SCORE，sum 逐位一致（新 Scorer 固定布局）。
            # 短序列任务（identity 70 token < chunk 256）才是 replay 最大受益者：
            # 每条序列首读捕获一次，之后每次 SCORE 只走 θ 的层。安全网不变：
            # 首次重放后自动比对「全读 vs 重放」逐位，不一致抛错，不静默降级。
            if _rl <= 0:
                print("replay: off（训练层从 0 开始，层 0 之前没有可复用的东西）", flush=True)
                _rl = -1
            elif int(layers[0]) != _rl:
                print(f"replay: off（请求层 {_rl}，但 θ 动的第一层是 {layers[0]} —— 层 {layers[0]}"
                      " 以下才与 θ 无关）", flush=True)
                _rl = -1
        if _rl > 0:
            try:
                eng.enable_replay(_rl)
                print(f"replay: on —— 层 {_rl}..{len(self.layout) - 1}（层 0..{_rl - 1} 与 θ 无关，"
                      f"一条序列只全读一次）；最长样本 {_longest} token，dirs {args.dirs}", flush=True)
            except NotImplementedError as _e:
                print(f"replay: off（{_e}）", flush=True)

        # θ₀ 缓存：此刻引擎里是训练起点的 adapter（init 或 --resume 的 theta）
        swap(self.theta, self.plus_path)
        self.theta0_cache()

        rng = np.random.default_rng(args.seed + 1)
        theta, layout = self.theta, self.layout
        # ---- rel-cap 的硬门（θ 已加载，能看真实范数了）。
        # 分母是「能动的那个范数」：`--zo-factors b` 下就是 ‖B‖，而 B 从 0 起、只能靠步长长大
        # ⇒「每步 ≤ rel×‖B‖」把步长钉在 rel×‖B‖ 上，‖B‖ 再也不动。这不是更保守，是**不动**，
        # 而且不报错（2026-11-14 idH：rel 0.01 把每步压到 1.4e-4，17 步 ‖B‖ 没长）。
        # 所以要约束零阶步长：b 用绝对 --zo-step-max（可加性，N 步总位移按 √N 规划），
        # 或改 ab（‖A‖ 是随机基，常数，rel 才有分母）。
        if getattr(args, "zo_rel", 0.0) and getattr(args, "zo_factors", "b") != "ab":
            _mov0 = theta_norms(theta, layers, ("b",))[1]
            if rel_cap_would_selflock(("b",), _mov0):
                raise SystemExit(
                    f"--zo-rel {args.zo_rel:g} 配 --zo-factors b 会自锁：这一步的上限是 "
                    f"rel×‖B‖，而 ‖B‖ = 0 起点只能靠步长长大 ⇒ 训练不报错但几乎不动。\n"
                    "两条可行路径：\n"
                    f"  --zo-factors ab --zo-rel {args.zo_rel:g}      （A 是随机基，‖A‖ 常数，rel 有分母）\n"
                    f"  --zo-factors b  --zo-step-max X  （绝对步长，可加性，X 取 0.01–0.03）")
            print(f"注意：--zo-rel {args.zo_rel:g} 的分母取已加载的 ‖B‖ = {_mov0:.3g}"
                  "（非零，不会自锁）；这是 --resume 的 θ，请确认它确实非零", flush=True)
        # 动量状态：与 theta 同形状，零初始化
        vel = {l: {w: (np.zeros_like(theta[l][w][0]), np.zeros_like(theta[l][w][1]))
                   for w in theta[l]} for l in layers}
        succ_rows = None
        graded_slots = None
        if args.success_data:
            from _eval_hard import load_rows, success_eval
            succ_rows = load_rows(args.success_data)
            print(f"success-data: {len(succ_rows)} rows ({args.success_data})", flush=True)
            # 身份改写要打分而不是只判 0/5：槽位从训练数据的追加段自动抽（微调者/版本/项目）。
            # 纯后处理，不多发一次 GEN；`diverged_at` 需要真实 θ₀ 基线，交给
            # `.workbuddy/tmp/idrun/eval_identity_graded.py` 离线跑。
            if str(getattr(args, "data", "")) == "identity" and getattr(args, "word", ""):
                import identity_score
                graded_slots = identity_score.slots_from_triples(IDENTITY_ROWS, args.word)
                print(f"success graded: slots {graded_slots}", flush=True)
        # eval_step 是另一个方法，槽位挂在实例上（None = 不产出 graded 读数）
        self.graded_slots = graded_slots

        t_train0 = time.time()
        # 一轮 = 过一遍数据。grpo/luspo 一步一组（batch_at 覆写），所以一轮 = items/group-size 步。
        # 与 --epochs 换算共用 steps_per_epoch()，两处口径不许各算一遍（曾经就是这里对、上面那份错）。
        per_epoch = self.steps_per_epoch()
        best_m, stall = None, 0
        self.sink.start(algo=self.objective, method=getattr(args, "method", "mezo"),
                        data=str(getattr(args, "conv_data", "") or args.data or args.dpo_data
                                 or args.rollouts),
                        items=len(self.items), rounds=round(args.steps / per_epoch, 3),
                        steps=args.steps, steps_per_epoch=per_epoch, lr=args.lr,
                        dirs=args.dirs, batch_rows=args.batch_rows, eps=args.eps,
                        rank=args.rank, seed=args.seed,
                        lp0=self.lp0, ce0=self.ce0,
                        theta_norm0=theta_norms(theta, layers, ("a", "b"))[0],
                        theta_norm_train0=theta_norms(theta, layers,
                                                      self.trainable_facs())[1])
        round_loss: list[float] = []      # 本轮的每步损失（画轮次曲线用）
        # `--zo-block K`：在上面的层集里**每步只动 K 层**、按步轮转。层集解决「能不能改到」，
        # 块坐标解决「这一步的方向估得准不准」—— 两者分别是 d 的位置和 d 的大小，分开调。
        _blk = int(getattr(args, "zo_block", 0) or 0)
        if _blk > 0 and _blk < len(layers):
            _rounds = (len(layers) + _blk - 1) // _blk
            print(f"zo-block: 每步动 {_blk}/{len(layers)} 层，{_rounds} 步轮完一轮；"
                  f"每步 d = {_blk / len(layers):.3f} × 全层，"
                  f"每层位移上限 = step-max/√{_blk}"
                  f"（动量在 {_rounds} 步后衰减到 {args.momentum ** _rounds:.3f}）", flush=True)
        # `--theta-max-rel`：‖θ‖ 的停机上限。单步上限管不住总量（step_max 钉死时 θ 匀速游走），
        # 而「训坏」的唯一可观测前兆就是 ‖θ‖ 的涨幅 —— v6 在 2.07× 处生成已进复读环、
        # 2.96× 处保持集从 44/48 掉到 3/48，期间 loss 曲线一路在降。
        _g_rel = float(getattr(args, "theta_max_rel", 0.0) or 0.0)
        # θ₀ 的范数总是算（一次 7 MB pass）：停机闸与 `--theta-avg` 的「省掉多少位移」都要它。
        _th0 = theta_norms(theta, layers, ("a", "b"))[0]
        if _g_rel > 0:
            print(f"theta-max: |theta| 起点 {_th0:.3f}，上限 {_g_rel:.2f}× = {_th0 * _g_rel:.3f}"
                  f"（到点即停，产物仍是 final.gguf）", flush=True)
        # `--theta-avg`：adapter 的指数滑动平均。**只改产物，不改轨迹** —— 每一步照旧用 θ_t
        # 前向，平均的是「拿哪个点去生成」。零阶的每步位移方向都带噪声，θ_T 里混着两部分：
        # 持续漂移（方向一致、贡献 logp 进展）和随机游走（方向随机、只贡献 ‖θ‖ 却决定伤害）。
        # 平均对漂移近似不动、对随机游走按 √窗口 压低。第一版从**第 1 步之后**起算，
        # 不含 θ₀：否则 1/(1−BETA) 步之内会把未训练的起点混进来，读数变成「平均了但没学到」。
        _avg_beta = float(getattr(args, "theta_avg", 0.0) or 0.0)
        _avg = None
        if _avg_beta > 0:
            if not 0.0 < _avg_beta < 1.0:
                raise SystemExit(f"--theta-avg 要在 (0,1) 之间（收到 {_avg_beta:g}）；0 是关")
            print(f"theta-avg: beta {_avg_beta:g}，有效窗口 {1.0 / (1.0 - _avg_beta):.0f} 步；"
                  "收尾会额外写 avg.gguf 并对它跑一次完整评测", flush=True)
        for step in range(args.steps):
            t0 = time.time()
            batch = self.batch_at(step)
            lr_now = self.lr_at(step)
            sub = guard.block_window(layers, step, _blk) if _blk > 0 else layers
            pj_vals, norm, rel, loss_mid, mov = self._zo_step(
                theta, vel, rng, batch, sub, lr_now,
                slot=(step % len(self.items)) if len(batch) == 1 else None)
            if loss_mid is not None:
                round_loss.append(loss_mid)
            if _avg_beta > 0:                      # --theta-avg：逐步 EMA（θ 已就地更新完）
                if _avg is None:
                    _avg = theta_ema_init(theta, layers)
                else:
                    theta_ema_update(_avg, theta, layers, _avg_beta)

            ms = (time.time() - t0) * 1000.0
            # 相对更新量 = ‖lr·v‖ / ‖θ_能动‖。这条比 lr 本身有判据价值：绝对步长要看参数
            # 范数，而 lr 与 loss 口径是耦合的（sum 口径下 pj ∝ T）。经验线：>0.05/步 必然
            # 发散（2026-11-01：128K sum 口径两步 ‖θ‖ 从 736 → 1419，rel ≈ 0.4）。
            # 分母只算参与更新的因子：--zo-factors b 下 A 冻结在初始化范数上，用它当分母
            # 会把 rel 压小约 3 个数量级，守卫永远不响（2026-11-14 GRPO 复盘）。
            pj_arr = np.array(pj_vals)
            pj_disp = f"{pj_arr.mean():+.3e}" if len(pj_arr) == 1 else \
                f"{pj_arr.mean():+.3e}±{pj_arr.std():.2e}"
            # 给了绝对上限（--zo-step-max）时，rel 会失真：‖b‖ 还在 0.01 量级时，
            # 步长 0.01 就是 rel=1.0，守门的是绝对上限本身。此时不再报 rel 发散。
            _sm = float(getattr(args, "zo_step_max", 0.0) or 0.0)
            warn = ("  ** 发散风险：rel>0.05，调小 --lr 或确认 --loss-reduction mean **"
                    if (rel > 0.05 and _sm <= 0) else "")
            rel_disp = "  n/a" if rel != rel else f"{rel:.2e}"
            print(f"step {step:3d}: pj {pj_disp}  lr {lr_now:.2e}  |theta| {norm:.3f}"
                  f"  |b| {mov:.4f}  rel {rel_disp}  {ms:6.0f} ms{warn}", flush=True)
            if getattr(self, "_prof", False) and (step + 1) % 200 == 0:
                _a = self._prof_acc
                _n = max(1, step + 1)
                # `loss_at()` 内部就走 swap（写盘 + 热加载），所以前向计时把它们包在里面：
                # 减掉才是纯 SCORE 的耗时，否则三段相加会超过总步时。
                _fwd = _a["forward"] - _a["write"] - _a["lora"]
                _fwd_n = max(1, _a["n"])
                print(f"  profile/{_n}: 写盘 {1000*_a['write']/_n:.0f} ms/步"
                      f" · 热加载 {1000*_a['lora']/_n:.0f} ms/步"
                      f" · 前向 {1000*_fwd/_fwd_n:.1f} ms/次 ({_a['n']/_n:.1f} 次/步"
                      f" = {1000*_fwd/_n:.0f} ms/步)"
                      f" · 其余 {ms - 1000*(_a['write']+_a['lora']+_fwd)/_n:.0f} ms/步",
                      flush=True)
            self.sink.step(step=step, round=step // per_epoch, loss=loss_mid,
                           pj=float(pj_arr.mean()), pj_std=float(pj_arr.std()) if len(pj_arr) > 1 else None,
                           rel=rel, theta_norm=norm, theta_norm_train=mov,
                           lr=lr_now, ms=round(ms, 1),
                           tok=int(sum(len(it.get("seq") or it.get("c_seq") or []) for it in batch)))

            if theta_guard_exceeded(norm, _th0, _g_rel):
                print(f"theta-max: |theta| {norm:.3f} > {_g_rel:.2f}×{_th0:.3f}"
                      f"（{norm / max(_th0, 1e-9):.2f}×）⇒ 第 {step + 1} 步停训。"
                      f"再往后就是 v6 那条路（生成进复读环、保持集崩）", flush=True)
                self.sink.end(steps_done=step + 1, wall_s=round(time.time() - t_train0, 1),
                              reason="theta-max")
                break

            if args.eval_every and (step + 1) % args.eval_every == 0:
                report.eval_step(self, succ_rows, t_train0, step=step,
                                 rnd=step // per_epoch + 1)
                if getattr(args, "ckpt_every_eval", False):
                    _ck = os.path.join(args.work, f"ckpt_{step + 1}.gguf")
                    swap(theta, _ck)
                    print(f"  ckpt: {_ck}", flush=True)

            # 每轮收尾：无条件是「记录一轮」（RSI 控制器按轮决策，曲线也要按轮画），
            # 早停只是在这个记录之上的附加判据（用户口径 2026-11-04：崩之前就停）。
            if (step + 1) % per_epoch == 0:
                rnd = (step + 1) // per_epoch
                rme = max(1, int(getattr(args, "round_metric_every", 1) or 1))
                if rnd % rme != 0:
                    # 只跳过「全量 CE 评测」，不跳过训练。round_loss 留着，下一次全量
                    # 评测记录的是整段 N 轮的均值，避免曲线丢掉中间信息。
                    continue
                swap(theta, self.final_path)
                m = self.answer_ce()
                if m is None:
                    m = -name_logprob(eng, tok, self.eval_render, args.probe, args.word)
                r_mean = float(np.mean(round_loss)) if round_loss else None
                if best_m is None or m < best_m - args.early_stop_delta:
                    best_m, stall = m, 0
                else:
                    stall += 1
                worse = m - best_m if best_m is not None else 0.0
                print(f"  round {rnd:2d}: metric {m:.4f}  best {best_m:.4f}"
                      f"  loss {r_mean if r_mean is not None else float('nan'):.4f}"
                      f"  stall {stall}", flush=True)
                self.sink.rnd(round=rnd, step=step, loss=r_mean, metric=m, best=best_m,
                              stall=stall, lr=lr_now, steps=len(round_loss),
                              wall_s=round(time.time() - t_train0, 1))
                round_loss.clear()
                if not args.early_stop:
                    continue
                if args.early_stop_worsen > 0 and worse > args.early_stop_worsen:
                    print(f"early stop: 第 {rnd} 轮比最好点恶化 {worse:+.4f}"
                          f" > {args.early_stop_worsen}（崩了）", flush=True)
                    self.sink.end(steps_done=step + 1, wall_s=round(time.time() - t_train0, 1),
                                  reason="worsen")
                    break
                if stall >= args.early_stop:
                    print(f"early stop: 连续 {stall} 轮改善 < {args.early_stop_delta}"
                          f"（吃满了），停在第 {rnd} 轮", flush=True)
                    self.sink.end(steps_done=step + 1, wall_s=round(time.time() - t_train0, 1),
                                  reason="stall")
                    break

        swap(theta, self.final_path)
        lp1 = name_logprob(eng, tok, self.eval_render, args.probe, args.word)
        ce1 = self.answer_ce()
        wall = time.time() - t_train0
        msg = f"after:  logp({args.word!r} | {args.probe!r}) = {lp1:.4f}"
        if ce1 is not None:
            msg += f"  answer-ce {ce1:.4f} ({ce1 - self.ce0:+.4f})"
        msg += f"  ({lp1 - self.lp0:+.4f})"
        print(msg, flush=True)
        done = step + 1 if args.steps else 0     # 早停后按实际跑的步数记账
        print(f"trained {done}/{args.steps} steps in {wall:.0f} s"
              f" ({wall / max(done, 1):.1f} s/step)", flush=True)
        # 上限钉死的步数：这是「lr 曲线是否真的作用到了位移上」的唯一直接读数。
        # 100% 且曲线不是 const ⇒ 修好之前 lr 衰减被重归一化吃掉；修好后上限自己也随曲线缩放，
        # 仍然 100% 是正常的（此时施加位移 ∝ 曲线），所以这里只报数、不告警。
        _n, _c = getattr(self, "_zo_step_n", 0), getattr(self, "_zo_clamped_n", 0)
        if _n and _c:
            _rate = 100.0 * _c / _n
            # const 曲线下 lr/lr0 恒为 1 ⇒ 上限不缩放、位移被重归一化回同一个范数 ⇒
            # `--lr` 与 `--lr-final` 对轨迹**一点作用都没有**，|pj| 的幅值也跟着丢失。
            # 这不是「lr 生效但被裁」，是「lr 死了」，必须说清楚 —— 2026-11-22 那批
            # 结构臂全部落在这一格（arm.sh 的 --lr 1e-3 + step-max 0.03），当时日志写的
            # 「已按 lr/lr0 作用到上限上」让人以为 lr 有效，实际四个臂测的是固定步长
            # 的 normalized SGD。
            if _rate > 99.0 and args.lr_schedule == "const":
                print(f"step-max 钉死 {_c}/{_n} 步（{_rate:.0f}%）；"
                      f"**--lr 是本 run 的死参数**：上限恒为 "
                      f"{float(args.zo_step_max):g}、const 曲线不缩放，位移每步被重归一化"
                      f"回同一范数、|pj| 幅值一并丢失 ⇒ 实际更新 = 固定步长的 normalized SGD。"
                      f"要比例步长请给 --zo-step-max 0 用 --lr 控制。", flush=True)
            else:
                print(f"step-max 钉死 {_c}/{_n} 步（{_rate:.0f}%）；"
                      f"曲线 {args.lr_schedule} 已按 lr/lr0 作用到上限上", flush=True)
        # 重放省了多少，必须让它自己说出来：否则「步时变快」无法与「换了别的配置」区分。
        _rs = getattr(eng, "replay_stats", None) or {}
        if _rs.get("captures"):
            _c, _r = _rs["captures"], _rs["replays"]
            _now = _rs["capture_s"] + _rs["replay_s"]
            _would = (_c + _r) * (_rs["capture_s"] / _c)      # 全部按全读算
            print(f"replay: {_c} capture(s) {_rs['capture_s']:.0f} s + {_r} replay(s) "
                  f"{_rs['replay_s']:.0f} s = {_now:.0f} s（若每次都全读：{_would:.0f} s，"
                  f"省 {100.0 * (1.0 - _now / max(_would, 1e-9)):.0f}%）", flush=True)
        # `--theta-avg` 的产物单独评一次。θ_T 与 θ̄ 是**同一条轨迹上的两个点**，差别只在
        # 「取哪个去生成」：若 θ̄ 在更小的位移上给出同样的注入量，那说明进展是漂移、伤害来自
        # 随机游走，平均就是免费的收益；若 θ̄ 只是「稍微落后一点、指标同时变差」，那说明漂移
        # 本来就是主导，平均没有意义 —— 这一比能把两种情形分开，不需要再跑一条臂。
        if _avg is not None:
            _avgp = os.path.join(args.work, "avg.gguf")
            _tn = theta_norms(theta, layers, ("a", "b"))[0]
            _an = theta_norms(_avg, layers, ("a", "b"))[0]
            print(f"  avg: |theta| {_an:.3f}（θ_T 是 {_tn:.3f}，EMA 窗口 "
                  f"{1.0 / (1.0 - _avg_beta):.0f} 步；"
                  f"只少了 {100.0 * (1.0 - (_an - _th0) / max(_tn - _th0, 1e-9)):.0f}% 的位移）",
                  flush=True)
            report.eval_step(self, succ_rows, t_train0, step=max(done - 1, 0),
                             path=_avgp, theta=_avg)
            print(f"  avg: 产物 {_avgp}", flush=True)
        self.sink.end(steps_done=done, steps_total=args.steps, wall_s=round(wall, 1),
                      s_per_step=round(wall / max(done, 1), 2), reason="done",
                      lp1=lp1, ce1=ce1, d_lp=lp1 - self.lp0,
                      d_ce=(ce1 - self.ce0) if (ce1 is not None and self.ce0 is not None) else None)

        out = {"layers": np.array(layers),
               "alpha": np.float64(args.alpha), "rank": np.int64(args.rank)}
        for l in layers:
            for w, (a, b) in theta[l].items():
                out[f"a_{l}_{w}"] = a
                out[f"b_{l}_{w}"] = b
                if _avg is not None:            # --theta-avg：两套权重都存，事后可离线复评
                    ea, eb = _avg[l][w]
                    out[f"a_{l}_{w}_avg"] = ea
                    out[f"b_{l}_{w}_avg"] = eb
        np.savez(args.out, **out)
        print(f"saved theta: {args.out}\n"
              f"saved adapter: {self.final_path} (engine --lora / ab_identity)", flush=True)

        eng.close()
        return 0

class SftMezo(EngineMezo):
    """--algo sft：answer 段逐 token CE（objectives.sft_loss，mean 口径）。"""
    objective = "ce"

    def load_items(self) -> list[dict]:
        """ce 数据。`both`（默认）时每条样本进两遍：一遍「思考段 + `</think>` + 答案」（提示以
        `<think>` 收尾），一遍「纯答案」（提示是空思考段）。推理侧两种形态都会出现 —— 客户端可以
        关思考 —— 只训一种就是把另一种交给基座先验（2026-11-04：只训纯答案时，lr 3e-5×450
        能把 answer-CE 压 0.38，生成却仍然跳过 `xiezhipeng-git`/`strata` 那段）。"""
        conv = getattr(self.args, "conv_data", "") or ""
        if conv:
            return self.load_conv_items()
        items = []
        ws: list[float] = []
        rows = load_data(self.args.data, self.args.word, weights=ws)
        # 已由 `_resolve_think_mode` 在 __init__ 里定下（auto 按数据的详细答案判），
        # 这里只取用：评测通道 `eval_render` 用的是同一个值，两条模板不许各判一次。
        mode = self.think_mode
        thinks = template_thinks(self.render)
        if mode == "on" and not thinks:
            # 模板不收思考段（nothink 口径）时硬塞 `</think>` 会把目标变成垃圾
            raise SystemExit("--think-targets on 但模板没有思考段（提示以 </think> 收尾）；"
                             "改用 both/off，或换带 thinking 的 chat 模板")
        ident = (self.args.data or "identity").strip().startswith("identity")
        system = getattr(self.args, "system_prompt", "")
        effort = getattr(self.args, "reasoning_effort", "") or None
        if not thinks:
            off_render = self.render
        elif system:
            off_render = load_template(self.args.pack, enable_thinking=False,
                                       system=system, reasoning_effort=effort)
        else:
            off_render = load_template(self.args.pack, enable_thinking=False,
                                       reasoning_effort=effort)
        items = []
        ws: list[float] = []
        for i, (q, a) in enumerate(rows):
            iw = ws[i] if ws else None                # item 权重（jsonl 的 weight 字段）
            if thinks and mode in ("both", "on", "raw"):
                # raw：目标就是数据里的那段文本，一个字不加。`on` 会补一段合成的思考前言
                # （THINK_GEN_*），那是身份事实的载体，与「策略刚生成的这条轨迹」不是一回事：
                # 数学轨迹本身就以 <think> 后面的推理开头，再补前言等于教模型先复述一遍题面
                # （2026-11-12 实测：16 条数学错题上补前言后 loss 稳定在 ~4 nats —— 全被那段
                # 前言吃掉，答案段的 CE 只有 0.81，梯度基本没落在要学的地方）。
                # RSI 形态的在线训练必须能「按原样」回灌自己采到的轨迹，所以留这一档。
                tgt = a if mode == "raw" else with_thinking(q, a, self.args.word, ident)
                seq, plen = enc_answer(self.tok, self.render, q, tgt, eos=self.eos)
                items.append({"seq": seq, "start": plen, "iw": iw})
            if mode in ("both", "off"):
                seq, plen = enc_answer(self.tok, off_render, q, a, eos=self.eos)
                items.append({"seq": seq, "start": plen, "iw": iw})
        return items

    def load_conv_items(self) -> list[dict]:
        """`--conv-data`：整条转录一次编码，每一轮 assistant 都进 loss（多段掩码 `region`）。

        `--think-targets` 在这里不适用（轨迹自带它采样时的思考段，补或删都是改目标）；
        `--system-prompt` 只在行里没有 system 消息时补上，不覆盖轨迹自己的那一条。
        """
        system = getattr(self.args, "system_prompt", "") or ""
        effort = getattr(self.args, "reasoning_effort", "") or None
        items = []
        for r in load_conversations(self.args.conv_data):
            msgs = list(r["messages"])
            head = msgs[0] if msgs else None
            if system and not (isinstance(head, dict) and head.get("role") == "system"):
                msgs = [{"role": "system", "content": system}] + msgs
            render = load_transcript_template(self.args.pack, tools=r.get("tools"),
                                              reasoning_effort=effort)
            seq, spans = enc_transcript(self.tok, render, msgs)
            region = np.zeros(len(seq), dtype=bool)
            for s, e in spans:
                region[s:e] = True
            items.append({"seq": seq, "start": spans[0][0], "region": region,
                          "turns": len(spans)})
        n_tok = sum(len(it["seq"]) for it in items)
        n_ans = int(sum(int(it["region"].sum()) for it in items))
        print(f"conversations: {len(items)} rows, {sum(it['turns'] for it in items)} assistant turns, "
              f"{n_tok} tokens ({n_ans} supervised, {100.0 * n_ans / max(n_tok, 1):.0f}%)",
              flush=True)
        return items


    def theta0_cache(self) -> None:
        # 引擎此刻装着 θ₀（run() 在调用前刚 swap 过）：掩码必须在这时候算
        self.prepare_selection()
        # KL 锚的 θ₀ 参照：全行 logp（keep=None），任务行之外的位置都要锚住。
        # --resume 时 θ₀ = 起点已训状态 —— 锚的语义是「钉住起点已会的行为」，
        # 恰好是数据侧实验要的：起点带身份，锚住它，闲聊/任务只能在监督 token 上动。
        lam = self.anchor_weight()
        if lam > 0:
            for it in self.items:
                lp0, _ = self.eng.score(it["seq"], keep=None)
                it["lp0"] = np.asarray(lp0, dtype=np.float32)
            n_rows = int(sum(len(it["lp0"]) for it in self.items))
            print(f"theta0 cache: {len(self.items)} score calls, {n_rows} rows "
                  f"(KL anchor λ={lam:g})", flush=True)

    def answer_ce(self) -> float:
        """全部 items 的 answer 段 CE（nats/token）：loss_at 的全量版，同一份数学。
        12 条 ~35 token 的答案全量 SCORE 约 2 s，eval-every=100 时开销可忽略。
        item 权重（iw）参与平均 —— 与 loss 的加权口径一致，否则 eval 与优化目标脱钩。"""
        total, ntok = 0.0, 0.0
        for it in self.items:
            # eval 只需要 answer 段，所以同一条路上也只让引擎算这些行（读数逐位不变）
            slp, _ = self.eng.score(it["seq"], keep=engine_keep(it, None))
            seg = answer_seg(it, slp)
            iw = it.get("iw")
            iw = 1.0 if iw is None else float(iw)
            total += iw * -float(np.sum(seg))
            ntok += iw * len(seg)
        return total / max(ntok, 1.0)

    def loss_at(self, th: dict, batch: list[dict]) -> float:
        self.swap(th, self.plus_path)
        lam = self.anchor_weight()
        logps, weights = [], []
        anchor_sq = []                         # 每条：非监督行 (lp−lp0)² 的均值
        weighted = False
        for it in batch:
            sel = it.get("sel")                      # None = --select-tokens off
            keep_task = it.get("keep")               # 任务行（θ₀ 处锚定的选择掩码）
            task_off = sel is not None and not sel.any()
            if task_off and lam <= 0:
                continue                             # 起点全对且无锚：这一步没有梯度
            # 锚模式一次全行 SCORE（keep=None）：任务行按 keep_task 切出，
            # 锚行 = 补集（prompt 行 + 未选中的答案行）。两次 SCORE 会把任务行算两遍。
            slp, _ = self.eng.score(it["seq"], keep=None if lam > 0 else keep_task)
            if not task_off:
                seg = answer_seg(it, slp)
                kept = seg if sel is None else seg[sel]
                w = it.get("w")                          # --gate-weight，None = 旧口径
                iw = it.get("iw")                        # item 权重（jsonl weight），None = 1.0
                logps.append(kept)
                if w is None and (iw is None or iw == 1.0):
                    weights.append(np.ones(kept.size, dtype=np.float64))
                else:
                    weighted = True
                    base = (np.ones(kept.size, dtype=np.float64) if w is None
                            else np.asarray(w, dtype=np.float64))
                    weights.append(base * iw if iw is not None else base)
            if lam > 0:
                slp = np.asarray(slp, dtype=np.float32)   # transport 回 list；锚要按行索引
                lp0 = it.get("lp0")
                if lp0 is None:
                    raise SystemExit("--anchor-weight > 0 但 θ₀ 全行 logp 缓存缺失："
                                     "锚没有参照就是 0，pj 会静默只含任务项（伪装成锚在工作）")
                rows = (np.zeros(slp.size, dtype=bool) if keep_task is None
                        else np.asarray(keep_task, dtype=bool))
                if rows.size != slp.size:
                    raise SystemExit(f"keep {rows.size} vs logp {slp.size}：锚的行口径对不上")
                drift = slp[~rows] - lp0[~rows].astype(slp.dtype)
                if drift.size:
                    anchor_sq.append(float(np.mean(drift * drift)))
        if not logps and not anchor_sq:
            return 0.0                               # 整批都被跳过：这一步没有梯度
        loss = 0.0
        denom = 1.0
        if logps:
            cat = np.concatenate(logps)
            if not weighted:
                # 旧口径逐位不变：不要在这里换成 np.dot，求和顺序变了会动 pj 的最后几位
                loss = float(objectives.sft_loss(cat, reduction="sum"))
                denom = max(cat.size, 1)
            else:
                # 整句监督 + 闸门加权：所有选中 token 都进 loss，只把前 --gate-n 个
                # 乘上 --gate-weight。按 Σw 归一 ⇒ 标量尺度与 mean 口径一致，lr 不用重标定。
                w = np.concatenate(weights)
                loss = -float(np.dot(w, cat))
                denom = max(float(w.sum()), 1.0)
            # 口径：**默认按 token 平均**（pj 不随 T 涨）。2026-11-01 实测：73569 token
            # 的数据上 sum 口径 pj = -4.0e+04、lr 1e-5 两步 ‖θ‖ +92%（发散）；mean 口径
            # pj ~0.55，与 HF 线同量级。`sum` 保留旧语义（sum/rows）只为复现历史。
            if self.args.loss_reduction == "mean":
                loss /= denom
            else:
                loss /= len(batch)
        if anchor_sq:
            # 锚按条平均、批内再平均：与任务项的 per-token mean 同为 O(1) 标量，
            # λ 的含义不随序列长度漂移（LUSPO 分母教训：零阶的 loss 标量就是步长）。
            loss += lam * float(np.mean(anchor_sq))
        if os.environ.get("STRATA_DEBUG_LOSS"):
            msg = (f"  [loss] ce {loss:.6f}  ntok {sum(x.size for x in logps)}"
                   f"  rows {len(batch)}")
            if anchor_sq:
                msg += f"  anchor {float(np.mean(anchor_sq)):.6f}"
            print(msg, flush=True)
        return loss


class DpoMezo(EngineMezo):
    """--algo dpo：偏好对。ref 侧在 θ₀ 算一次；margin 按答案 token 归一（objectives.dpo_margin，
    长度不变量 ⇒ lr 不随序列长度重标定）。"""
    objective = "dpo"

    def load_items(self) -> list[dict]:
        if not self.args.dpo_data:
            raise SystemExit("--objective dpo 需要 --dpo-data"
                             "（偏好对 jsonl：prompt/chosen/rejected 或 messages_chosen/rejected）")
        effort = getattr(self.args, "reasoning_effort", "") or None
        items = []
        for r in load_dpo(self.args.dpo_data):
            if r.get("messages_chosen"):
                render = load_transcript_template(self.args.pack, reasoning_effort=effort)
                cs, cspans = enc_transcript(self.tok, render, r["messages_chosen"])
                rs, rspans = enc_transcript(self.tok, render, r["messages_rejected"])
                it = {"c_seq": cs, "c_start": cspans[0][0], "r_seq": rs, "r_start": rspans[0][0]}
                for side, seq, spans in (("c", cs, cspans), ("r", rs, rspans)):
                    region = np.zeros(len(seq), dtype=bool)
                    for s, e in spans:
                        region[s:e] = True
                    it[f"{side}_region"] = region
                    it[f"{side}_ntok"] = int(region[1:].sum())
                # 偏好对的两条必须共享同一个 prompt 段：模板渲染出来的前缀不一样
                # （比如一边带 system 一边不带），ref 侧减掉的就不是同一个起点分布。
                if cs[:it["c_start"]] != rs[:it["r_start"]]:
                    raise SystemExit("chosen 与 rejected 的 prompt 段不同：DPO 要求同题同前缀")
                items.append(it)
            else:
                cs, cn = enc_answer(self.tok, self.render, r["prompt"], r["chosen"],
                                    eos=self.eos)
                rs, rn = enc_answer(self.tok, self.render, r["prompt"], r["rejected"],
                                    eos=self.eos)
                items.append({"c_seq": cs, "c_start": cn, "r_seq": rs, "r_start": rn,
                              "c_ntok": len(cs) - cn, "r_ntok": len(rs) - rn})
        return items

    def _seg(self, it: dict, slp, side: str) -> np.ndarray:
        return answer_seg({"seq": it[f"{side}_seq"], "start": it[f"{side}_start"],
                           "region": it.get(f"{side}_region")}, slp)

    def theta0_cache(self) -> None:
        n = 0
        for it in self.items:
            sc, _ = self.eng.score(it["c_seq"]); sr, _ = self.eng.score(it["r_seq"])
            it["ref_c"] = float(np.sum(self._seg(it, sc, "c")))
            it["ref_r"] = float(np.sum(self._seg(it, sr, "r")))
            n += 2
        print(f"theta0 cache: {n} score calls (dpo ref)", flush=True)

    def margin_stats(self) -> tuple[float, float]:
        """当前 θ 下的 DPO margin 均值 + 为正的样本比例（report.eval_step 自动调用）。

        ref 侧是 θ₀ 处缓存的那份，所以 margin 的含义是「相对起点的偏好净增益」：
        起点必然是 0 / 50%。2026-11-16 那次 DPO 只看 ce（0.30→0.89）分不清
        「在优化但步子太大」还是「纯噪声打转」，加这一行才判得出来。
        """
        vals = []
        for it in self.items:
            sc, _ = self.eng.score(it["c_seq"])
            sr, _ = self.eng.score(it["r_seq"])
            pol_c = float(np.sum(self._seg(it, sc, "c")))
            pol_r = float(np.sum(self._seg(it, sr, "r")))
            vals.append(objectives.dpo_margin(pol_c, pol_r, it["ref_c"], it["ref_r"],
                                              it["c_ntok"], it["r_ntok"]))
        if not vals:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.mean([v > 0 for v in vals]))

    def loss_at(self, th: dict, batch: list[dict]) -> float:
        self.swap(th, self.plus_path)
        losses = []
        for it in batch:
            sc, _ = self.eng.score(it["c_seq"]); sr, _ = self.eng.score(it["r_seq"])
            pol_c = float(np.sum(self._seg(it, sc, "c")))
            pol_r = float(np.sum(self._seg(it, sr, "r")))
            margin = objectives.dpo_margin(pol_c, pol_r, it["ref_c"], it["ref_r"],
                                           it["c_ntok"], it["r_ntok"])
            # β 跟着归一化换算（objectives.dpo_beta_effective）：不换算则 KL 约束
            # 强度被除以 n̄，长序列上等于消失 —— 零阶差分信号同样会被这个尺度淹没。
            beta_eff = objectives.dpo_beta_effective(self.args.beta,
                                                     it["c_ntok"], it["r_ntok"])
            losses.append(float(objectives.dpo_loss(margin, beta_eff)))
        return float(np.mean(losses)) if losses else 0.0


class GrpoMezo(EngineMezo):
    """--algo grpo / luspo：回放轨迹。old logp 在 θ₀ 缓存；优势组内归一（一步一组）。

    `--kl-coef > 0` 时 loss 里再加 k3 形式的 KL(π_θ ‖ π_θ₀)。这里几乎白送：ref 的
    per-token logp 就是 `theta0_cache` 里那份 `it["old"]`，不需要再 SCORE 一次。
    默认 0 = 关；只约束偏移量，给不出遗忘上界。

    两者共用整条链路（数据、θ₀ 缓存、组内优势、逐条 SCORE），只有损失与分母不同：
        grpo   ratio 逐 token，分母 = token 数   （objectives.grpo_loss）
        luspo  ratio 逐序列，分母 = 序列数 G     （objectives.luspo_loss，长度无偏）
    分派写在 loss_at 里而不是拆成两个子类：链路上任何一处改动（比如以后加 TIS）
    都该同时作用于两者，拆类反而会漏。
    """
    objective = "grpo"

    def load_items(self) -> list[dict]:
        if not self.args.rollouts:
            raise SystemExit(f"--objective {self.objective} 需要 --rollouts（回放轨迹 jsonl："
                             "prompt/completion/reward；on-policy 采样走 --method backward）")
        raw = load_rollouts(self.args.rollouts)
        if len(raw) % self.args.group_size:
            raise SystemExit(f"--rollouts {len(raw)} 条不是 --group-size "
                             f"{self.args.group_size} 的整数倍（同一条 prompt 的采样必须连续存放）")
        self._check_template_match(raw)
        items = []
        effort = getattr(self.args, "reasoning_effort", "") or None
        for r in raw:
            if r.get("messages"):
                render = load_transcript_template(self.args.pack, tools=r.get("tools"),
                                                  reasoning_effort=effort)
                seq, spans = enc_transcript(self.tok, render, r["messages"])
                region = np.zeros(len(seq), dtype=bool)
                for s, e in spans:
                    region[s:e] = True
                items.append({"seq": seq, "start": spans[0][0], "region": region,
                              "nseg": int(region[1:].sum()), "reward": r["reward"],
                              "turns": len(spans), "gold": r.get("gold") or ""})
            else:
                seq, plen, lps = self._seq_from_rollout(r)
                items.append({"seq": seq, "start": plen, "nseg": len(seq) - plen,
                              "reward": r["reward"], "turns": 1, "sampler_lp": lps,
                              "gold": r.get("gold") or ""})
        if self.objective == "luspo":
            self._check_luspo_premises(items)
        self._build_fallback(items)
        return items

    def _seq_from_rollout(self, r: dict) -> tuple[list, int, np.ndarray | None]:
        """单轮轨迹的 (seq, answer 起点, 采样侧 per-token logp 或 None)。

        采样侧（`collect_rollouts.py` / `grpo.py --save-rollouts`）会把引擎的
        `prompt_ids` / `token_ids` / `token_logprobs` 一起写进 jsonl。那时**不重编码**：

          * `enc_answer` 会用自己的渲染重新出 prompt token 与 completion token。只要
            与采样的渲染差一个 token（模板版本、thinking 开关、`parse_special` 边界），
            拼出来的序列就不是引擎采样时看到的那条，而 `token_logprobs` 会整体错位 ——
            这种错位不报错，只让 ratio 的分母悄悄变成「另一条序列的 logp」。
          * 序列逐位一致时，`token_logprobs` 才敢拿来当 θ₀ 的 logp 用（`theta0_cache`
            靠它省掉逐条重打一次 SCORE；满上下文一次 SCORE 45 s，33 条就是 25 分钟）。

        判据是「`prompt_ids` 在就用它」——**以采样为准，不是以本地渲染为准**。奖励是在
        采样那条序列上得的、`token_logprobs` 也是在那条上算的，换成本地渲染就等于把
        模型放到另一段上下文里去复现一个不是它的奖励。两者不一致时只提示（实测是
        `reasoning_effort` 没传给采样侧：21 tok vs 61 tok），不挡 —— 挡掉反而丢掉
        最忠实的那条。老 rollouts 没有这些字段 ⇒ 整条退回 `enc_answer`，行为与加这段
        之前一致。
        """
        tids, pids = r.get("token_ids"), r.get("prompt_ids")
        if not tids:
            seq, plen = enc_answer(self.tok, self.render, r["prompt"], r["completion"],
                                   eos=self.eos)
            return seq, plen, None

        if pids:
            # **跟采样走**，不跟本地渲染走。奖励是在采样那条序列上得的、`token_logprobs`
            # 也是在那条上算的；换成本地渲染的 prompt 就等于把模型放到另一段上下文里
            # 去复现一个不是它的奖励。
            head = list(pids)
            mine = self.tok.encode(self.render(r["prompt"]), parse_special=True)
            if list(mine) != head:
                # 实测（2026-11-18）：采样侧没传 `reasoning_effort`，模板默认 `xhigh`
                # 会多出一整段 system（21 tok vs 61 tok）。这种差异必须看得见 ——
                # 它同时说明「采样与训练不是同一档模板」，而这会让 SCORE 的 prompt 段
                # 与采样时对不上（off-policy）。
                self._warn_once(
                    f"采样 prompt（{len(head)} tok）与本地渲染（{len(mine)} tok）不同："
                    f"训练按**采样那条**走。这通常意味着两边模板参数不一致"
                    f"（`reasoning_effort` / `enable_thinking` / system 段），"
                    f"采样时要传与训练同档，否则 SCORE 的 prompt 段与采样时对不上")
        else:
            head = self.tok.encode(self.render(r["prompt"]), parse_special=True)

        body = list(tids)
        if self.eos is not None and (not body or body[-1] != int(self.eos)):
            body.append(int(self.eos))     # 与 enc_answer 的幂等规则一致
        seq = list(head) + body
        plen = len(head)

        # logp 能与 answer 段逐位对齐的前提：eos 本来就在（截断的轨迹补出来的那一位
        # 采样时没算过，多出来一位就对不齐）。缺 eos 的轨迹本来就该被截断门处理。
        lps = r.get("token_logprobs")
        if not lps or len(lps) != len(seq) - plen:
            return seq, plen, None
        # `answer_seg` 与采样侧现在是同一份 token 口径：`ce[j]` = 答案第 j 个 token
        # （tids[j]，含首位与 eos）的 logp，`lps[i]` 也是 tids[i] 的 —— 逐位对齐，不砍位。
        # （2026-11-24 修复前这里砍掉 lps[0] 去迁就 answer_seg 漏首位的旧切片。）
        return seq, plen, np.asarray(lps, dtype=np.float32)

    def _warn_once(self, msg: str) -> None:
        """同一条警告只说一次：5000 条轨迹逐条打印会把真正的告警埋掉。"""
        seen = getattr(self, "_warned", None)
        if seen is None:
            seen = self._warned = set()
        if msg not in seen:
            seen.add(msg)
            print(f"  WARN: {msg}", flush=True)

    def _check_template_match(self, raw: list[dict]) -> None:
        """采样模板与训练模板不一致时说出来（不挡）。

        实测（2026-11-18）：`collect_rollouts.py` 原来没有 `--reasoning-effort`，而训练
        模板默认 `xhigh` 会插一整段 system 推理指令 —— 同一条题采样出 21 token 的 prompt，
        本地渲染 61 token。奖励是在前者上得的，所以 `_seq_from_rollout` 跟着采样走；
        但训练完的模型推理时用的是训练模板，两边不同档就是个必须让人看见的缝。
        旧 rollouts 没记这个字段，跳过（不猜）。
        """
        got = {r.get("reasoning_effort") for r in raw if not r.get("messages")}
        think = {r.get("enable_thinking") for r in raw if not r.get("messages")}
        if got in (set(), {None}) and think in (set(), {None}):
            return
        mine = getattr(self.args, "reasoning_effort", "") or None
        # 训练侧的 `self.render` 不传 `enable_thinking`（模板默认：留 `<think>` 开标签，让模型
        # 自己写思考段）；采样器 `--think` 默认关 ⇒ 显式传 false ⇒ 空思考段。两种上下文。
        if think == {False}:
            self._warn_once(
                "采样是 `enable_thinking=false`（空 `<think>` 段），训练模板不传这个变量"
                "（留开标签）。训练按采样那条走，但训练后的模型在「开思考」模式下见过的"
                "上下文形状不同 —— 要一致就采样加 `--think` 或训练侧也关掉思考段")
        if got not in (set(), {None}) and len(got) > 1:
            self._warn_once(f"rollouts 里 reasoning_effort 不统一（{sorted(map(str, got))}）："
                            f"同一批混了不同模板，组内优势与结果都不可比")
        elif got - {None} and next(iter(got - {None})) != mine:
            self._warn_once(
                f"采样 reasoning_effort={next(iter(got))!r}，训练模板={mine!r}："
                f"训练按采样那条走（奖励与 logp 都在它上面），但训练后的模型推理时用的是"
                f"训练模板。两边要同档才谈得上一致 —— 采样加 `--reasoning-effort`")

    def _check_luspo_premises(self, items: list[dict]) -> None:
        """跑之前把「门①：适配性」打出来（不阻止训练，但让日志自己带判据）。

        与「门②：步长」（每步打印的 `rel = ‖lr·v‖/‖θ‖`）是两道独立的门，顺序不能颠倒：
        门① 判「这个任务和这个奖励值不值得用 RL」，门② 判「这一步会不会把参数打飞」。
        rel 是纯几何量（位移/‖θ‖），不含任何「方向对不对」的假设 —— 所以它在坏任务上
        照样准，甚至更早报警（身份题 pj 比数学题还大），但它**量不出方向错**：把 lr 降到
        让 rel 过关，只等于「慢慢学错的东西」。反过来，奖励退化到全同时 pj 恒 0，
        rel 也会静默通过 —— 假阴性要由门①拦。

        门① 的三个量：
        1) 轨迹长度：损失里的 `* |y_i|` 修的是「长轨迹每 token 梯度被 1/|y_i| 稀释」。
           没有推理链的单句回答长度只有几十 token，组内长短差是噪声 ⇒ 这条曲线的
           lr 不能当 LUSPO 的默认值（2026-11-04 身份题踩过：5e-6×5 轮扫出来的数
           在论文口径 1e-6 / |y|≈32K / n=8 下没有可比性）。
        2) 组内奖励方差：全对或全错的组优势恒 0、梯度恒 0（RLVR 的前提）。
        3) 奖励是否可验证（二值）：稠密代理奖励有方差、有梯度、rel 也会报警，但优势
           指向的是代理指标（字符串重叠）而不是正确性。这是唯一一种「门① 三道全过、
           跑得很热闹、什么也没学会」的情形（2026-11-05 身份 48 条 F1 奖励实测）。
        """
        lens = [int(it["nseg"]) for it in items]
        k = max(1, self.args.group_size)
        groups = [items[i:i + k] for i in range(0, len(items) - k + 1, k)]
        deg = sum(1 for g in groups
                  if max(x["reward"] for x in g) == min(x["reward"] for x in g))
        rw = [float(it["reward"]) for it in items]
        uniq = len({round(x, 6) for x in rw})
        binary = all(x in (0.0, 1.0) for x in rw)
        print(f"luspo 前提①（适配性）：序列中位 {int(np.median(lens))} tokens"
              f"（{min(lens)}~{max(lens)}，CV {np.std(lens) / max(1e-9, np.mean(lens)):.2f}），"
              f"组内奖励全同 {deg}/{len(groups)} 组；"
              f"奖励 {uniq} 个不同值 {'（二值 0/1 = 可验证）' if binary else '（连续/稠密）'}"
              f"中位 {np.median(rw):.3f} 范围 {min(rw):.3f}~{max(rw):.3f}", flush=True)
        if np.median(lens) < 128:
            print("  ⚠ 轨迹过短：LUSPO 的长度无偏项没有作用对象（它修的是长推理链的"
                  "每 token 梯度稀释）。这里的 lr 只能当「短答任务」的取值，"
                  "不要当 LUSPO 默认值引用。", flush=True)
        if deg == len(groups):
            print("  ⚠ 所有组奖励相同 ⇒ 优势恒 0、梯度恒 0，本次训练不会产生更新"
                  "（注意：此时 rel 判据也会恒 0，**不告警不等于安全**）", flush=True)
        if not binary:
            print("  ⚠ 奖励不是可验证的 0/1：GRPO/LUSPO 的假设是「答对/答错」。稠密代理"
                  "奖励（如字符串 F1）会让优势指向措辞重叠而不是正确性 —— 实测身份题在"
                  "F1 奖励下 |pj| 中位 106（比数学题 87 还大），lr 5e-5 三步就把 ‖θ‖ "
                  "19.6→34.7。rel 判据在这里照样准（它只量位移），但它量不出方向错；"
                  "「rel 没超界」不等于「学到了」。知识类目标请走 sft/dpo。", flush=True)


    def _build_fallback(self, items: list[dict]) -> None:
        """RL 的次级算法兜底：预先找出「全错组」，把每组的次级更新材料备好。

        奖励在回放文件里是**固定的** ⇒ 哪些组全错在 load 时就知道，不用等训练中撞上
        才发现。全错组（奖励全 0）在 GRPO/LUSPO 里优势恒 0、pj 恒 0 —— 模型一条
        可模仿的正样本都没采出来，这一步原本是纯粹的空转（步时间照付、参数纹丝不动）。
        兜底 = 这一步换成 gold answer 的次级更新：

          sft  gold answer 的 answer-CE（每 token 平均，与 CE 臂同尺度）
          dpo  (chosen=gold answer, rejected=该组第一条失败轨迹) 的偏好对；
               ref 在 theta0_cache 里按 θ₀ 预打，与 DpoMezo 同口径

        全对组、或「全等但非 0」（words:N 那种整组同分的）不触发：前者模型已会，
        后者仍有稠密奖励语义，不归兜底管。
        """
        self.fb_mode = getattr(self.args, "rl_fallback", "off") or "off"
        self.fb_used = 0
        self.fb_groups = 0
        if self.fb_mode == "off":
            return
        k = max(1, self.args.group_size)
        rewards = np.asarray([it["reward"] for it in items], dtype=np.float32)
        # 判据在算法层（objectives.fallback_group_indices），与反传线共用同一份
        fails = objectives.fallback_group_indices(rewards, k)
        groups = [items[i:i + k] for i in fails]
        if not groups:
            print(f"rl-fallback({self.fb_mode})：{len(items) // k} 组里没有全错组，本次不会触发", flush=True)
            return
        for g in groups:
            gold = next((x.get("gold") for x in g if x.get("gold")), "")
            if not gold:
                raise SystemExit(
                    "--rl-fallback 需要回放里带 gold（采样任务 jsonl 的 `target` 字段，"
                    "collect_rollouts 本来就会写）。当前文件没有 —— 把 target 带上重采，"
                    "或显式 --rl-fallback off 走旧行为（全错组空转）")
            if g[0].get("region") is not None:
                self._warn_once("多轮轨迹（messages）的兜底暂不支持（prompt 头不是单一 "
                                "span，gold 序列没法与采样头对齐）：该组保持空转")
                continue
            head = list(g[0]["seq"][:g[0]["start"]])
            body = list(self.tok.encode(gold, parse_special=True))
            if self.eos is not None and (not body or body[-1] != int(self.eos)):
                body.append(int(self.eos))
            if self.fb_mode == "sft":
                gi = {"seq": head + body, "start": len(head), "nseg": len(body)}
                for it in g:
                    it["gold_item"] = gi        # 同组共享一份，SCORE 一次
            else:                               # dpo：chosen=gold，rejected=首条失败轨迹
                pair = {"c_seq": head + body, "c_start": len(head), "c_ntok": len(body),
                        "r_seq": list(g[0]["seq"]), "r_start": int(g[0]["start"]),
                        "r_ntok": int(g[0]["nseg"])}
                for it in g:
                    it["dpo_fb"] = pair
            self.fb_groups += 1
        print(f"rl-fallback({self.fb_mode})：{len(groups)} 组里 {self.fb_groups} 组全错"
              f" → 那些步换 {'gold answer 的 CE' if self.fb_mode == 'sft' else '(gold, 失败轨迹) 的 DPO'}"
              f"（其余组照常走组内优势）", flush=True)

    def _fallback_loss(self, batch: list[dict]) -> float:
        """zero_std 时的次级更新。返回值就是这一步的 loss —— 与 RL loss 同一条
        θ±ε 差分链路（swap 在 loss_at 开头已做），pj 的语义不变：次级目标的梯度。"""
        if max(float(x["reward"]) for x in batch) > 0.0:
            return 0.0        # 全对 / 全等非 0：不触发兜底
        if self.fb_mode == "off" or self.fb_groups == 0:
            return 0.0
        self.fb_used += 1
        # 计数单位是 loss 评测（零阶一步 = θ±ε 两次 loss_at），不是训练步。
        if self.fb_mode == "sft":
            ces, seen = [], set()
            for it in batch:
                gi = it.get("gold_item")
                if gi is None or id(gi) in seen:
                    continue
                seen.add(id(gi))
                slp, _ = self.eng.score(gi["seq"])
                seg = answer_seg(gi, slp)
                # 损失也走算法层那份 sft_loss（mezo/backward 同一个数）
                ces.append(float(objectives.sft_loss(seg)))   # nats/token
            val = float(np.mean(ces)) if ces else 0.0
            print(f"  [rl-fallback sft] 评测#{self.fb_used}：gold answer-CE {val:.4f}", flush=True)
            return val
        losses, seen = [], set()
        for it in batch:
            pair = it.get("dpo_fb")
            if pair is None or id(pair) in seen:
                continue
            seen.add(id(pair))
            sc, _ = self.eng.score(pair["c_seq"])
            sr, _ = self.eng.score(pair["r_seq"])
            pol_c = float(np.sum(answer_seg({"seq": pair["c_seq"], "start": pair["c_start"]}, sc)))
            pol_r = float(np.sum(answer_seg({"seq": pair["r_seq"], "start": pair["r_start"]}, sr)))
            margin = objectives.dpo_margin(pol_c, pol_r, pair["ref_c"], pair["ref_r"],
                                           pair["c_ntok"], pair["r_ntok"])
            beta_eff = objectives.dpo_beta_effective(self.args.beta,
                                                     pair["c_ntok"], pair["r_ntok"])
            losses.append(float(objectives.dpo_loss(margin, beta_eff)))
        val = float(np.mean(losses)) if losses else 0.0
        print(f"  [rl-fallback dpo] 评测#{self.fb_used}：margin-loss {val:.4f}", flush=True)
        return val

    def batch_at(self, step: int) -> list[dict]:
        # 一步一组：优势必须在组内归一，跨组的批没有意义（--batch-rows 不参与）。
        k = self.args.group_size
        g0 = (step % (len(self.items) // k)) * k
        return self.items[g0:g0 + k]

    def steps_per_epoch(self) -> int:
        # 一步一组 ⇒ 一轮 = 组数，不是条数（48 条 / group 4 = 12 步一轮）
        return max(1, len(self.items) // max(1, self.args.group_size))

    def theta0_cache(self) -> None:
        # old = ratio 的分母 = 采样时的策略。轨迹必须是 θ₀ 策略采的（通常是基座），
        # 否则重要性修的是错的量。θ₀ = 训练起点的 adapter 状态（--resume 时是已训 θ）。
        #
        # 采样侧带了 per-token logp 的轨迹**不必重打**：引擎 `LP` 行取的是「采样前、
        # 惩罚前、温度前」的 head log-softmax（src/program/generate.cpp 的 emit_lp_line），
        # 与这里 SCORE 同一口径。满上下文时这一步很贵 —— 128K 一次 45 s，33 条 25 分钟。
        #
        # 但「同口径」有三个前提，全都不能靠假设：
        #   ① 请求没带 allowed_token_ids（带了服务端按允许集重归一化，差一个 log Z）；
        #   ② 采样策略就是本轮 θ₀（--resume 时 θ₀ 是已训状态，而轨迹可能是更早采的）；
        #   ③ 序列没被重编码打乱（_seq_from_rollout 已挡，对齐不了的轨迹不带 sampler_lp）。
        # 三个前提的共同表现都是「系统性、整体同向的偏差」，与浮点噪声（1e-3 量级）差一个
        # 数量级 ⇒ 抽几条复核即可分辨。不过就整体退回全量 SCORE，并说明是哪一类。
        reuse = [i for i, it in enumerate(self.items) if it.get("sampler_lp") is not None]
        if reuse and self.args.sampler_logp != "off":
            bad = self._verify_sampler_logp(reuse)
            if not bad:
                for it in self.items:
                    lp = it.get("sampler_lp")
                    # 复核之外、没有采样 logp 的轨迹（截断的、prompt 渲染不一致的）
                    # 仍然逐条重打：混着用不影响正确性，只在日志里说清各自几条。
                    it["old"] = (np.asarray(lp, dtype=np.float32) if lp is not None
                                 else answer_seg(it, self.eng.score(it["seq"])[0]))
                k = sum(1 for it in self.items if it.get("sampler_lp") is not None)
                print(f"theta0 cache: {len(self.items) - k} score calls, "
                      f"{k} rows reused from sampler logprobs "
                      f"({min(len(reuse), self.args.sampler_logp_check)} verified)", flush=True)
                self._fallback_refs()
                return
            print(f"  WARN: 采样侧 logp 不可信 → {bad}", flush=True)
        elif reuse:
            print(f"  sampler logprobs 存在但 --sampler-logp off ⇒ 全部重打 "
                  f"({len(reuse)} 条可复用)", flush=True)

        for it in self.items:
            slp, _ = self.eng.score(it["seq"])
            it["old"] = answer_seg(it, slp)
        print(f"theta0 cache: {len(self.items)} score calls (grpo old logp)", flush=True)
        self._fallback_refs()

    def _fallback_refs(self) -> None:
        """--rl-fallback dpo：兜底偏好对的 ref 侧（θ₀ 处 chosen/rejected 的 logp 和）。
        必须在 θ₀ 缓存这一步打 —— 与 DpoMezo 的 ref 同一语义：margin 相对训练起点。"""
        if getattr(self, "fb_mode", "off") != "dpo" or not getattr(self, "fb_groups", 0):
            return
        pairs, seen = [], set()
        for it in self.items:
            p = it.get("dpo_fb")
            if p and id(p) not in seen:
                seen.add(id(p))
                pairs.append(p)
        for p in pairs:
            sc, _ = self.eng.score(p["c_seq"])
            sr, _ = self.eng.score(p["r_seq"])
            p["ref_c"] = float(np.sum(answer_seg({"seq": p["c_seq"], "start": p["c_start"]}, sc)))
            p["ref_r"] = float(np.sum(answer_seg({"seq": p["r_seq"], "start": p["r_start"]}, sr)))
        print(f"rl-fallback dpo: +{2 * len(pairs)} score calls（兜底对的 ref @ θ₀）", flush=True)

    def _verify_sampler_logp(self, idx: list[int]) -> str:
        """抽样复核采样侧 logp 与 θ₀ 处 SCORE 的差；返回空串=可信，否则返回原因。

        均匀跨整个 items 取（不是取前几条）：同一条 prompt 的 G 条轨迹几乎同分布，
        连取前 K 条等于只验了一个 prompt。均匀抽样才能让 `--resume` 那种「整体漂移」
        在每个组上都露头。
        """
        k = max(1, min(int(self.args.sampler_logp_check), len(idx)))
        pick = [idx[round(j * (len(idx) - 1) / max(1, k - 1))] for j in range(k)] if k > 1 \
            else [idx[0]]
        tol = float(self.args.sampler_logp_tol)
        worst, worst_i, tot, n = 0.0, -1, 0.0, 0
        signed, absd, tail = [], [], []
        for i in pick:
            it = self.items[i]
            ref = answer_seg(it, self.eng.score(it["seq"])[0])
            got = np.asarray(it["sampler_lp"], dtype=np.float32)
            if got.shape != ref.shape:
                return (f"形状不符（采样 {got.shape} vs SCORE {ref.shape}）")
            d = got - ref                      # 有符号：判「整体同向」还是「双向噪声」
            dd = float(np.mean(np.abs(d)))
            tot += dd
            n += 1
            signed.append(float(np.mean(d)))
            absd.append(dd)
            tail.append(float(np.mean(np.abs(d[-max(1, len(d) // 4):]))))   # 末四分之一
            if dd > worst:
                worst, worst_i = dd, i
        mean = tot / max(1, n)
        sv, av = float(np.mean(signed)), float(np.mean(absd))
        tv = float(np.mean(tail))
        # 判成因：整体同向（|均值| ≈ 绝对值均值）说明是系统性口径差（重归一化 / 换了策略）；
        # 双向抵消（|均值| ≪ 绝对值均值）说明是数值路径不同（prefill vs decode、量化累加）。
        kind = ("整体同向（系统性口径差）" if abs(sv) > 0.6 * av
                else "双向噪声（数值路径差）")
        print(f"  sampler logp 诊断：|Δ| 均值 {av:.4f}　有符号均值 {sv:+.4f}　"
              f"末四分之一 {tv:.4f}　⇒ {kind}", flush=True)
        if worst <= tol:
            print(f"  sampler logp 复核 {n} 条：|Δ| 最大 {worst:.2e}（容差 {tol:g}）"
                  f"⇒ 与 θ₀ 同口径，复用", flush=True)
            return ""
        return (f"逐 token 平均 |Δ| 最大 {worst:.3f}（容差 {tol:g}，第 {worst_i} 条，"
                f"抽样均值 {mean:.3f}，{kind}）。三类原因都会这样：请求带了 allowed_token_ids"
                f"（服务端按允许集重归一化）、--resume 时 θ₀ 与采样策略不是同一个、"
                f"轨迹是手工造的。已退回全量 SCORE")


    def loss_at(self, th: dict, batch: list[dict]) -> float:
        self.swap(th, self.plus_path)
        rewards = np.array([it["reward"] for it in batch], dtype=np.float32)
        adv, _used, zero_std = objectives.group_advantages(rewards, self.args.group_size)
        if zero_std:
            # 这组全对/全错：优势恒 0，没有可学方向。全错时模型连一个可模仿的正样本
            # 都没采出来 —— 按次级算法兜底（gold answer 的 sft/dpo），不再空转。
            return self._fallback_loss(batch)
        cfg = objectives.GrpoCfg(clip_low=self.args.clip, clip_high=self.args.clip)
        total_tok = sum(int(it["nseg"]) for it in batch)
        # 分母是这两条算法唯一的实质差别：grpo 除 token 数（每个 token 等权），
        # luspo 除序列数 G（每条轨迹等权、序列内 token 求和）—— 后者就是论文里
        # 乘 |y_i| 的那一步，梯度里的 1/|y_i| 因此被抵消（见 objectives.luspo_loss）。
        is_luspo = self.objective == "luspo"
        loss_fn = objectives.luspo_loss if is_luspo else objectives.grpo_loss
        # 两条算法的分母**都取 token 数**。luspo 的论文口径是「除序列数 G」（= 乘 |y_i|），
        # 但那会让 loss 的**标量**跟着平均序列长度 T 一起放大，而零阶线是纯 SGD ——
        # loss 的标量直接就是步长。2026-11-14 同一份数据、同一 lr 5e-5 实测：
        #   grpo  pj 1e-2…5e-1    ，78 步 ‖b‖ 0 → 0.071（权重相对位移 ~1e-5，实测等于没动）
        #   luspo pj 1.5e3…2.4e3  ，**一步** ‖b‖ 0 → 74.5、第 17 步 139.7，rel 1.58 ⇒ adapter 打飞
        # 比值正好是 T（本数据 3e3~1.2e4）。反传线上 Adam 把统一因子从 m/√v 里约掉，所以那边
        # 从没暴露；也正因为它没暴露，「两个目标同一个 lr」这条口径才一直显得成立。
        # 换分母只是**全局标量**（一次 batch 里 Σ_t m 是一个数），序列之间的相对权重不变
        # ⇒ 长度无偏性不受影响：|y_i| 抵消的是序列内的 1/|y_i|，那一步在 luspo_loss 里面。
        denom = float(total_tok)
        kl_coef = float(getattr(self.args, "kl_coef", 0.0) or 0.0)
        if kl_coef > 0.0 and not getattr(self, "_kl_announced", False):
            self._kl_announced = True
            print(f"kl: coef {kl_coef}  ref = θ₀ 缓存（theta0_cache 的 it['old']）"
                  f"⇒ 零额外 SCORE；罚的是 exp(r)−r−1，r = old − logp", flush=True)
        total = 0.0
        for i, it in enumerate(batch):
            slp, _ = self.eng.score(it["seq"])
            logp = answer_seg(it, slp)
            li, _st = loss_fn(logp, it["old"], adv[i],
                              np.ones_like(logp), cfg,
                              denom=denom)
            total += float(li)
            if kl_coef > 0.0:
                # ref 就是 θ₀，而它的 per-token logp 在 theta0_cache 里已经算好了 ——
                # 这里只多一次 O(T) 的 numpy，**没有额外 SCORE**（反传线要多一次完整前向）。
                # 满上下文下这一点是决定性的：128K 一次 SCORE 45 s，任何"多打一次"都不可行。
                if np.shape(it["old"]) != np.shape(logp):
                    raise SystemExit(
                        f"kl: θ₀ 缓存 {np.shape(it['old'])} 与当前 logp {np.shape(logp)} 形状不符"
                        "（theta0_cache 与 loss 的 answer 段口径必须一致）")
                kl, _kst = objectives.kl_k3(logp, it["old"], np.ones_like(logp), denom=denom)
                total += kl_coef * float(kl)
        return total


class LuspoMezo(GrpoMezo):
    """--algo luspo：链路同 grpo（回放轨迹 + θ₀ old logp + 组内优势），损失换 luspo。"""
    objective = "luspo"


TRAINER_FOR = {"ce": SftMezo, "dpo": DpoMezo, "grpo": GrpoMezo, "luspo": LuspoMezo}


def load_init_theta(args, layout: dict, init_path: str):
    """`--init-adapter` → (θ, 引擎该起的 adapter 文件)。

    两种输入都接：
      * `.gguf`：`lora.read_adapter` 直接读回 A/B；文件本身继续当引擎起点（外部输入不覆盖）。
      * `.npz` ：训练器自己的检查点（`--out` 写的那种），只含**训过的层**，未训层补 B=0 骨架；
                 引擎要 gguf，所以在 `--work` 下写一份 `init_from_<名>.gguf`。
    rank/alpha 一律以文件为准并与命令行比对：这两个只要对不上，A/B 形状或 `s=alpha/rank`
    就换了，读进去等于悄悄换了另一个 adapter。
    """
    if not os.path.isfile(init_path):
        raise SystemExit(f"--init-adapter 指向的文件不存在：{init_path}")
    if init_path.endswith(".npz"):
        z = np.load(init_path, allow_pickle=True)
        f_rank, f_alpha = int(z["rank"]), float(z["alpha"])
        if f_rank != args.rank:
            raise SystemExit(f"--init-adapter 的 rank 不匹配：{init_path} 是 rank {f_rank}，"
                             f"本次 --rank {args.rank}")
        if f_alpha != args.alpha:
            raise SystemExit(f"--init-adapter 的 alpha 不匹配：{init_path} 是 {f_alpha}，"
                             f"本次 --alpha {args.alpha}；s=alpha/rank 变了等于换了个 adapter")
        restored = {int(l): {str(w): (np.array(z[f"a_{l}_{w}"], copy=True),
                                     np.array(z[f"b_{l}_{w}"], copy=True))
                            for w in ("ssm_out", "attn_output") if f"a_{l}_{w}" in z}
                    for l in z["layers"]}
        restored = {l: dict(v) for l, v in restored.items() if v}
        theta = init_theta(layout, f_rank, args.seed)
        for l, v in restored.items():
            theta[l] = v
        # npz 是外部输入，不能就地覆盖；引擎要的 gguf 写到 --work 下，名字带上来源，方便回看
        init_path = os.path.join(args.work, "init_from_" + os.path.basename(init_path) + ".gguf")
        write_adapter(init_path, layout, theta, f_alpha)
        print(f"init theta ← {os.path.basename(args.init_adapter)}：恢复层 {sorted(restored)}，"
              f"其余 {len(layout) - len(restored)} 层 B=0；引擎起点 {init_path}", flush=True)
        return theta, init_path

    f_rank = read_adapter_rank(init_path)
    f_alpha = read_adapter_alpha(init_path)
    if f_rank != args.rank:
        raise SystemExit(f"--init-adapter 的 rank 不匹配：{init_path} 是 rank {f_rank}，"
                         f"本次 --rank {args.rank}")
    if f_alpha != args.alpha:
        raise SystemExit(f"--init-adapter 的 alpha 不匹配：{init_path} 是 {f_alpha}，"
                         f"本次 --alpha {args.alpha}；s=alpha/rank 变了等于换了个 adapter")
    theta = read_adapter(init_path, layout)
    bl = max((float(np.linalg.norm(v[1])) for d in theta.values() for v in d.values()),
             default=0.0)
    print(f"init theta ← {init_path}（{len(theta)} 层，max ‖B‖ {bl:.4g}）", flush=True)
    return theta, init_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--gguf", required=True, help="the base model's first shard (geometry + --native)")
    ap.add_argument("--engine", default=str(REPO / "build-mezo" / "strata.exe"))
    ap.add_argument("--engine-api", default="",
                    help="http://127.0.0.1:8080 等常驻 API 服务的地址：训练走它的 /v1/training/*（服务端拉起"
                         "训练引擎、推理互斥、会话关掉自动恢复），不再自己 spawn 引擎；缺省仍是本进程直起")
    ap.add_argument("--serve-url", default="",
                    help="http://127.0.0.1:8080 等常驻 API 服务的地址，但走**在线**那条：用它的 /v1/engine/score 与 "
                         "/v1/engine/lora 直接训练**驻留的推理引擎**，不卸载、不重启 ⇒ 推理不中断。要求服务端用 "
                         "--lora <初始 adapter> --no-prefill-borrow 启动；gen 不可用。"
                         "与 --engine-api 互斥")
    ap.add_argument("--prefill-chunk", default="",
                    help="prompt 的分块大小（引擎 --prefill）。留空 = 按数据里最长的样本取"
                         "（向上对齐 256，夹在 256..8192）：每条样本一块读完，预留只按真实数据大小。"
                         "这是分块大小、不是长度上限 —— 比它长的序列按块数读完；长上下文数据"
                         "（128K）显式抬它才有意义。`auto` = 让引擎自己挑（no-borrow 下它取 2048）")
    ap.add_argument("--expert-profile", default=str(REPO / "data" / "expert-profile.bin"))
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--no-adapter-check", action="store_true",
                    help="跳过开跑前的 adapter 自检。自检 = 把选中层的 B 换成随机数后 SCORE 一次，"
                         "logp 必须变；不变就说明引擎的前向没应用 LoRA（零阶差分恒为 0，"
                         "训练会跑完但什么都没发生）。代价两次 256 token 的 SCORE")
    ap.add_argument("--objective", choices=("ce", "dpo", "grpo", "luspo"), default="ce",
                    help="算法 = 目标函数（与 train.py --algo 对应）；损失数学都来自 objectives.py")
    ap.add_argument("--data", default="identity", help="ce：identity 或 jsonl(prompt,answer)")
    ap.add_argument("--conv-data", default="",
                    help="ce：多轮轨迹 jsonl（每行 messages/tools，与引擎 HTTP 层同一形状）。"
                         "整条转录按模板渲染一次，每轮 assistant 都进 loss；"
                         "工具调用轮的结构 token 因此是模板自己的特殊 token")
    ap.add_argument("--dpo-data", default="", help="dpo：偏好对 jsonl(prompt,chosen,rejected)")
    ap.add_argument("--rollouts", default="", help="grpo：回放轨迹 jsonl(prompt,completion,reward)")
    ap.add_argument("--ce-data", default="",
                    help="健康度口径：目标文本的 jsonl(prompt,answer) 或 identity；"
                         "dpo/grpo/luspo 的 eval 靠它才有 ce 一项（不然只有一个词的 logp）")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO 的 β")
    ap.add_argument("--group-size", type=int, default=4, help="GRPO：一条 prompt 的采样条数 K")
    ap.add_argument("--rl-fallback", choices=("sft", "dpo", "off"), default="sft", metavar="sft|dpo|off",
                    help="RL 的次级算法兜底（默认 sft）：某组 K 条轨迹**全错**（奖励全 0）时"
                         "优势恒 0、这一步什么也学不到 —— 模型连一个能模仿的正样本都没有。"
                         "此时这一步不再空转，换成用 gold answer（采样任务里的 `target` 字段）"
                         "做一次次级更新：sft = gold answer 的 answer-CE；dpo = "
                         "(gold=chosen, 该组第一条失败轨迹=rejected) 的偏好对。判据与损失在"
                         "算法层（objectives.fallback_group_indices / sft_loss / dpo_loss），"
                         "与 backward 线共用同一份。奖励全对或全等但非 0 的组不触发"
                         "（模型已会/有信号）。回放文件缺 target 时**报错**（不静默降级）"
                         "—— 要旧行为请显式 --rl-fallback off")
    ap.add_argument("--clip", type=float, default=0.2, help="GRPO 裁剪 ε（对称）")
    ap.add_argument("--kl-coef", type=float, default=0.0, metavar="C",
                    help="对 θ₀ 的 KL 惩罚系数（k3 估计），默认 0=关。**零阶线是零成本**："
                         "θ₀ 的 per-token logp 本来就在 theta0_cache 里（it['old']），"
                         "不需要任何额外 SCORE —— 这正是它对满上下文路线可用的原因。"
                         "它约束的是偏离量，不是遗忘上界")
    ap.add_argument("--sampler-logp", choices=("auto", "off"), default="auto",
                    help="复用 rollouts 里采样侧的 per-token logp 当 θ₀ 的 old（默认 auto，"
                         "但复核通常会挡住，见下）。引擎 LP 行取的是「采样前、惩罚前、温度前」"
                         "的 head log-softmax，与 theta0_cache 的 SCORE 同一套定义；"
                         "**实测（2026-11-18）两者逐 token 仍差 0.11 nats**：|Δ| 均值 0.1096、"
                         "有符号均值 −0.0281、末四分之一 0.0738 ⇒ 双向噪声，不是整体同向偏移，"
                         "即 decode 与 prefill 两条数值路径的差（IQ3 量化 + 累加顺序）。"
                         "0.11 会让 ratio 在**起点**就偏离 1、clip_frac 虚假活跃，无法用一个"
                         "偏移量校正 ⇒ 默认容差 0.02 会挡住并退回全量 SCORE，与加这个开关之前"
                         "行为一致。要真正省掉这轮 SCORE，前提是采样与训练走同一条数值路径；"
                         "在那之前这个开关只是把判据留在代码里，不是一个能开的省时开关")
    ap.add_argument("--sampler-logp-check", type=int, default=3, metavar="K",
                    help="复核几条（默认 3，跨整个数据集均匀取）。复核不通过就整体退回重打")
    ap.add_argument("--sampler-logp-tol", type=float, default=0.02, metavar="NATS",
                    help="复核容差：逐 token 平均 |Δ| 的上限（默认 0.02）。定这么小是因为"
                         "要挡的正是「看起来同口径、实际差 0.1」这一类；实测浮点噪声在 1e-3")
    ap.add_argument("--word", default="Strata")
    ap.add_argument("--probe", default="你是谁")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--epochs", type=float, default=0.0,
                    help="按轮次训练：steps = epochs × 样本条数（think-targets=both 时一条样本算两遍）。"
                         "给了它就忽略 --steps —— 同一步数在 12 条 / 24 条数据上是完全不同的轮数，"
                         "按轮次比按步数可比")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="LoRA 尺度（d≈3.3M）的量级；论文的 1e-6 是全参数 MeZO 的，这里要大 2~3 个数量级")
    ap.add_argument("--lr-schedule", choices=("const", "cosine", "linear"), default="const",
                    help="const = 旧行为（全程固定 --lr）；cosine/linear = 1cycle 的衰减段，从 --lr 降到 "
                         "--lr-final。上限由 --lr-finder 定（Smith & Topin, arXiv 1708.07120 的 "
                         "LR range test + super-convergence）")
    ap.add_argument("--lr-final", type=float, default=0.0,
                    help="衰减终点（--lr-schedule cosine/linear 用）。取 0 = 原文的「降到接近 0」")
    ap.add_argument("--lr-warmup", type=float, default=0.0,
                    help="上升段占 --steps 的比例（0 = 不预热，一步就到上限）。零阶没有二阶矩要热身，"
                         "默认 0；只在想复现原文 1cycle 形状时给 0.1~0.3")
    ap.add_argument("--lr-finder", action="store_true",
                    help="只做 LR range test，不训练：每档从 θ₀ 重置跑一个短窗口，输出 ΔCE/步 与 rel 后退出")
    ap.add_argument("--lr-min", type=float, default=1e-6, help="--lr-finder 扫描下限")
    ap.add_argument("--lr-max", type=float, default=3e-4, help="--lr-finder 扫描上限")
    ap.add_argument("--lr-probes", type=int, default=10, help="--lr-finder 档数（对数等距）")
    ap.add_argument("--lr-probe-steps", type=int, default=6, help="--lr-finder 每档跑几步")
    ap.add_argument("--early-stop", type=int, default=0, metavar="N",
                    help="连续 N 轮（一轮 = 过一遍数据）主指标改善 < --early-stop-delta 就停（0=关）。"
                         "主指标：ce 用 answer_ce，dpo/grpo 用 -logp(word|probe)")
    ap.add_argument("--early-stop-delta", type=float, default=0.0,
                    help="改善小于它算「没进步」。ce 建议 0.002（nats/tok），dpo/grpo 建议 0.3")
    ap.add_argument("--early-stop-worsen", type=float, default=0.0,
                    help="比历史最好点恶化超过它立刻停（崩了）。ce 建议 0.1，0=关闭。"
                         "实测 lr 2e-4 第 3-4 轮 CE 从 2.89 跳到 13.4，有它就能在崩前收手")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0, help="s = alpha/rank，加载器把它乘在 B 上")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-adapter", default="", help="缺省：生成一个 B=0 的初始 adapter 到 --work 下")
    ap.add_argument("--work", default=str(HERE / "_engine_mezo_work"))
    ap.add_argument("--out", default=str(HERE / "lora_engine_mezo.npz"))
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--ckpt-every-eval", action="store_true",
                    help="每个 --eval-every 评估点把当前 θ 存一份 ckpt_<step>.gguf 到 --work 下。"
                         "缺省关闭（历史行为：只在结束写 final.gguf，中途停=全丢）。长跑判据"
                         "「达标即停」依赖它：评估点达标时可直接用最近一份 ckpt 当 final，"
                         "代价是每个评估点多写一份 ~6.7 MiB 适配器（<1 s）。")
    ap.add_argument("--round-metric-every", type=int, default=1, metavar="N",
                    help="每 N 轮做一次全量 answer-ce 轮次评测（1=每轮，默认，历史行为逐字不变）。"
                         "一轮的全量评测要对全部 items 各发一次 SCORE；12 条数据 / 每轮 12 步时，"
                         "实测它占 wall 的 ~28%%（800 步 374 s 里约 103 s）。被跳过的轮次只累积 "
                         "round_loss，不 swap、不评测、不早停；下一次全量评测照常记录整段均值。"
                         "只影响评测频率，训练数值逐位不变。")
    ap.add_argument("--think-targets", choices=("auto", "both", "on", "off", "raw"), default="auto",
                    help="ce 目标形态。**auto（默认）= 按数据判**：详细答案里有 `<think>`/`</think>` "
                         "就训思考形态（on），没有就训纯答案（off）—— 目标里有思考段就必须开思考模板，"
                         "没有就不该让模型先写一段没人监督的自由文本（见 common/data.dataset_thinks）。"
                         "both = 每条样本进两遍（思考段+答案 / 纯答案）；on/off 只训一种。"
                         "raw = 思考段形态但不补合成前言，目标逐字等于数据文本（回灌自己采的轨迹用这一档）")
    ap.add_argument("--eval-render", choices=("auto", "off", "on"), default="auto",
                    help="评测与 logp 探针走哪条通道。auto = 跟着 --think-targets"
                         "（off → 非思考模板，both/on/raw → 思考模板，历史口径不变）。"
                         "必须与被监督的那条通道一致：思考模板的 prompt 以 `<think>` 收尾、"
                         "答案在 `</think>` 之后，用 64 token 的预算在思考模板上评测，"
                         "连思考段都写不完，`success` 会恒为 0（与学没学会无关）")
    ap.add_argument("--system-prompt", default="",
                    help="插一条 system 消息。取值 `identity` 会展开成身份事实模板 —— 事实放在系统提示"
                         "里，训练只需要学会「照着复述」，比逼模型自己记住一段稀有字符串容易得多")
    ap.add_argument("--reasoning-effort", choices=("xhigh", "medium", "low"), default="",
                    help="覆盖 chat_template.jinja 的推理风格档位。留空 = 不传变量，模板 "
                         "default('xhigh')，历史行为逐字不变。采样侧用哪档这里就用哪档："
                         "render 的 system 段必须逐字一致，否则 SCORE 算的不是采样时的序列")
    ap.add_argument("--dirs", type=int, default=1,
                    help="每步平均 q 个随机方向（方差 ~1/q，代价 q 倍；_snr.py 的 C 组说明单方向 pj 信噪比 ~0.15）")
    ap.add_argument("--zo-one-point", action="store_true",
                    help="residual-feedback 单点估计：每步 1 次前向而不是 2 次（首步多一次基线）。"
                         "偏差与方差都比两点估计大，默认关闭；开启后 loss_mid 报扰动点损失。"
                         "这是速度旋钮，不是 MeZO 原文算法，默认行为逐位不变")
    ap.add_argument("--zo-factors", choices=("b", "ab"), default="b",
                    help="零阶更新哪一族 LoRA 参数：b=只更新 B（默认；A 是随机基，B=0 起点上 "
                         "∂L/∂A 恒为 0，更新 A 只会注入噪声）；ab=两族都更新（旧行为，复现历史数字用）")
    ap.add_argument("--momentum", type=float, default=0.0,
                    help="ZO-SGDM：v = b*v + pj*z，theta -= lr*v（噪声零均值抵消，信号累加；b=0 即原版）")
    ap.add_argument("--zo-subspace", type=int, default=0, metavar="K",
                    help="结构化方向采样：0=关（MeZO 原文均匀 z，默认 —— 实验验证的唯一 4/5 路径）。"
                         "K>0 = 方向从 K 维持久随机正交基张成的子空间采（lozo 模式，对齐度 1/√k）。"
                         "2026-11-16 实测：LoZO 在本设定（zo-layers 2, d=69k）下比均匀慢"
                         "（success 2/5 vs 4/5，ce 降速一半）—— 它的收益场景是全参数微调 d~1e9。"
                         "依据：LoZO arXiv:2410.07698 (ICLR 2025)、InferZO arXiv:2605.28760")
    ap.add_argument("--zo-subspace-mode", choices=["lozo", "adaptive"], default="lozo",
                    help="子空间基的来源：lozo=随机正交基每 --zo-subspace-refresh 步懒刷新"
                         "（InferZO/LoZO 的算法，默认）；adaptive=从「loss 创新低」步收集"
                         "（2026-11-16 实验判负，仅作对照保留）")
    ap.add_argument("--zo-subspace-refresh", type=int, default=50, metavar="NU",
                    help="lozo 模式的基懒刷新间隔 ν（LoZO 论文默认 50 步）")
    ap.add_argument("--zo-subspace-noise", type=float, default=0.2, metavar="F",
                    help="--zo-subspace 开启时子空间外各向同性噪声占比（0.2 = 噪声能量为子空间分量的 4%%）")
    ap.add_argument("--zo-rel", type=float, default=0.0, metavar="RELTARGET",
                    help="每步相对步长上限（‖Δθ‖/‖θ_能动‖）。默认 0 = 关。"
                         "**它只在 --zo-factors ab 下有意义，配默认的 b 会自锁**：分子分母都 ∝ ‖B‖，"
                         "而 B 从 0 起、只能靠步长长大 ⇒ 每步被钉在 rel·‖B‖ 上不动，且不报错"
                         "（2026-11-14 idH：rel 0.01 把每步压到 1.4e-4，17 步 ‖B‖ 没长）。"
                         "所以 b 要用绝对 --zo-step-max，或改 ab（‖A‖ 是常数基线）。"
                         "为什么需要尺度：一步的位移是 lr·|pj|·‖z‖，z 是 d 维高斯、‖z‖=√d，"
                         "同一串 lr 在不同 d 下含义完全不同 —— 2026-11-18 实测 lr 1e-4 跑 400 步，"
                         "‖b‖ 0.98→326、answer-CE 4.84→9.44、生成退化成固定模板。"
                         "ab + 1e-2 时位移变成等速，累积的信号才不会被随机游走淹掉"
                         "（等价于带信赖域约束的 normalized SGD）")
    ap.add_argument("--zo-step-max", type=float, default=0.0, metavar="STEPMAX",
                    help="每步**绝对**步长上限 ‖Δθ‖（LoRA 的 B 空间单位，0=关=默认）。"
                         "**它是安全阀，不是步长控制器**：给了它就要看日志里的钉死率，"
                         "100% 意味着每步都被重归一化回同一个范数，`|pj|` 的幅值信息一并丢失 ⇒ "
                         "更新退化成固定步长的 normalized SGD，而那会**在最优点附近照样走大步**，"
                         "于是「学会目标」与「伤到其他能力」必然同步发生。"
                         "实测（2026-11-23，identity 21+12，末 2 层 rank4，d=69632，"
                         "id20 一律 256 token 预算）："
                         "step-max 0.03 钉死 2000/2000 步 ⇒ ‖θ‖ 2.40×、logp +38.5、"
                         "easy48 40/48、id20 stop 4/20；"
                         "关掉它 + `--lr 3e-5 --lr-schedule cosine --theta-max-rel 2.0` ⇒ "
                         "‖θ‖ 2.02×、logp +33.0、easy48 44/48、id20 stop 17/20，"
                         "math60 / easy48 / hard12 的配对变化全部不显著（p 0.25–1.0）。"
                         "要控制步长请用 --lr（位移 = lr·|pj|·√d，随 |pj| 自适应）。"
                         "只在已知 lr 会打飞（如 luspo 的 pj 1e3 量级）时给一个远高于常态的值当保险。")
    ap.add_argument("--theta-max-rel", type=float, default=0.0, metavar="R",
                    help="‖θ‖/‖θ₀‖ 的停机上限（0=关）。**关掉 --zo-step-max 之后这是唯一的"
                         "总位移硬门**：位移本身由 lr·|pj|·√d 决定，|pj| 会随训练涨（拟合越好"
                         "梯度越大），所以同一串 lr 在不同阶段给出的步长并不相同，"
                         "总量需要一个外生上限。"
                         "实测 identity v6 4000 步：‖θ‖ 2.82→5.85（2.07×）时生成进复读环、"
                         "保持集 48 题 44→5。**2.0 是实测的甜点，2.4 已经越过悬崖**："
                         "同一配方 ‖θ‖ 2.02× 时 id20 stop 17/20，2.42× 时只剩 2/20。"
                         "停训那一步的 θ 照样写成 final.gguf，不需要重跑。"
                         "注意它只是**止损**，不解决步长机制 —— 步长错的话在到达上限前就已经伤了。")
    ap.add_argument("--theta-avg", type=float, default=0.0, metavar="BETA",
                    help="对 adapter 做指数滑动平均（0=关），有效窗口 1/(1−BETA) 步，"
                         "产物写 <work>/avg.gguf 并在收尾时对它跑一遍完整评测。"
                         "零阶每步位移方向都带噪声，路径是「漂移 + 随机游走」的叠加：随机游走"
                         "只把 θ 推离 θ₀（伤解码器）而不贡献 logp 进展，平均能把这一半压掉而留住"
                         "漂移。它改的是**取哪个点当产物**，不改轨迹 —— 与 --anchor-weight 不同，"
                         "后者在训练中施加约束、会连注入一起压死。建议 0.99（窗口 100 步）。")
    ap.add_argument("--zo-layers", type=int, default=0, metavar="K",
                    help="只更新**末 K 层**（0=全部）。零阶单方向的信噪比是 1/√d，"
                         "d = Σ rank×(n_in+n_out)（48 层 rank8 ⇒ 3.34e6，末 2 层 rank4 ⇒ 6.97e4）。"
                         "默认（--zo-projs out）每层一个矩阵（QSA 层 attn_output / GDN 层 ssm_out），"
                         "「K 层」= K 个输出投影；--zo-projs all 时每层 3–4 个挂点，d 按挂点数放大 —— "
                         "想改中间层（事实召回所在）请用 --zo-range")
    ap.add_argument("--zo-range", default="", metavar="L0,L1[,STEP]",
                    help="只更新层号区段（含两端），可带步长：`20,34` / `20` / `12,44,4`。"
                         "与 --zo-layers 互斥。为什么需要它：输出投影只给词表加与位置无关的偏置，"
                         "而「在位置 t 说 X」要改中间层写进 KV 的事实；历史 run 全是末 1–2 层")
    ap.add_argument("--zo-projs", default="out", choices=["out", "all", "ffn", "all+ffn"],
                    metavar="out|all|ffn|all+ffn",
                    help="adapter 挂哪些投影。out（默认，历史路径）= 每层只挂输出投影"
                         "（QSA 层 attn_output / GDN 层 ssm_out）；all = 对齐 InferZO 的全模块口径"
                         "（QSA 层加 attn_q/attn_k/attn_v，GDN 层加 attn_qkv/attn_gate，即引擎适配的"
                         "全部 7 类矩阵）；ffn = 每层 shared expert 的 gate/up/down（专家 FFN 的入口，"
                         "r=4 约 1.8 M 参数）；all+ffn = 两者相加。注意：挂点多 ⇒ d 大 ⇒ 零阶信噪比"
                         "1/√d 更差，且每步的 adapter 写盘/换装开销按挂点数放大 —— 这是能力项，"
                         "不是默认更优")
    ap.add_argument("--zo-block", type=int, default=0, metavar="K",
                    help="在上面的层集里**每步只动 K 层**、按步轮转（0=每步全动）。零阶信噪比 ∝ 1/√d，"
                         "而覆盖全层又要求 d 大，块坐标把两者拆开：每步 d 小、多步轮转覆盖全层。"
                         "代价是动量的跨层连续性（K 层 ⇒ 间隔 ceil(L/K) 步，动量衰减该次方）")
    ap.add_argument("--batch-rows", type=int, default=1,
                    help="每个方向用几条样本求损失（0 = 全部；数据小就全量，损失方差再降一档）")
    ap.add_argument("--no-answer-eos", dest="answer_eos", action="store_false",
                    help="**不要用这个开关**：它只为复现 2026-11-15 之前的历史 run。默认把回合结束"
                         " token（`<|im_end|>`）接到答案末尾 —— 不接的话训练目标里从来没有「说到哪"
                         "收尾」，生成时答案末尾没有「停」这个出口，概率质量回到高概率 token 就成了"
                         " `由由由由…` / `当前当前…` / `-git-git-git…` 的自环复读；闸门要对抗的对手"
                         "也正是这个 token，它在那个位置从没被训练过，gate CE 就一直卡在 11 附近")
    ap.add_argument("--select-tokens", choices=("off", "excess", "all"), default="excess",
                    help="token 选择：excess=只有训练起点策略觉得意外（ref CE > --select-thresh）的"
                         "答案 token 进 loss，基座本来就会的 token 拿不到梯度（Rho-1 选择性建模）；"
                         "off=全答案段平均；all=**整段都进 loss**，再把 ref CE > --select-thresh 的"
                         "位置按 --span-weight 加权（必须给 --span-weight）。"
                         "excess 的问题是没选中的位置无人约束、被更新带跑；all 保住每一位的梯度，"
                         "只让难位置主导方向。与反传线 backward/sft.py 同名同默认值，只对 --objective ce 生效")
    ap.add_argument("--select-thresh", type=float, default=1.0, metavar="NATS",
                    help="excess 的阈值（nats）：ref CE 严格大于它才进 loss。1.0 ≈ 基座给该 token 的概率 < e^-1")
    ap.add_argument("--select-first", type=int, default=0, metavar="N",
                    help="每条样本只训练 excess 选中的**前 N 个** token（0=不限，默认）。"
                         "「闸门课程」：追加式数据的生成卡在追加段第一个 token 上（θ₀ CE 12.3），"
                         "它只占选中量 5%%，却决定后面 95%% 的位置在生成时能否被访问到。"
                         "2026-11-14 实测 --select-first 0（=idC）跑 3000 步：追加段均值 5.30→3.90，"
                         "12 条闸门 CE 中位数 8.70→9.08（反升），success 仍 0/5")
    ap.add_argument("--select-cluster", action="store_true",
                    help="**通用**口径：逐行减去均值后求最大子段（= 这一行里「最大的那一件事」），"
                         "再按 ref CE 补若干段外孤立高点。与 --select-mass 的区别是它先回答"
                         "「哪一片区域最值得学」，而 mass 会把预算摊到所有高 CE 的点上。"
                         "2026-11-15 实测：人工指定的 8 位 1500 步到 core 1/5，mass+fill 的 12 位"
                         "1500 步 0/5 —— 位数不是关键，「是不是同一件事」才是")
    ap.add_argument("--select-extra", type=int, default=1, metavar="N",
                    help="--select-cluster 在主簇之外补几个孤立高点（默认 1，每个 ±1 token 上下文）。"
                         "身份数据上补的正是追加段第一个 token「当前」：它是一堵孤立的高墙，"
                         "不补就永远开不了口")
    ap.add_argument("--select-mass", type=float, default=0.0, metavar="FRAC",
                    help="**通用**的 token 选择口径（0=关，默认）：逐行按 ref CE 降序累加，"
                         "只留贡献了总 surprise 这个比例的位置。它问的是「模型最不会的是哪一段」，"
                         "不问那一段是名字、数字还是代码 —— 身份数据上自动选中闸门+目标词，"
                         "数学题上自动选中出错的那一步，代码上自动选中写错的那行。"
                         "与 --select-first / --select-word / --gate-weight 互斥")
    ap.add_argument("--replay", default="auto", metavar="auto|off|LAYER",
                    help="层区间重放：adapter 只挂在层 >= L 上时，层 0..L-1 的残差是「序列+基座权重」"
                         "的纯函数，与 θ 无关 —— 同一条序列只全读一次（顺手捕获层 L 的输入残差），"
                         "之后每次 SCORE 只走层 [L, 48)。零阶一步要把同一条序列读 2×dirs 次，"
                         "所以省的正是 2×dirs-1 次的前 46 层。auto（默认）= 取训练层的最小层。"
                         "短序列同样受益（2026-11-16 撤销 n<chunk 门槛：实测 64-token 序列 "
                         "SCORE 逐位可重复）；off 关掉。首次捕获后引擎侧会自动比对一次"
                         "「全读 vs 重放」逐位，不一致就抛错，不静默降级。实测 8192 token：2.6 s -> 1.2 s")
    ap.add_argument("--select-fill", type=int, default=0, metavar="G",
                    help="--select-mass 的簇内填充：相邻两个选中位置间距 ≤ G+1 时把中间补上，"
                         "但**不向簇外扩张**（默认 0）。差异大的位置成片出现，片内会夹着模型"
                         "已经会的过渡 token（`xiezhipeng-git` 里 `eng` 的 ref CE 只有 0.09）；"
                         "用 --select-grow 覆盖它会把簇外的低 CE 前缀一起拖进来，填充不会")
    ap.add_argument("--select-grow", type=int, default=0, metavar="R",
                    help="--select-mass 的邻域扩张：每个选中位置向两侧各多吃 R 位（默认 0）。"
                         "surprise 只集中在簇的入口（目标词首 token CE 高、后面几个条件概率高），"
                         "不扩张就只学到入口")
    ap.add_argument("--select-word", default="", metavar="TOKENS",
                    help="只监督答案段里**某个目标词所在的那一段**（空=关，默认）。这是"
                         "**诊断用特例**，把「要学的那一段」人工指定死，主路径是通用的 --select-mass。"
                         "零阶一步只回一个标量 pj，25 个 token 全放进 loss 时目标词那 5 个"
                         "只占 20%%：2026-11-15 实测 1500 步 ab+课程把 answer-CE 3.29→2.67、"
                         "探针 +15.6 nats，但生成改的是「独立开发→自主研发」这类前缀同义词"
                         "——梯度全花在改前缀上了。与 --select-first 同给时取并集（闸门+目标词）")
    ap.add_argument("--span-weight", type=float, default=0.0, metavar="K",
                    help="**整句留在 loss + 目标词加权**（0=关，默认）。与 --select-first 的截断"
                         "互为反面：截断会把没监督到的位置变成 argmax 自环（2026-11-15 实测"
                         "8 位截断 ⇒ `由由由`，6 位截断 ⇒ `当前当前`），而整句平摊又让目标词只占"
                         "20%%（⇒ 事实永远不出现）。这个口径两位一体：每位都有梯度，目标词 ×K。"
                         "需要同时给 --select-word。与 --select-first / --gate-weight 互斥")
    ap.add_argument("--select-lead", type=int, default=2, metavar="N",
                    help="--select-word 时往前多吃 N 个 token（默认 2）。只 push 名字本身、不给它"
                         "正确的前置 context，等于让它在一个模型自己不会走到的位置突然出现")
    ap.add_argument("--gate-weight", type=float, default=0.0, metavar="K",
                    help="整句监督 + 闸门加权（0=关闭，默认）：所有 excess token 都留在 loss 里，"
                         "只把每条前 --gate-n 个乘上 K。与 --select-first 互斥。"
                         "2026-11-14 实测：截断（select-first 2）把闸门 CE 中位 10.87→2.82，"
                         "但第 N+1 位没有 loss，自由生成塌成「当前当前当前…」；rank 4 闸门全开"
                         "到 0.14 仍复读。要学完整追加句必须保留整句监督并加权，而不是截断")
    ap.add_argument("--gate-n", type=int, default=2, metavar="N",
                    help="--gate-weight 加权的前 N 个选中 token（默认 2 = 追加句闸门及其后一位）")
    ap.add_argument("--anchor-weight", type=float, default=0.0, metavar="LAM",
                    help="KL 锚（0=关，默认）：任务监督不到的位置（prompt 行 + 未选中的答案行）"
                         "惩罚 (lp_θ − lp_θ₀)²，θ₀ = 训练起点（--resume 时即起点已训状态）。"
                         "治「全局 logit 污染」——teacher-forced CE 只有正信号，LoRA 参数全局生效"
                         "⇒ 优化走捷径无条件抬高槽位 token（2026-11-16 三臂对照定案：conv_v1 生成"
                         "损坏全是槽位泄漏/自环，闲聊数据治标（open_contam 2→0）但同权重混训把"
                         "身份压死 0/5）。锚把「不许动」机械化，闲聊数据不再是必需品。零阶只吃标量，"
                         "锚直接加进 loss；锚行走同一次全行 SCORE 的补集，每步多付的是短序列 head"
                         "成本。判「锚在工作」：STRATA_DEBUG_LOSS=1 看 anchor 分量非零")
    ap.add_argument("--resume", action="store_true", help="从 --out 的 npz 恢复 theta 继续训")
    ap.add_argument("--loss-reduction", choices=("mean", "sum"), default="mean",
                    help="损失口径。mean=按 answer 段 token 数平均（默认，与 hf_ref.py 一致，"
                         "pj 不随 T 变大）；sum=旧语义 sum/rows（pj ∝ T，长序列上等于把 lr 乘 T，"
                         "只用于复现历史结果）")
    ap.add_argument("--success-data", default="",
                    help="jsonl；每次 eval 对这些题 GEN 并判分（_eval_hard.success_eval），出「到 100%% 的耗时」曲线")
    ap.add_argument("--success-max-new", type=int, default=512,
                    help="成功率评测的生成预算（协议口径，基线测量与训练后必须同值）")
    ap.add_argument("--metrics", default="",
                    help="训练指标 JSONL（每步/每轮一行，写完立即 flush）。RSI 控制器 tail 它即可，"
                         "不必等训练退出。留空 = 不记录（默认行为不变）")
    ap.add_argument("--metrics-port", type=int, default=0,
                    help="在本机 127.0.0.1 上起只读 HTTP 指标口：/health、/metrics?since=N、/summary。"
                         "0 = 不开。训练进程自己起线程，不额外拉服务")
    ap.add_argument("--metrics-run", default="",
                    help="指标里的 run 名（默认 <objective>-mezo），控制器按它区分并发/连续的多次训练")
    args = ap.parse_args()

    # 参数互斥先判：两条 transport 的差别是「推理要不要停」，选错了要白加载一次 tokenizer 与权重
    if args.serve_url and args.engine_api:
        raise SystemExit("--serve-url 与 --engine-api 是两条不同的 transport（在线 / 离线），只能选一个")

    os.makedirs(args.work, exist_ok=True)
    tok = load_tokenizer(args.pack)
    # system 提示在这里先展开（`identity` 是占位写法），训练与 eval 的 render 必须拿到同一份
    if args.system_prompt == "identity":
        args.system_prompt = SYSTEM_IDENTITY.format(w=args.word)
    render = load_template(args.pack, system=args.system_prompt,
                           reasoning_effort=args.reasoning_effort or None)
    geom = read_geometry(args.gguf)
    layout = adapter_layout(geom, projs=args.zo_projs)
    print(f"geometry: {geom['block_count']} layers, n_embd {geom['embedding_length']}, "
          f"qsa every {geom['full_attention_interval']}, rank {args.rank}", flush=True)

    # 参数量：每层每个挂点 r*(n_in + n_out)
    n_params = sum(args.rank * (n_in + n_out) for projs in layout.values() for n_in, n_out in projs.values())
    print(f"theta: {n_params / 1e6:.2f} M fp32 params over {len(layout)} layers", flush=True)

    init_path = args.init_adapter or os.path.join(args.work, "init.gguf")
    if args.resume and args.init_adapter:
        raise SystemExit("--resume 与 --init-adapter 互斥：前者从 --out 恢复 θ，后者从另一份 "
                         "adapter 起；同时给会出现「进程内 θ 一个、引擎起点另一个」，不猜")
    if args.resume and os.path.isfile(args.out):
        z = np.load(args.out, allow_pickle=True)
        z_rank, z_alpha = int(z["rank"]), float(z["alpha"])
        if z_rank != args.rank:
            raise SystemExit(f"--resume 的 rank 不匹配：{args.out} 是 rank {z_rank}，本次 "
                             f"--rank {args.rank}；A/B 形状不兼容，不静默继续")
        if z_alpha != args.alpha:
            raise SystemExit(f"--resume 的 alpha 不匹配：{args.out} 是 {z_alpha}，本次 "
                             f"--alpha {args.alpha}；s=alpha/rank 变了等于换了一个 adapter")
        # 按 npz 实际键名恢复**所有挂点**（不能硬编码 ssm_out/attn_output：
        # all+ffn / shexp 的挂点也在 npz 里，漏掉的挂点会让恢复后的层字典不完整，
        # write_adapter 按 layout 全挂点写就 KeyError 'attn_qkv'）。
        import re as _re
        restored = {}
        for l in z["layers"]:
            l = int(l)
            ws = {}
            for k in z.files:
                m = _re.match(rf"a_{l}_(.+)$", k)
                if not m:
                    continue
                w = m.group(1)
                if f"b_{l}_{w}" in z:
                    ws[w] = (np.array(z[f"a_{l}_{w}"], copy=True),
                             np.array(z[f"b_{l}_{w}"], copy=True))
            if ws:
                restored[l] = ws
        # npz 只存**训过的层**（--zo-layers 1 ⇒ 只剩第 47 层）。self.theta 必须是全层：
        #   * swap(self.theta, ...) 在 before / theta0_cache 时直接拿它写 final/plus.gguf，
        #     而 write_adapter 按 layout 全层写，缺层就 KeyError；
        #   * _zo_step 的 th_plus/th_minus 也只带训过的层，要靠 {**self.theta, **th} 补全。
        # 未训层沿用同 seed/rank 的初始骨架（B=0 ⇒ ΔW=0，语义就是「没动」）。
        # 层内必须 update 合并而非整体替换：npz 没存的挂点（例如那次 --zo-projs 只训了
        # 一部分）要保留 init 骨架的完整挂点集合，层字典才满足 layout。
        theta = init_theta(layout, z_rank, args.seed)
        for l, v in restored.items():
            theta[l].update(v)
        print(f"resumed theta from {args.out}: 恢复层 {sorted(restored)}，"
              f"其余 {len(layout) - len(restored)} 层保持 B=0", flush=True)
        # 引擎是拿 `--lora <init>` 起的：resume 时也必须有一份 adapter 才能起得来。
        # 之前这一支不写文件，`--work` 一换（或上一次的 work 目录被清掉）就
        # 「the engine refused to start … cannot open init.gguf」，而且 rc=None 不报原因。
        # 写恢复后的 θ 本身也是对的：θ₀ 就该是引擎启动时的状态。所以同样**总是重写**
        # （`--init-adapter` 是外部输入，不覆盖）—— 残留的旧 init 会让引擎从另一个 θ 起。
        if not args.init_adapter:
            write_adapter(init_path, layout, theta, z_alpha)
            print(f"wrote the resumed adapter as the engine's start state: {init_path}", flush=True)
    else:
        # 非 resume 分支**必须**重写 init.gguf：`--work` 目录被复用（同名 tag 重跑、
        # 复制上次的 work 目录、上次崩在中途）时，里面会留着一份**上次训练过**的 adapter。
        # 旧代码这里写的是 `if not isfile(...)`，于是引擎拿旧 adapter 起、而进程内 θ 是
        # B=0 的新骨架 —— 训练从「不是它以为的 θ₀」出发，且没有任何提示。
        if args.init_adapter:
            # `--init-adapter` 是外部输入（不覆盖它），但**θ 必须真的从它读回来**。
            # 2026-11-16 之前这里只检查文件存在、然后把文件路径交给引擎 `--lora`，
            # 进程内 θ 仍是 B=0：DPO 的 ref 落在 SFT 的 θ、policy 侧每步 swap 出去的
            # 却是「基座 + εb」，2000 步跑完 rc=0、pj 正常、‖θ‖ 正常涨，只有起点数字
            # （probe −51.69 / ce 3.19）暴露它其实在训基座。见 lora.read_adapter 的 docstring。
            theta, init_path = load_init_theta(args, layout, init_path)
        else:
            theta = init_theta(layout, args.rank, args.seed)
            stale = os.path.isfile(init_path)
            write_adapter(init_path, layout, theta, args.alpha)
            print(f"wrote the initial adapter (B=0){' over the stale one' if stale else ''}: "
                  f"{init_path}", flush=True)

    # 会由本进程（或 API 服务端）拉起的训练引擎，其 `--prefill` 是**整个会话**的 buffer 预留
    # （--no-prefill-borrow 下不借专家缓存槽），起完就改不了 ⇒ 先读数据，由最长样本定它。
    # 在线那条（--serve-url）驱动的是**已经在跑**的服务，它的 chunk 在服务端启动时定了，
    # 这里既不能改也不该为它多读一遍数据。
    trainer = TRAINER_FOR[args.objective](args, tok, render, None, theta, layout)
    longest = 0
    if not args.serve_url:
        trainer.items = trainer.load_items()
        longest = max((n for it in trainer.items for n in trainer.seq_lens(it)), default=0)
        args.prefill_chunk = resolve_prefill_chunk(longest, args.prefill_chunk)

    if args.serve_url:                       # 在线：驱动驻留引擎，推理不中断
        eng = EngineHttp(args.serve_url)
        load_s = 0.0
        print(f"online: driving the resident engine at {eng.base} (nothing is unloaded)", flush=True)
    elif args.engine_api:
        eng = EngineApi(args.engine_api, engine_argv(args, init_path))
        load_s = 0.0
    else:
        print(f"prefill chunk: {args.prefill_chunk} tokens, the longest sample ({longest} tokens); "
              f"--prefill-chunk overrides", flush=True)
        if longest > 0 and str(args.prefill_chunk).isdigit() and longest > int(args.prefill_chunk):
            n = (longest + int(args.prefill_chunk) - 1) // int(args.prefill_chunk)
            # 实测（2026-11-15，128K）：8192 → 32768 只把一条 SCORE 从 45.5 s 压到 43.5 s（−4%），
            # 而 prompt buffer 是 4×。别为了这点去开大 —— 长上下文的开销在「每个 token 都要
            # 走完 48 层」，不在分块次数。
            print(f"  note: the longest sample reads in {n} chunks of up to {args.prefill_chunk} tokens "
                  f"(not truncated)。分块再开大买不到速度：128K 上 8192→32768 只有 −4%",
                  flush=True)
        eng, load_s = launch_engine(args, os.path.join(args.work, "engine.log"), init_path)
    if not args.serve_url:
        print(f"engine up in {load_s:.0f} s (log: {args.work}/engine.log)", flush=True)

    try:
        trainer.eng = eng
        return trainer.run()
    finally:
        # 指标出口必须在这里关：run() 抛异常时也要收掉 HTTP 线程与文件句柄，
        # 否则一次失败的 RSI 轮次会留着一个还活着的监听端口，下一轮换端口/端口占用。
        trainer.sink.close()
        if isinstance(eng, EngineApi):
            eng.close()



if __name__ == "__main__":
    sys.exit(main())
