"""训练回环里的「评测与记录」—— 判据展示的唯一落点。

为什么不在 `mezo.py`：`EngineMezo.run()` 是**引擎侧训练宿主**（θ 生命周期、batch 采样、
步长调度、断点存盘、早停）。判据是另一回事。历史上每加一个指标（`success` → `graded`
→ 第三方评审）都要回宿主里改打印和 `MetricSink` 调用，于是每换一次尺子 `mezo.py` 就要动
一次 —— 那不是 MeZO 算法在变，是「报告」这件事没有自己的位置。

这里把三件事收在一处：

  取数   probe 的 `logp(word|prompt)` / `answer_ce` / 一段贪心生成；`--success-data` 的行
  判定   交给 `bench/`（`_eval_hard.success_eval` → `identity_score`），本文件不写判据
  输出   stdout 一行 + `MetricSink` 一条事件

不做的事：**不碰 θ、不碰 batch、不决定何时评测**（那是宿主的调度），也不在引擎上传
任何新端点。宿主侧只剩一行 `report.eval_step(self, ...)`。
"""

from __future__ import annotations

import time


def eval_step(trainer, succ_rows, t_train0: float, step: int | None = None,
              rnd: int | None = None) -> None:
    """一次回环内评测：probe 读数 + 可选 success/graded 判分 → 打印 + 发指标。

    `trainer` 只按用到的属性鸭子取：`args` `eng` `tok` `eval_render` `theta`
    `final_path` `swap` `answer_ce()` `lp0` `ce0` `sink`；判据相关的是
    `graded_slots`（由宿主从 `identity_score.slots_from_triples` 抽好后挂上）。
    """
    from common.encode import name_logprob
    from _eval_hard import success_eval

    args, eng, tok, render = trainer.args, trainer.eng, trainer.tok, trainer.eval_render
    trainer.swap(trainer.theta, trainer.final_path)
    lp_now = name_logprob(eng, tok, render, args.probe, args.word)
    ce_now = trainer.answer_ce()
    text = ""
    if getattr(eng, "supports_gen", True):      # 在线 transport 没有按 ids 生成的端点
        out_ids, _ = eng.gen(tok.encode(render(args.probe), parse_special=True), 64)
        text = tok.decode(out_ids) if hasattr(tok, "decode") else ""
    msg = f"  eval: logp({args.word!r}|probe) {trainer.lp0:.3f} -> {lp_now:.3f}"
    if ce_now is not None:
        msg += f"; ce {trainer.ce0:.4f} -> {ce_now:.4f} nats/tok"
    # DPO 的直接判据：训练目标的 margin 有没有涨。只看 ce 会把两件相反的事混成一件 ——
    # margin 涨而 ce 也涨是可能的（chosen 那侧的 logp 被抬高，但金标串里的其他位置被扰动），
    # 2026-11-16 那次 DPO 就是「ce 0.30→0.89、margin 未知」，只能靠猜。
    margin_stats = getattr(trainer, "margin_stats", None)
    if margin_stats is not None:
        m, frac = margin_stats()
        msg += f"; margin {m:+.3f}（{frac:.0%} 对为正）"
    msg += (f"; gen: {text[:160]!r}" if text else "")
    print(msg, flush=True)
    trainer.sink.eval(round=rnd, step=step, tag="probe", lp=lp_now, ce=ce_now,
                      lp0=trainer.lp0, ce0=trainer.ce0, gen=text[:200] if text else None)
    if succ_rows is None:
        return

    ok, total, detail = success_eval(eng, tok, render, succ_rows, args.success_max_new,
                                     slots=getattr(trainer, "graded_slots", None),
                                     max_context=int(getattr(args, "max_context", 0) or 0))
    line = f"  success: {ok}/{total} ({100.0 * ok / max(total, 1):.0f}%)"
    # clean = 内容对 **且** 干净收尾：复读垃圾只满足前者（2026-11-16 用户抓出的判据漏洞）。
    n_clean = detail.get("ok_clean")
    if n_clean is not None:
        line += f"  clean {n_clean}/{total}"
    # 收尾原因：`stop` = 模型自己写了回合结束，`length` = 撞满预算硬停。撞满就是自环复读的
    # 直接签名（训练目标里没有 eos 时 100% 是 length），而只看 ok/total 完全分不出来。
    fin = detail.get("finish")
    if fin:
        done = fin.get("stop", 0)
        line += f"  stop {done}/{total}"
        rest = {k: v for k, v in fin.items() if k != "stop"}
        if rest:
            line += " (" + " ".join(f"{k or '?'} {v}" for k, v in sorted(rest.items())) + ")"
    g = detail.get("graded_summary")
    if g:
        line += (f"  graded: progress {g['progress']:.3f}  "
                 f"core {g['core_hits']}/{g['n']}  aux {g['aux_hits']}/{g['aux_total']}")
        if g["think_only"]:
            line += f"  (think-only {g['think_only']})"
    print(f"{line}  [t+{time.time() - t_train0:.0f} s]", flush=True)
    trainer.sink.eval(round=rnd, step=step, tag="success", ok=ok, total=total,
                      wall_s=detail.get("wall_s"), rows=detail.get("rows"))
