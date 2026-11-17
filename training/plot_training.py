"""画训练曲线：读 metrics JSONL（就是 RSI 控制器读的那份），出「轮次-损失」图。

为什么单独一个脚本、且只读 metrics 文件：图上的每个点必须和控制器看到的是同一个数。
如果画图时再去 parse 人类日志，就会出现「图里有 20 轮、接口只报 18 轮」这类对不上的情况。

版式是**每算法一格**（不共用 y 轴）：四条损失的尺度根本不是一回事 —— SFT 是逐 token CE、
DPO 是偏好 margin 的对数比、GRPO 是带 clip 的策略比 —— 画在同一根轴上只会得到
「谁的量纲大谁有戏」的错觉（第一版就这么画过：DPO 塌到 0 之后，SFT 的 5.1 看着像一条直线）。

用法:
  python plot_training.py --run sft=runs/sft.jsonl --run dpo=runs/dpo.jsonl \
      --run grpo=runs/grpo.jsonl --out fig-training.png
  python plot_training.py --eval evals.json --out fig-eval.png      # 只画柱状图
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from metrics import read_metrics, rounds_of  # noqa: E402

COLOR = {"sft": "#2f6fed", "dpo": "#e0682b", "grpo": "#12855c", "luspo": "#9b3fb5",
         "sft_backward": "#7a7a7a", "bwd_sft": "#7a7a7a"}
LABEL = {"sft": "SFT (zero-order)", "dpo": "DPO (zero-order)", "grpo": "GRPO (zero-order)",
         "luspo": "LUSPO (zero-order)", "sft_backward": "SFT (backprop)",
         "bwd_sft": "SFT (backprop)"}


def curves(path: str) -> dict:
    ev = read_metrics(path)
    start = next((e for e in ev if e["event"] == "start"), {})
    rs = rounds_of(path)
    steps = [(e["step"], e["loss"], e["round"]) for e in ev
             if e["event"] == "step" and e.get("loss") is not None]
    return {"rounds": [r["round"] for r in rs], "loss": [r.get("loss") for r in rs],
            "metric": [r.get("metric") for r in rs], "step_loss": steps,
            "per_epoch": start.get("steps_per_epoch") or 1,
            "end": next((e for e in reversed(ev) if e["event"] == "end"), {}),
            "start": start}


def draw_runs(specs: list[str], out: pathlib.Path) -> None:
    runs = [(s.split("=", 1)[0], curves(s.split("=", 1)[1])) for s in specs]
    live = [(n, c) for n, c in runs if c["step_loss"]]
    if not live:
        print("no step events in the given files")
        return
    fig, axes = plt.subplots(1, len(live), figsize=(4.4 * len(live), 4.0), squeeze=False)
    for ax, (name, c) in zip(axes[0], live):
        col = COLOR.get(name, "#333333")
        per = max(c["per_epoch"], 1)
        xs = [(s + 1) / per for s, _l, _r in c["step_loss"]]
        ys = [l for _s, l, _r in c["step_loss"]]
        ax.plot(xs, ys, color=col, alpha=0.35, linewidth=0.9, label="per step")
        if c["rounds"]:
            rx = [r for r in c["rounds"]]
            ax.plot(rx, c["loss"], marker="o", markersize=3.6, color=col, linewidth=1.8,
                    label="round mean")
        end = c["end"]
        ttl = LABEL.get(name, name)
        if end.get("wall_s"):
            ttl += f"\n{end.get('steps_done', '?')} steps / {end['wall_s'] / 60:.1f} min"
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("epoch (one pass over the wrong-problem set)", fontsize=8)
        ax.set_ylabel("training loss (this algorithm's own scale)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"-> {out}")


def draw_eval(path: str, out: pathlib.Path) -> None:
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    arms = d.get("arms", [])
    if not arms:
        print("no arms in eval json")
        return
    names = [x["arm"] for x in arms]
    tr = [x["train"]["ok"] / max(x["train"]["total"], 1) for x in arms]
    ho = [x["hold"]["ok"] / max(x["hold"]["total"], 1) for x in arms]
    fig, ax = plt.subplots(figsize=(2.1 * len(names) + 3.0, 4.0))
    idx = range(len(names))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in idx], tr, w, label="trained (wrong problems)", color="#2f6fed")
    b2 = ax.bar([i + w / 2 for i in idx], ho, w, label="never trained (hold-out)", color="#9aa4b2")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.0%}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(idx))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("pass rate (greedy, same 768-token budget)")
    ax.set_title("Trained vs never-trained problems", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"-> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--eval", default="")
    ap.add_argument("--out", default=str(ROOT.parent / ".workbuddy" / "tmp" / "idrun" /
                                        "fig-training.png"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if a.run:
        draw_runs(a.run, out)
    if a.eval:
        draw_eval(a.eval, out.with_name(out.stem.replace("training", "eval") + ".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
