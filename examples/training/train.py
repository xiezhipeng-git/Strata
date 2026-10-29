"""默认训练入口：一条命令就够，模型默认就是**桌面快捷方式会起的那个**。

之前每个实验都要手抄一长串路径：

    python3 sft.py --gguf /mnt/d/.../swift-...-0000{1,2}-of-00002.gguf \
        --pack /root/experts_nf4.bin --tokenizer <权重目录>/tokenizer \
        --pretrain-data /root/pretrain_blocks.npy --seqlen 2048 --ckpt ...

那段命令有三个坑：写错一个分片就静默少一层专家、`--pack` 忘了就报一句 KeyError、
`--tokenizer` 抄成引擎包里的那个目录（`AutoTokenizer` 直接 `KeyError: 'added_tokens'`，
实测）。这个文件把它们变成默认值：

    python3 train.py                        # 身份数据 + 反传 + 你钉住的模型
    python3 train.py --algo mezo            # 同一份数据，零阶，显存贴住推理
    python3 train.py --algo mezo --objective grpo --rollouts r.jsonl
    python3 train.py --data jsonl:/root/my.jsonl --steps 200 --out /root/lora.pt
    python3 train.py --list                 # 算法 / 目标函数 / 数据源一览
    python3 train.py --dry-run              # 只打印最终交给算法的命令

## 模型默认值从哪来

`launch-choice.json` 的 `"config"` 指向哪份 `strata-*.json`，就是桌面快捷方式会起的那个模型；
那份 `strata-*.json` 里有 `exe` 和完整的 `args`。这里读的就是它的 `--native` / `--pack`，
**不另写一份选择逻辑**（两处各写一份，迟早会不一致，那样「要训的」和「在跑的」就会分叉）。

没有 `launch-choice.json` 就退回「仓库根目录里 mtime 最新的 `strata-*.json`」，都没有才要求
`--gguf`。Windows 路径（`D:\\…`）在 WSL 里自动转成 `/mnt/d/…`：配置文件是 Windows 侧写的。

tokenizer 的默认值同理走约定而不是硬编码路径：**放在权重旁边**（`<gguf 目录>/tokenizer`），
因为这个目录可以从 GGUF 自己导出（见 `loader_qwen4.py tokenizer`）：
GGUF 存了词表，不存 HF 的切分规则，所以导出时要给一份同族的 `--base`。

## 和推理的关系

这个入口只 import `examples/training/` 下的东西，不碰 `serve/`、不碰引擎二进制。
反过来推理也从不 import 这里 —— 两边的唯一交界是导出的 LoRA（`lora_to_gguf.py`）。
所以**训练代码的任何改动都不会改变推理速度**，同理推理引擎的改动也不会改训练数字；
能互相影响的只有一张卡的显存/带宽（同时跑就是抢，见 README 的显存预算）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import algos

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- 模型默认值

def repo_root() -> Path:
    """本文件所在的 worktree（`examples/training/` 往上两层）。"""
    return HERE.parents[1]


def search_roots() -> list[Path]:
    """到哪里找 `strata-*.json`：本 worktree，以及同一仓库的其它 worktree。

    那些配置被 `.gitignore` 掉了，只会出现在**主工作树**里；而训练常在另一个 worktree
    里跑（例如 `Strata-train`），所以只看自己的根目录会找不到 —— 顺带看一眼 `git worktree
    list` 是唯一不写死路径的做法。`$STRATA_LAUNCH_DIR` 可以显式指定，排在第一个。
    """
    roots = [repo_root()]
    env = os.environ.get("STRATA_LAUNCH_DIR", "")
    if env:
        roots.insert(0, Path(env))
    try:
        out = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=repo_root(),
                             capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if line.startswith("worktree "):
                p = Path(line[len("worktree "):].strip())
                if p not in roots and p.is_dir():
                    roots.append(p)
    except (OSError, subprocess.SubprocessError):
        pass
    if len(roots) == 1:      # git 用不上（WSL 读不了 worktree 的 .git 文件里的 Windows 路径）
        for cand in sorted(repo_root().parent.iterdir()):
            if cand.is_dir() and cand not in roots and any(cand.glob("strata-*.json")):
                roots.append(cand)
    return roots


def read_cfg(path: Path) -> dict | None:
    """一份 `strata-*.json`：有 `exe` 和 `args` 才算有效，否则返回 None。"""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or not d.get("exe") or not d.get("args"):
        return None
    return d


def pinned_config(roots: list[Path] | None = None) -> Path | None:
    """启动器钉住的那份配置：`launch-choice.json` 优先，其次 mtime 最新的 `strata-*.json`。"""
    roots = roots or search_roots()
    for root in roots:
        choice = root / "launch-choice.json"
        if not choice.exists():
            continue
        try:
            want = json.loads(choice.read_text(encoding="utf-8")).get("config", "")
        except (OSError, ValueError):
            want = ""
        if want:
            p = Path(want)
            p = p if p.is_absolute() else root / p
            if read_cfg(p) is not None:
                return p
    best, best_mtime = None, -1.0
    for root in roots:
        for p in sorted(root.glob("strata-*.json")):
            if ".bak" in p.name or read_cfg(p) is None:
                continue
            m = p.stat().st_mtime
            if m > best_mtime:
                best, best_mtime = p, m
    return best


def to_wsl(p: str) -> str:
    """`D:\\a\\b.gguf` → `/mnt/d/a/b.gguf`（非 Windows 路径原样返回）。"""
    if os.sep == "/" and re.match(r"^[A-Za-z]:[\\/]", p):
        return f"/mnt/{p[0].lower()}/" + p[3:].replace("\\", "/")
    return p


def expand_shards(path: str) -> list[str]:
    """`…-00001-of-00002.gguf` → 同目录下的全部分片（按名字排序）。

    只给第一个分片是很容易犯的错：loader 不会报错，它只是少装后面的层。
    """
    name = os.path.basename(path)
    m = re.match(r"^(.*)-\d{5}-of-(\d{5})(\.gguf)$", name)
    if not m:
        return [path]
    pattern = os.path.join(os.path.dirname(path), f"{m.group(1)}-*-of-{m.group(2)}{m.group(3)}")
    got = sorted(glob.glob(pattern))
    return got or [path]


def _flag_value(args: list[str], flags: tuple[str, ...]) -> str:
    """args 里第一个出现的 `flag value` 的 value（支持同一个开关的多个别名）。"""
    for f in flags:
        if f in args and args.index(f) + 1 < len(args):
            return args[args.index(f) + 1]
    return ""


def launcher_model() -> dict:
    """快捷方式会起的那个模型：{name, shards, pack, tokenizer, config}；取不到返回空 dict。"""
    cfg_path = pinned_config()
    if cfg_path is None:
        return {}
    cfg = read_cfg(cfg_path) or {}
    args = [str(a) for a in cfg.get("args", [])]
    native = _flag_value(args, ("--native", "-m", "--model", "--gguf"))
    if not native:
        return {}
    # 注意：配置里的 `--pack` 是**引擎**的 pack 目录（1.5 GB 的 dense/索引），不是训练的
    # NF4 专家 pack（60 GB 的 experts_nf4.bin）。两个都叫 "pack"，所以这里不取它。
    return {"name": cfg.get("model_name") or Path(native).stem,
            "config": str(cfg_path),
            "shards": [to_wsl(p) for p in expand_shards(to_wsl(native))],
            "tokenizer": to_wsl(str(cfg.get("tokenizer") or ""))}


def default_pack(shards: list[str]) -> str:
    """`<权重目录>/experts_nf4.bin`：pack 跟着权重放（`convert` 的默认输出位置也是这里）。"""
    return str(Path(shards[0]).parent / "experts_nf4.bin") if shards else ""


def resolve_tokenizer(a, info: dict, shards: list[str]) -> tuple[str, str]:
    """(目录, 来源说明)：命令行 > $STRATA_TOKENIZER > 权重旁边 > 启动器配置。

    引擎包里的 `tokenizer/` 不是 HF 布局（`AutoTokenizer` 直接 `KeyError: 'added_tokens'`，
    实测），所以每个候选都要先探一下能不能加载，不能就跳过而不是带病往下传。
    """
    for cand, src in ((a.tokenizer, "--tokenizer"),
                      (os.environ.get("STRATA_TOKENIZER", ""), "$STRATA_TOKENIZER"),
                      (str(Path(shards[0]).parent / "tokenizer"), "权重目录旁的 tokenizer/"),
                      (info.get("tokenizer", ""), "启动器配置")):
        if cand and _tokenizer_ok(cand):
            return cand, src
    out = Path(shards[0]).parent / "tokenizer"
    raise SystemExit(
        "没有可用的 tokenizer：候选（命令行 / $STRATA_TOKENIZER / 权重旁 / 启动器配置）都读不了。\n"
        "  GGUF 里存的是词表，不是 HF 的切分规则，先导一份出来：\n"
        f"    python3 loader_qwen4.py tokenizer {shards[0]} --base <同族 HF tokenizer 目录> --out {out}\n"
        "  之后不用再带 --tokenizer（默认就找权重旁边的这一份）。")


def resolve_model(a) -> tuple[list[str], str, str, str]:
    """(gguf 分片, pack, tokenizer, 来源说明)。命令行 > 启动器钉住的那份配置。"""
    info = {} if a.gguf else launcher_model()
    shards = [to_wsl(p) for p in a.gguf] if a.gguf else info.get("shards", [])
    if not shards:
        raise SystemExit(
            "没有模型：命令行没给 --gguf，仓库根目录也没有可用的 strata-*.json。\n"
            "  要么带 --gguf <分片…>，要么在 Windows 上双击一次 Strata CN.lnk 让配置生成/钉住。")
    origin = "--gguf" if a.gguf else f"启动器钉住的 {info.get('name')}（{Path(info['config']).name}）"

    pack = a.pack or os.environ.get("STRATA_PACK", "") or default_pack(shards)
    if not os.path.exists(pack):
        raise SystemExit(
            f"没有 NF4 专家 pack：{pack}（可以用 --pack 或 $STRATA_PACK 指一个）。\n"
            "  先转一次（约 10~20 分钟，之后训练零读盘）：\n"
            f"    python3 loader_qwen4.py convert {' '.join(shards)} --pack-out {pack}")
    if not os.path.exists(pack + ".json"):
        raise SystemExit(f"{pack}.json 缺失（loader 靠它拿每层专家偏移），pack 不完整？")

    tok, tok_from = resolve_tokenizer(a, info, shards)
    return shards, pack, tok, f"{origin}；tokenizer 来自 {tok_from}"


def _tokenizer_ok(path: str) -> bool:
    """这个目录能不能被 `AutoTokenizer` 直接加载（引擎包那份不能，见 resolve_tokenizer）。"""
    if not os.path.isdir(path):
        return False
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- 命令拼装

def build_argv(a, shards: list[str], pack: str, tok: str) -> list[str]:
    algo = algos.ALGOS[a.algo]
    if a.algo == "grpo":
        # 两段式：学生是 HF/4bit 目录（grpo.load_student 走 loader.py），不是 GGUF+pack
        if not a.model:
            raise SystemExit("--algo grpo 需要 --model <HF 权重目录>（它的学生走 loader.py，"
                             "125B 的 GGUF+NF4 路径还没接过去）；125B 上的 RL 用 "
                             "--algo mezo --objective grpo")
        argv = ["--model", a.model, "--steps", str(a.steps), "--lr", str(a.lr),
                "--out", a.out]
        return argv + list(a.rest)

    argv = ["--gguf", *shards, "--pack", pack, "--tokenizer", tok]
    argv += _data_flags(a)
    argv += ["--steps", str(a.steps), "--lr", str(a.lr), "--out", a.out,
             "--seed", str(a.seed)]
    if a.algo == "sft":
        argv += ["--accum", "1"]
    if a.objective and a.objective != "ce":
        if a.objective not in algo.objectives:
            raise SystemExit(f"--algo {a.algo} 不支持 --objective {a.objective}；"
                             f"可选 {algo.objectives}")
        argv += ["--objective", a.objective]
        if a.objective == "grpo":
            if not a.rollouts:
                raise SystemExit("--objective grpo 需要 --rollouts <轨迹 jsonl>"
                                 "（用 collect_rollouts.py 采，或 grpo.py --no-student）")
            argv += ["--rollouts", a.rollouts, "--clip", str(a.clip)]
    argv += list(algo.extra)
    return argv + list(a.rest)


def _data_flags(a) -> list[str]:
    """`--data` → 该算法认识的参数（identity/jsonl 走文本样本，pretrain 走块）。"""
    kind, arg = _resolve_data(a.data, a.word)
    if kind == "identity":
        return ["--word", arg]
    if kind == "jsonl":
        return ["--data", arg]
    return ["--pretrain-data", arg, "--seqlen", str(a.seqlen)]


def _resolve_data(spec: str, word: str) -> tuple[str, str]:
    import datasets

    kind, arg = datasets.resolve(spec)
    if kind == "identity":
        return kind, (arg or word)
    return kind, arg


# ---------------------------------------------------------------- 入口

def main() -> int:
    ap = argparse.ArgumentParser(
        description="默认训练入口：模型取自启动器钉住的那份，数据默认「改变身份」。",
        epilog="其余参数原样转给该算法的脚本（见 --list 或 <algo>.py --help）。")
    ap.add_argument("--algo", default="sft", choices=algos.names(),
                    help="默认 sft（反传）。mezo=零阶，grpo=两段式 RL")
    ap.add_argument("--objective", default="ce", choices=list(algos.OBJECTIVES),
                    help="目标函数。mezo 支持 grpo ⇒ 只靠前向做 RL")
    ap.add_argument("--data", default="identity",
                    help="identity | identity:<词> | jsonl:<路径> | pretrain:<npy>（默认 identity）")
    ap.add_argument("--word", default="Strata", help="identity 数据的目标词（也是探针词）")
    ap.add_argument("--gguf", nargs="+", default=[],
                    help="模型分片；不给就从启动器配置取（= 桌面快捷方式会起的那个）")
    ap.add_argument("--model", default="", help="HF 权重目录（--algo grpo 必需）")
    ap.add_argument("--pack", default="",
                    help="NF4 专家 pack（默认 <权重目录>/experts_nf4.bin，或 $STRATA_PACK）")
    ap.add_argument("--tokenizer", default="", help="HF tokenizer 目录")
    ap.add_argument("--seqlen", type=int, default=2048, help="pretrain 块长")
    ap.add_argument("--steps", type=int, default=None, help="留给算法自己的默认值")
    ap.add_argument("--lr", type=float, default=None, help="默认按算法给（反传 1e-4 / MeZO 1e-6）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="LoRA 输出，默认 ./lora_<algo>.pt")
    ap.add_argument("--rollouts", default="", help="--objective grpo 的轨迹 jsonl")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--list", action="store_true", help="打印算法/目标函数/数据源")
    ap.add_argument("--dry-run", action="store_true", help="只打印最终命令，不加载模型")
    a, rest = ap.parse_known_args()
    a.rest = rest
    a.steps = a.steps if a.steps is not None else (20 if a.algo == "mezo" else 40)
    a.lr = a.lr if a.lr is not None else algos.ALGOS[a.algo].lr
    a.out = a.out or f"lora_{a.algo}.pt"

    if a.list:
        import datasets

        print(algos.describe())
        print("数据源（--data）：")
        for k, v in datasets.SPECS.items():
            print(f"  {k:<9s} {v}")
        return 0

    shards, pack, tok, origin = resolve_model(a)
    argv = build_argv(a, shards, pack, tok)
    kind, arg = ("rollouts", a.rollouts) if a.objective == "grpo" else _resolve_data(a.data, a.word)

    print("=" * 78, flush=True)
    print(f"algo      {a.algo}（{algos.ALGOS[a.algo].kind}）  objective {a.objective}")
    print(f"model     {origin}")
    for s in shards:
        print(f"          {s}")
    print(f"pack      {pack}")
    print(f"tokenizer {tok}")
    print(f"data      {kind}:{arg}   steps {a.steps}  lr {a.lr:g}  seed {a.seed}")
    print(f"out       {a.out}")
    print(f"cmd       python3 {algos.ALGOS[a.algo].script} " + " ".join(argv), flush=True)
    print("=" * 78, flush=True)
    if a.dry_run:
        return 0
    return algos.run(a.algo, argv)


if __name__ == "__main__":
    sys.exit(main())
