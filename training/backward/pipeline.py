"""一条命令跑完一件事：训练 → 退出训练进程 → 起引擎 → 只换 adapter 判分 → 停引擎 → 报告。

为什么需要一个编排器，而不是两步手动命令
----------------------------------------
一台 48 GB 卡放不下训练栈和引擎：训练峰值 26.19 GiB（T=512、开 `--ckpt`）+ 引擎常驻
8.5 GiB，同时起来实测顶到 45.8 GiB，整机卡死（2026-11-17 被强关过一次）。所以规则是
**任何时刻只有一个 GPU 任务**，训练和引擎判分必须在时间上分开。

手动两步的三个问题，这里逐条消掉：
  ① 人容易在训练没退出时就起引擎（尤其训练自己 early-stop 提前结束的时候）；
  ② 训练训完往往只关心 stdout 里那一行 `pass@1`，而下一次还要重新起引擎（每次 ~20 s
     启动 + `--pack` 读盘），于是「多评估几个点」变成「多等几分钟」；
  ③ 判分结果只活在日志里。按 pass@1 选 ckpt、跟上一轮比较，都需要一份结构化记录。

职责边界
--------
本文件**不加载 torch、不碰权重、不算 loss**：它只拼命令行、看退出码、等 ready 文件。
真正的两件事分别由 `sft.py`（训练）和 `sft.py --eval-only`（引擎判分）做 —— 那两处
的口径（判据、渲染契约、adapter 导出）已经只有一份，编排器再抄一遍就会分叉。

用法
----
    python training/backward/pipeline.py \
        --engine-config engine_eval.json \
        --report runs/m1.report.json \
        -- --data math.jsonl --held-data hold.jsonl --held-gen 8 --held-max-new 512 \
           --out runs/m1.pt --epochs 1 --lr 1e-4 --rank 16 --ckpt --seqlen 768

`--` 之后是**原样透传给 `sft.py` 的训练参数**，本文件只从里面读三样东西：`--out`
（要评估哪个 ckpt）、`--held-data` / `--held-gen`（判分哪一批、抽几条）。

退出码：训练失败 → 1（不评估）；训练成功但评估失败 → 2（ckpt 还在，可单独 `--eval-only`
重跑）；都成功 → 0。
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
import subprocess
import sys
import time
import urllib.request

HERE = _p.Path(__file__).resolve().parent
TRAIN = HERE / "sft.py"


def _opt(argv: list[str], name: str, default: str = "") -> str:
    """从透传参数里取一个 `--name value` 的值（取最后一次出现，与 argparse 一致）。"""
    val = default
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            val = argv[i + 1]
        elif a.startswith(name + "="):
            val = a.split("=", 1)[1]
    return val


def _flag(argv: list[str], name: str) -> bool:
    return any(a == name or a.startswith(name + "=") for a in argv)


def _drop(argv: list[str], names: tuple[str, ...]) -> list[str]:
    """从透传参数里摘掉这几个开关（`--name value` 与 `--name=value` 两种写法都认）。

    用途只有一个：训练侧不做 held 生成。那一段是 torch decode（~1.2 tok/s），8 条 ×
    512 token 实测要 500 s 量级；引擎做同一件事是 1.2 s/条。留着它等于每次实验白付
    8 分钟的墙钟，而结果还不如引擎侧那份（decode 路径不同，见 README 的 40 s vs 1.2 s）。
    训练侧仍保留 `--held-data`：held **CE** 只要一次前向，是 keep-best 的选点依据。
    """
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        hit = next((n for n in names if a == n or a.startswith(n + "=")), None)
        if hit is None:
            out.append(a)
        elif a == hit:            # 分开写法：下一个 token 是它的值
            skip = True
    return out


def _run(cmd: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    """跑一条查询命令。**必须显式给编码**：Windows 的 `tasklist` 输出是 GBK，
    `text=True` 只表示「按平台默认编码解码」——在本机是 UTF-8 的 Python 会直接抛
    `UnicodeDecodeError`，而 `capture_output` 于是把 `stdout` 留成 None（2026-11-17
    实测就在这里崩的：`'NoneType' object has no attribute 'lower'`）。"""
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _engine_running() -> bool:
    """有没有别的进程占着 GPU —— 单任务铁律的守门人。

    只查 `strata.exe`：训练进程由本文件自己 spawn、退出码由 `run()` 保证。若外部还有
    引擎在跑，评估阶段会先是「换 adapter 换到别人的引擎上」，然后是显存不够。
    """
    if sys.platform == "win32":
        return "strata.exe" in (_run(["tasklist"]).stdout or "").lower()
    return bool((_run(["pgrep", "-f", "strata"]).stdout or "").strip())


def _kill_engine() -> None:
    if sys.platform == "win32":
        _run(["taskkill", "/F", "/IM", "strata.exe"])
    else:
        _run(["pkill", "-f", "strata"])


def _wait_ready(path: _p.Path, proc: subprocess.Popen, timeout: float) -> dict:
    """等 `serve.server --ready-file` 落盘。进程先死就立刻报错，别等到超时。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass                      # 半写状态：下一轮再读
        if proc.poll() is not None:
            raise SystemExit(f"引擎进程在 ready 前退出（rc={proc.returncode}）—— 看引擎日志")
        time.sleep(1.0)
    raise SystemExit(f"引擎 {timeout:.0f}s 内没有 ready（ready 文件 {path}）")


def _gpu_used_mib() -> int | None:
    """当前整卡已用显存（MiB）；查不到就返回 None（没有 nvidia-smi 的机器不是目标平台）。"""
    try:
        r = _run(["nvidia-smi", "--query-gpu=memory.used",
                  "--format=csv,noheader,nounits"], timeout=20)
        if r.returncode != 0:
            return None
        return int((r.stdout or "").splitlines()[0].strip())
    except (ValueError, IndexError):
        return None


def _wait_gpu_free(limit_mib: int, timeout: float) -> bool:
    """等训练进程的显存真正退回驱动。

    为什么需要这一步：进程 `exit` 和显存释放不是同一时刻 —— 解释器退出时要逐块 free
    torch 的 caching allocator、再销毁 CUDA context，实测有几十秒的窗口。手动两步时人
    会「等一下」，自动串联时没人挡，引擎就在这个窗口里 OOM。轮询到低于 `limit_mib`
    再起引擎，比加大超时可靠。
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        used = _gpu_used_mib()
        if used is None or used <= limit_mib:
            return True
        time.sleep(3.0)
    return False


def _stream(cmd: list[str], log: str, env: dict) -> int:
    """跑子进程并把 stdout+stderr 同时写屏与写文件（长时间训练要能边跑边看）。

    两处 flush 都不能省：`sft.py` 的 print 都带 `flush=True`，但**本函数这一侧**的
    文件缓冲和 stdout 缓冲仍在——不 flush 的话 `tail -f train.log` 是空的，
    而进程其实已经跑了一分多钟（2026-11-17 实测：GPU 已占 29 GiB、日志 0 字节）。
    用 `tail` 之类的管道看输出时更严重：缓冲攒到进程结束才落地，等于没有实时性。
    """
    with open(log, "w", encoding="utf-8", errors="replace", buffering=1) as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        return proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="训练 + 引擎判分，串行（任何时刻只有一个 GPU 任务）",
        epilog="`--` 之后的参数原样传给 sft.py")
    ap.add_argument("--engine-config", required=True,
                    help="serve.server 的 strata 配置 json（exe / args / tokenizer / model_name）")
    ap.add_argument("--port", type=int, default=0, help="引擎端口（0=用配置里的）")
    ap.add_argument("--report", default="", help="报告 json 路径（默认 <out>.report.json）")
    ap.add_argument("--ready-timeout", type=float, default=600.0,
                    help="等引擎 ready 的上限秒数（--pack 冷读盘可能上百秒）")
    ap.add_argument("--logdir", default="", help="子进程日志目录（默认与 ckpt 同目录）")
    ap.add_argument("--skip-eval", action="store_true",
                    help="只训练不评估（调试用；正常路径别开 —— 训练完不给判分等于没结论）")
    ap.add_argument("--keep-engine", action="store_true",
                    help="评估完不停引擎（连续评估多个 ckpt 时省一次 20 s 启动）")
    ap.add_argument("--seed-path", default="",
                    help="零 seed adapter 的路径（默认 <ckpt 目录>/seed.gguf）。**这是唯一"
                         "一份会被反复覆盖的 adapter**：它只有 B=0、数值上等于基座，"
                         "每份 ckpt 各造一份 92 MiB 的副本纯属浪费磁盘（实测一个实验目录"
                         "长到 5.7 GiB，全是 .seed.gguf 与 .stepN.pt）")
    ap.add_argument("--reuse-seed", action="store_true",
                    help="seed 已存在就直接用（同一套 --targets/--rank 连续跑时省一次 92 MiB 写盘）。"
                         "投影集合变了会被引擎以 `layer 0 changed its projection set` 拒载，"
                         "那时去掉这个开关重跑即可")
    ap.add_argument("--prune", action="store_true",
                    help="判分完删掉 --save-every 产出的 .stepN.pt（曲线已进报告）。默认保留："
                         "删了就再也复核不了那一份，而 4 份 ≈ 180 MiB")
    ap.add_argument("--eval-only", action="store_true",
                    help="跳过训练，直接对 --out 指向的现成 ckpt 做引擎判分（重评上一轮产物）")
    ap.add_argument("--boost", choices=("auto", "on", "off"), default="auto",
                    help="训练前先跑投影采样（arXiv:2610.02140）生成 <--data>.boost.jsonl，"
                         "sft.py 的 --data-boost auto 会自动用它。auto（默认）= 有 --data 就跑；"
                         "on = 没有 --data 就报错；off = 不跑。产物已存在且指纹相同则直接复用")
    ap.add_argument("--boost-arg", action="append", default=[], metavar="ARG",
                    help="透传给 data_boost.py 的额外参数（可重复），例如 --boost-arg=--steps=8")
    ap.add_argument("train_argv", nargs=argparse.REMAINDER,
                    help="sft.py 的训练参数（`--` 之后）")
    args = ap.parse_args()

    argv = list(args.train_argv)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise SystemExit("需要在 `--` 之后给 sft.py 的训练参数（至少 --data 与 --out）")

    ckpt = _opt(argv, "--out")
    if not ckpt:
        raise SystemExit("训练参数里必须有 --out：没有 ckpt 就没有可评估的产物")
    held = _opt(argv, "--held-data")
    held_gen = _opt(argv, "--held-gen", "0")
    report_path = args.report or (ckpt + ".report.json")
    logdir = _p.Path(args.logdir) if args.logdir else _p.Path(ckpt).parent
    logdir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)      # 注入的 PYTHONPATH 会让 training/ 之外的包抢先命中
    env["PYTHONUTF8"] = "1"          # 中文探针走 GBK 会假失败
    py = sys.executable

    rep: dict = {"ckpt": ckpt, "held_data": held, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ---------- 0. 投影采样（训练前，独占 GPU）----------
    # 为什么放在训练之前、而且单独起一个引擎：boost 要读 logprobs + 生成，训练要占满
    # 48 GiB —— 两者不能同时在卡上（MEMORY 的「GPU 单任务铁律」）。data_boost 自己
    # launch_engine、自己 close，退出时卡是干净的，训练才起来。
    # **必须在训练前**：sft.py 的 --data-boost auto 是在装载数据时读 <data>.boost.jsonl 的。
    if args.boost != "off" and not args.eval_only:
        data_path = _opt(argv, "--data")
        if not data_path:
            if args.boost == "on":
                raise SystemExit("--boost on 但训练参数里没有 --data（没数据可 boost）")
            print("boost: 没有 --data（内置身份数据），跳过", flush=True)
        elif _flag(argv, "--data-boost"):
            # 训练参数里自己写了 --data-boost：由它决定用不用，这里不重复跑一遍
            print("boost: 训练参数已显式给 --data-boost，跳过", flush=True)
        else:
            print("\n=== 投影采样（数据侧，一次性成本）===", flush=True)
            if _engine_running():
                print("已有 strata.exe 在跑：单任务铁律，先停掉", flush=True)
                _kill_engine()
                time.sleep(2.0)
            if not _wait_gpu_free(limit_mib=6000, timeout=120.0):
                print(f"WARN: 显存仍高于 6 GiB（{_gpu_used_mib()} MiB），继续尝试", flush=True)
            bcfg = json.loads(_p.Path(args.engine_config).read_text(encoding="utf-8"))
            bargs = list(bcfg.get("args") or [])
            bcmd = [py, "-u", str(_p.Path(__file__).resolve().parent / "data_boost.py"),
                    "--data", data_path,
                    "--engine", str(bcfg.get("exe") or ""),
                    "--pack", _opt(bargs, "--pack"),
                    "--gguf", _opt(bargs, "--native"),
                    "--max-context", _opt(bargs, "--max-context", "8192"),
                    "--log", str(logdir / "boost.engine.log")] + list(args.boost_arg)
            prof = _opt(bargs, "--expert-profile")
            if prof:
                bcmd += ["--expert-profile", prof]
            rep["boost_secs"] = 0.0
            t_b = time.time()
            brc = _stream(bcmd, str(logdir / "boost.log"), env)
            rep["boost_secs"] = round(time.time() - t_b, 1)
            rep["boost_rc"] = brc
            print(f"=== 投影采样 rc={brc} 用时 {rep['boost_secs']} s ===\n", flush=True)
            if brc != 0:
                # 数据侧失败不该悄悄退化成「用原始数据训练」：那一轮实验的前提就变了。
                rep["ok"] = False
                rep["error"] = "投影采样失败（模型/数据未变，修好后重跑；或 --boost off）"
                _write(report_path, rep)
                return 1
            # 把「这一轮到底用没用增强数据」写进报告。不加这一段，报告里只有 loss 曲线，
            # 两周后没人能说清 easyG2 与 easyB2 的差别是不是数据。
            try:
                from data_boost import detect_provenance, provenance_line, read_input

                _m, _r = read_input(data_path + ".boost.jsonl")
                rep["boost"] = detect_provenance(_r, _m)
                print("  " + provenance_line(rep["boost"], verb="boost"), flush=True)
            except Exception as _e:      # 读不到不该让整轮失败，但要留痕
                rep["boost"] = {"state": "unknown", "error": str(_e)}
                print(f"  WARN: 读 boost 产物身份失败：{_e}", flush=True)
    else:
        # 报告里永远有 `boost` 这一段：有它才能一眼分清「这轮是增强数据还是原始数据」。
        # 缺省不写 = 以后只能靠回忆。
        rep["boost"] = {"state": "off" if args.boost == "off" else "skipped",
                        "note": "本轮未生成/未注入增强数据"}

    # ---------- 1. 训练（唯一占 GPU 的进程）----------
    # held 生成留到评估阶段（见 _drop 的说明）：训练侧只算 held CE。
    train_argv = _drop(argv, ("--held-gen", "--held-max-new"))
    if args.eval_only:
        if not _p.Path(ckpt).is_file():
            raise SystemExit(f"--eval-only 但 ckpt 不存在：{ckpt}")
        print(f"=== 跳过训练，直接评估 {ckpt} ===", flush=True)
        rep["train_secs"] = 0.0
        rep["train_rc"] = rc = 0
    else:
        print(f"\n=== 训练 ===\n{py} {TRAIN} " + " ".join(train_argv), flush=True)
        t0 = time.time()
        rc = _stream([py, "-u", str(TRAIN)] + train_argv, str(logdir / "train.log"), env)
        rep["train_secs"] = round(time.time() - t0, 1)
        rep["train_rc"] = rc
        print(f"=== 训练 rc={rc} 用时 {rep['train_secs']} s ===\n", flush=True)
        if rc != 0:
            rep["ok"] = False
            rep["error"] = "训练失败"
            _write(report_path, rep)
            return 1
    meta_path = _p.Path(ckpt + ".meta.json")
    if meta_path.is_file():
        try:
            rep["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rep["meta"] = {}

    if rc != 0:
        _write(report_path, rep)
        return 1
    if args.skip_eval or not held:
        # 没有 held 集就没有「准确率」这一项：held CE 也不能替代生成（CE 降 82% 而
        # 生成 0/5 是实测过的）。所以这里如实标成未评估，而不是拿训练 loss 冒充。
        rep["ok"] = True
        rep["evaluated"] = False
        rep["note"] = ("--skip-eval" if args.skip_eval else "没有 --held-data：只训练未评估")
        _write(report_path, rep)
        return 0

    # ---------- 2. 引擎判分（训练进程已经退出）----------
    if _engine_running():
        print("已有 strata.exe 在跑：单任务铁律，先停掉再评估", flush=True)
        _kill_engine()
        time.sleep(2.0)
    # 训练进程退出 ≠ 显存已释放（见 _wait_gpu_free）。桌面本身占 ~1 GiB，留 6 GiB 余量。
    if not _wait_gpu_free(limit_mib=6000, timeout=120.0):
        print(f"WARN: 等了 120 s 显存仍高于 6 GiB（现在 {_gpu_used_mib()} MiB）——"
              "继续起引擎，若 OOM 请单跑 --eval-only", flush=True)

    cfg = json.loads(_p.Path(args.engine_config).read_text(encoding="utf-8"))
    port = args.port or int(cfg.get("port") or 8095)

    # 引擎 `--lora` 启动时加载的那一份，必须与训练产物的**投影集合相同**：热换是按层的，
    # CUDA graph 捕获了投影表的地址，`layer 0 changed its projection set` 就是拒载的理由。
    # 因此不用配置文件里写死的老 seed（往往只有两个输出投影），而是从这次训练的 ckpt
    # 现造一份「同结构、B 全零」的 —— 数值上等于基座，于是起点评估就是未训练点。
    #
    # 路径**固定**（`--seed-path`，默认 <ckpt 目录>/seed.gguf）而不是跟着 ckpt 命名：
    # B=0 的 adapter 与 ckpt 无关，只有投影集合/rank 有关。每个 ckpt 各造一份的旧做法
    # 让一个实验目录堆到 5.7 GiB，而其中 4 份内容完全相同。
    seed = _p.Path(args.seed_path) if args.seed_path else (_p.Path(ckpt).parent / "seed.gguf")
    alpha = _opt(argv, "--alpha", "32")
    if args.reuse_seed and seed.is_file():
        print(f"\n=== 复用已有零 seed {seed}（--reuse-seed）===", flush=True)
        src = 0
    else:
        seed_cmd = [py, "-u", str(HERE / "lora_to_gguf.py"), "--pt", ckpt, "--out", str(seed),
                    "--alpha", alpha, "--zero-b"]
        print(f"\n=== 造同结构零 seed ===\n" + " ".join(seed_cmd), flush=True)
        src = _stream(seed_cmd, str(logdir / "seed.log"), env)
    if src != 0 or not seed.is_file():
        rep["ok"] = False
        rep["error"] = "零 seed 生成失败（adapter 结构会与热换要求不符）"
        _write(report_path, rep)
        return 2
    eargs = list(cfg.get("args") or [])
    if "--lora" in eargs:
        eargs[eargs.index("--lora") + 1] = str(seed.resolve())
    else:
        eargs += ["--lora", str(seed.resolve())]
    cfg["args"] = eargs
    run_cfg = (logdir / f"engine_cfg_{port}.json").resolve()
    run_cfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    # `serve.ready.preflight` 拒绝相对路径，也拒绝已存在的文件（它把残留文件当证据污染）。
    # 所以这里必须 resolve 成绝对路径，并且起引擎前清掉上一轮的残留。
    ready_file = (logdir / f"engine_ready_{port}.json").resolve()
    if ready_file.exists():
        ready_file.unlink()
    serve_log = open(logdir / "serve.log", "w", encoding="utf-8", errors="replace")
    print(f"=== 起引擎（port {port}，config {run_cfg}）===", flush=True)
    serve = subprocess.Popen(
        [py, "-m", "serve.server", "--engine", "strata", "--config", str(run_cfg),
         "--port", str(port), "--ready-file", str(ready_file)],
        stdout=serve_log, stderr=subprocess.STDOUT, env=env, cwd=str(_p.Path.cwd()))
    try:
        info = _wait_ready(ready_file, serve, args.ready_timeout)
        model = cfg.get("model_name") or (info.get("model") if isinstance(info, dict) else "") or ""
        print(f"引擎 ready：{info}", flush=True)

        eval_json = str(_p.Path(ckpt + ".eval.json"))
        base = (f"http://127.0.0.1:{port}")
        eval_cmd = [py, "-u", str(TRAIN), "--eval-only", "--init-adapter", ckpt,
                    "--held-data", held, "--held-gen", held_gen or "5",
                    "--gen-engine", base, "--eval-json", eval_json]
        # 渲染契约必须与训练逐字一致，否则引擎评估的是另一条 prompt（而训练侧 CE 仍是
        # 训练侧渲染，两个数不同源）。训练参数里给了就原样搬过来。
        for name in ("--chat-template", "--tokenizer", "--gen-engine-model",
                     "--held-max-new", "--alpha", "--word", "--system-prompt",
                     "--reasoning-effort", "--answer-strict"):
            v = _opt(argv, name)
            if v:
                eval_cmd += [name, v]
        if not _flag(argv, "--gen-engine-model") and model:
            eval_cmd += ["--gen-engine-model", model]

        print(f"\n=== 引擎判分 ===\n" + " ".join(eval_cmd), flush=True)
        t1 = time.time()
        erc = _stream(eval_cmd, str(logdir / "eval.log"), env)
        rep["eval_secs"] = round(time.time() - t1, 1)
        rep["eval_rc"] = erc
        if _p.Path(eval_json).is_file():
            rep["eval"] = json.loads(_p.Path(eval_json).read_text(encoding="utf-8"))
        rep["ok"] = erc == 0
        if erc != 0:
            rep["error"] = "引擎判分失败（ckpt 可用，可单跑 sft.py --eval-only 重试）"

        # 中途存点（`--save-every`）：引擎已经起来了，逐个换表判分，不必重启。
        # 每份 adapter 的换表 + 8 条生成约 70 s，换来的是「步数 → pass@1」的曲线 ——
        # 训练侧插一个 CE 评估点就要同样的量级，而 CE 并不能替代生成（README 有实测）。
        points = sorted(_p.Path(ckpt).parent.glob(_p.Path(ckpt).name + ".step*.pt"),
                        key=lambda q: int(q.name.split(".step")[-1].split(".")[0]))
        if points and erc == 0:
            print(f"\n=== 中途存点判分（{len(points)} 份）===", flush=True)
            curve = []
            for q in points:
                pj = str(q) + ".eval.json"
                cj = [c if c != eval_json else pj for c in eval_cmd]
                cj[cj.index(ckpt)] = str(q)
                print(f"\n--- {q.name} ---", flush=True)
                t2 = time.time()
                r2 = _stream(cj, str(logdir / f"eval_{q.stem}.log"), env)
                row = {"ckpt": q.name, "rc": r2, "secs": round(time.time() - t2, 1)}
                if _p.Path(pj).is_file():
                    ej = json.loads(_p.Path(pj).read_text(encoding="utf-8"))
                    h = ej.get("held") or {}
                    row.update({"hits": h.get("hits"), "n": h.get("n"),
                                "pass_rate": h.get("pass_rate"),
                                "rows": h.get("rows", [])})
                curve.append(row)
            rep["curve"] = curve
            ok_rows = [c for c in curve if c.get("pass_rate") is not None]
            if ok_rows:
                b = max(ok_rows, key=lambda c: (c["pass_rate"], -int(c["n"] or 0)))
                rep["best_by_pass"] = {k: b.get(k) for k in ("ckpt", "hits", "n", "pass_rate")}
            if args.prune:
                # 曲线已经在 rep["curve"] 里（含每条的 hit/text/eos），中间检查点可以走了。
                # 末步 ckpt 不删：那是这次实验的产物。
                freed = 0
                for q in points:
                    for f in (q, _p.Path(str(q) + ".eval.json")):
                        if f.is_file():
                            freed += f.stat().st_size
                            f.unlink()
                print(f"已删 {len(points)} 份中间检查点，回收 {freed / 2**20:.0f} MiB", flush=True)
    finally:
        if not args.keep_engine:
            serve.terminate()
            try:
                serve.wait(timeout=15)
            except subprocess.TimeoutExpired:
                serve.kill()
            _kill_engine()            # serve 会拉起 strata.exe，父进程死了子进程不一定跟着
        serve_log.close()

    rep["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write(report_path, rep)
    _summary(rep, report_path)
    return 0 if rep.get("ok") else 2


def _write(path: str, rec: dict) -> None:
    p = _p.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告 → {path}", flush=True)


def _summary(rep: dict, path: str) -> None:
    """终端里先给结论：训练多久、每问几遍、held 对几条。细节在 json 里。"""
    meta = rep.get("meta") or {}
    ev = (rep.get("eval") or {}).get("held") or {}
    total = rep.get("train_secs", "?")
    # 两个数必须分开写：`steps_secs` 是步本身（meta 里的 train_secs），`total` 是训练子进程的
    # 墙钟（含 122 s 权重加载 + 一次 θ₀ 前向做 token 选择）。混成一个数会让「每步 17 s、
    # 一次实验 15 分钟」看起来像矛盾，也会让人以为优化步内代码能省掉加载。
    steps_secs = meta.get("train_secs")
    print("\n===== 结论 =====", flush=True)
    print(f"训练 {meta.get('steps_done', '?')} 步 / 每问训练 {meta.get('per_question_done', '?')} 次"
          + (f" / 步耗时合计 {steps_secs} s" if steps_secs is not None else "")
          + f" / 训练进程墙钟 {total} s（含加载与编码）"
          f" / 峰值 {meta.get('peak_gib', '?')} GiB", flush=True)
    if ev:
        print(f"held pass@1 {ev.get('hits')}/{ev.get('n')}"
              f"（{100 * float(ev.get('pass_rate') or 0):.0f}%）"
              f"  生成 {ev.get('gen_secs', '?')} s / 上限 {ev.get('max_new', '?')} token", flush=True)
        bad = [r for r in ev.get("rows", []) if not r["hit"]][:3]
        for r in bad:
            print(f"  错例 gold {r['gold']!r} → {r['text'][-70:]!r}  eos={int(r['eos'])}", flush=True)
    else:
        print("未做引擎判分" + (f"：{rep.get('error', '')}" if rep.get("error") else ""), flush=True)
    curve = rep.get("curve") or []
    if curve:
        print("中途存点 pass@1 曲线:", flush=True)
        for c in curve:
            pr = c.get("pass_rate")
            print(f"  {c['ckpt']:<28} {c.get('hits')}/{c.get('n')}"
                  + (f"  ({100 * pr:.0f}%)" if pr is not None else "  (判分失败)")
                  + f"  {c['secs']} s", flush=True)
        b = rep.get("best_by_pass")
        if b:
            print(f"  最好的一步：{b['ckpt']}（{b.get('hits')}/{b.get('n')}）", flush=True)
    print(f"完整记录 {path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
