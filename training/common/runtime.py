"""引擎会话 —— 把 strata 进程包装成 score() / lora() / gen() 三个动作。

三种 transport：命令管道（Engine，本地子进程）、EngineApi（引擎自带 HTTP API）、
EngineHttp（OpenAI 兼容端点，只读）。训练只用到前两个。
"""

from __future__ import annotations

import array
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

PREFILL_CHUNK_CAP = 8192   # 引擎 `--prefill auto` 自己的上限；再大是上游 opt-in（#282）


class ReplayScoreMixin:
    """LAYER REPLAY 的训练侧接线（引擎侧：`src/prefill/prefill.hpp` 的 `set_replay`）。

    零阶一步要把**同一条序列**读 2 × dirs 次。adapter 只挂在层 >= L 上时（训练一直如此：
    末 2 层 ⇒ L = 46），层 0..L-1 写进残差流的东西**不含 θ**，于是层 L 的输入残差是
    「序列 + 基座权重」的纯函数。读一次、存下来，之后每次只走层 [L, 48)。

    引擎命令（`--serve` 的命令循环）：
      `FRONTIER L n`   捕获层 L 的输入残差（n 个绝对位置，n × residual_row 个 float，host）
      `FRONTIER -1 0`  停止捕获，**保留**已捕获的行
      `REPLAY L`       从这些行起步，只走层 [L, 48)
      `REPLAY -1`      退回全读（位置 0、token embedding）

    本类做的事只有一件：**按序列缓存捕获**。同一条序列第二次起直接重放；换了序列才退回
    全读重新捕获。训练代码照旧调 `score()`，不需要知道重放在不在。

    逐位验证（2026-11-16，8192 token，big_r4）：
      * 同一 adapter：重放 vs 全读 Δsum +0.0000、max|Δtok| 0.00000；
      * 换 adapter（同 rank，B×0.5）：重放**吃到了新表**（Δsum +24827.89），换回原表逐位复原
        ⇒ 重放没有把 LoRA 静默丢掉（这正是 2026-11-15 那条 layer-major 走法的病）；
      * 重放后退回全读也逐位复原 —— 前提是引擎 `Prefill::reset` 清掉 `hand_in_`，否则全读
        会把重放的行走成 embedding（128K 上的签名是 −12.4 nats/token，即词表均匀分布）。
    步时（8192）：全读 2.6 s → 重放 1.2 s（2.2×）。层数越多、序列越长，比值越大。

    `enable_replay` 的 `layer` 必须是**训练层集合里最小的那一层**：给大了（层 < L 上还有
    非零 B）读数会看着正常但是错的。所以第一次捕获后自动做一次全读 vs 重放的逐位比对，
    不等就抛错并关掉重放 —— 不让它静默降级成一个错的加速。
    """

    replay_layer = -1          # < 0：未启用（score 直接走 _score_raw，开销是一次属性比较）
    replay_stats: dict = {}
    _replay_key = None
    _replay_checked = False
    _keep_sig = None           # 引擎此刻装着的行掩码（None = 没设过 / 已清空）
    last_score: dict = {}      # 上一遍 SCORE 的引擎自报账（SSUM 行）

    def enable_replay(self, layer: int) -> None:
        if layer is None or int(layer) <= 0:
            self.replay_layer = -1
            return
        if not hasattr(self, "_command"):
            raise NotImplementedError("this transport has no command channel "
                                      "(LAYER REPLAY needs FRONTIER / REPLAY)")
        self.replay_layer = int(layer)
        self._replay_key = None
        self._replay_checked = False
        self.replay_stats = {"captures": 0, "replays": 0, "capture_s": 0.0, "replay_s": 0.0}

    def disable_replay(self) -> None:
        if self.replay_layer >= 0:
            # keyed 或 dance 状态下路径都可能停在重放态：回全读，后续 plain SCORE / GEN 才是安全的
            self._command("REPLAY -1", ("REPLAYOK",), 1800.0)
        self.replay_layer = -1
        self._replay_key = None
    @staticmethod
    def _ids_key(ids) -> bytes:
        # 长度 + 全部 id：128K ⇒ 1 MB，blake2b 约 1 ms，相对一次 SCORE 可忽略。
        return hashlib.blake2b(array.array("q", ids).tobytes(), digest_size=16).digest()

    @staticmethod
    def _keep_bits(keep) -> bytes:
        """行掩码 → 位图（bit t = 位置 t，LSB 优先）。位置数一次几十万，numpy 走 packbits。"""
        try:
            import numpy as np
            return np.packbits(np.asarray(keep, dtype=bool), bitorder="little").tobytes()
        except ImportError:   # 没有 numpy 的调用方（tools/ 下的小脚本）：纯 python 也只要 20 ms
            out = bytearray((len(keep) + 7) // 8)
            for i, v in enumerate(keep):
                if v:
                    out[i >> 3] |= 1 << (i & 7)
            return bytes(out)

    def apply_keep(self, keep) -> None:
        """把 loss 真正读的那些行告诉引擎（`SCOREKEEP`）。没变就不发：一条掩码 128K 是 32 KB。

        这不是近似：零阶一步的目标是被选中 token 的**求和**，没选中的位置对 f(θ) 的贡献恒等于 0，
        而引擎在一行上跑 head 的代价与位置数成正比（128K 全词表 15.4 s、8 行一批）。跳过它们读数
        逐位不变，只有被选中那些行的值要一模一样。"""
        if not hasattr(self, "_command"):
            if keep is not None:
                raise NotImplementedError("this transport has no command channel (SCORE's row mask "
                                          "needs SCOREKEEP)")
            return
        bits = b"" if keep is None else self._keep_bits(keep)
        sig = hashlib.blake2b(bits, digest_size=8).digest()
        if sig == self._keep_sig:
            return
        n = 0 if keep is None else len(keep)
        self._command(f"SCOREKEEP {n} {bits.hex()}" if n else "SCOREKEEP 0", ("SCOREKEEPOK",), 120.0)
        self._keep_sig = sig

    def _note_ssum(self, line: str) -> None:
        """记下引擎这一遍 SCORE 的自报账（head 毫秒 / 跳过的行 / keyed 模式），
        训练日志用它判断掩码生没生效、这一遍是全读还是重放。"""
        p = line.split()
        self.last_score = {"sum": float(p[1]) if len(p) > 1 else 0.0,
                           "rows": int(p[2]) if len(p) > 2 else 0,
                           "ms": float(p[3]) if len(p) > 3 else 0.0,
                           "head_ms": float(p[4]) if len(p) > 4 else 0.0,
                           "cols": int(p[5]) if len(p) > 5 else 0,
                           "skipped": int(p[6]) if len(p) > 6 else 0,
                           "mode": p[7] if len(p) > 7 else "full"}

    def _score_keyed(self, ids: list[int], key: int, layer: int, force: bool = False) -> tuple[list[float], float]:
        """`SCORE @<key>,<L>[!] <ids>`：引擎端按 key 缓存层 L 的输入残差（每条序列一槽）。
        命中重放（~6 ms），未命中全读并捕获；`!` 强制重捕获（自检拿全读真值用）。
        回答压缩成一行 `SLA <n> <csv>`（%.9g 保 float32 精确回读）；SSUM 末尾多一个
        mode 字段（full/cap/rep）。"""
        t0 = time.time()
        head = f"SCORE @{key:x},{layer}" + ("!" if force else "") + " "
        lines = self._command(head + ",".join(map(str, ids)), ("SSUM",), timeout=1800.0)
        slp: list[float] = []
        for ln in lines:
            if ln.startswith("SLA "):
                parts = ln.split(" ", 2)
                slp = [float(v) for v in parts[2].split(",")] if len(parts) > 2 and parts[2] else []
            elif ln.startswith("SL "):
                slp.append(float(ln.split()[2]))
        self._note_ssum(lines[-1])
        return slp, (time.time() - t0) * 1000.0

    def score(self, ids: list[int], keep=None) -> tuple[list[float], float]:
        if keep is not None or self._keep_sig is not None:
            self.apply_keep(keep)
        L = self.replay_layer
        if L < 0:
            return self._score_raw(ids)
        # keyed 通道（2026-11-16）：引擎端每条序列一槽，轮转数据集不再每次切序列都重新全读
        # （旧单槽 dance 实测 20 步 57 次捕获，占掉 67% 的 score 时间）。命中重放 ~6 ms。
        key = int.from_bytes(self._ids_key(ids)[:8], "big")
        if not self._replay_checked:
            # 一次性自检，语义不变（重放与全读必须逐位一致）：`!` 强制重捕获拿全读真值，
            # 再普通读一次（应命中重放），比对。两次的读数都是训练能用的数。
            truth, _ = self._score_keyed(ids, key, L, force=True)
            chk, ms = self._score_keyed(ids, key, L)
            if len(chk) != len(truth) or chk != truth:
                self.replay_layer = -1
                bad = next((i for i, (a, b) in enumerate(zip(chk, truth)) if a != b), -1)
                raise RuntimeError(
                    f"LAYER REPLAY 自检失败：层 {L} 起的重放与全读不一致（首个不同的位置 {bad}，"
                    f"{len(truth)} 个位置）。要么 adapter 在层 < {L} 上有非零 B，要么引擎的重放路径坏了。"
                    "关掉 --replay 再跑。")
            self._replay_checked = True
            self.replay_stats["captures"] += 1
            self.replay_stats["capture_s"] += ms / 1000.0
            return chk, ms
        slp, ms = self._score_keyed(ids, key, L)
        mode = (self.last_score or {}).get("mode", "")
        if mode == "cap":
            self.replay_stats["captures"] += 1
            self.replay_stats["capture_s"] += ms / 1000.0
        else:
            self.replay_stats["replays"] += 1
            self.replay_stats["replay_s"] += ms / 1000.0
        return slp, ms

    def gen(self, ids: list[int], max_new: int) -> tuple[list[int], float]:
        # GEN 要逐 token 读 prompt，必须在全读态。keyed 重放后路径停在重放态，dance 亦然。
        # 回全读后**不再**重进重放：keyed 模式下引擎没有单槽 frontier（REPLAY <L> 会 ERR），
        # 下一次 keyed score 自己布状态（命中布重放、未命中布捕获）。
        if self.replay_layer >= 0:
            self._command("REPLAY -1", ("REPLAYOK",), 1800.0)
        return self._gen_raw(ids, max_new)



class Engine(ReplayScoreMixin):
    """stdin/stdout 命令管道（同 tools/bench_engine_mezo.py 的模式）。"""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc

    def close(self) -> None:
        """收尾统一入口：两种 transport 都有 close()，调用方不感知差别。"""
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.proc = None

    def send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_reply(self, prefixes: tuple[str, ...], timeout: float = 600.0) -> list[str]:
        """读到 prefixes 之一的应答行为止；`ERR ...` 直接抛。"""
        deadline = time.time() + timeout
        lines = []
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise SystemExit(f"the engine died: rc={self.proc.poll()} (see the log)")
            line = line.rstrip("\n")
            if line.startswith("ERR"):
                raise RuntimeError(f"the engine said: {line}")
            if any(line.startswith(p) for p in prefixes):
                lines.append(line)
                return lines
            lines.append(line)   # SL ... / T id ... / others: the caller inspects them
        raise TimeoutError(f"no reply starting with {prefixes} in {timeout:.0f} s")

    def _command(self, line: str, until: tuple[str, ...], timeout: float = 1800.0) -> list[str]:
        """发一条命令、读到 `until` 之一的应答（`ReplayScoreMixin` 的 FRONTIER / REPLAY 用）。
        与 `EngineApi._command` 同签名，这样重放的接线不用分辨 transport。"""
        self.send(line)
        return self.read_reply(until, timeout=timeout)

    def _score_raw(self, ids: list[int]) -> tuple[list[float], float]:
        t0 = time.time()
        self.send("SCORE " + ",".join(map(str, ids)))
        slp: list[float] = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise SystemExit(f"the engine died during SCORE: rc={self.proc.poll()}")
            line = line.rstrip("\n")
            if line.startswith("SL "):
                p = line.split()
                v = float(p[2])
                slp.append(v)
            elif line.startswith("SSUM "):
                self._note_ssum(line)
                return slp, (time.time() - t0) * 1000.0
            elif line.startswith("ERR"):
                raise RuntimeError(f"SCORE failed: {line}")

    def lora(self, path: str) -> None:
        self.send("LORA " + path)
        line = self.proc.stdout.readline()
        if not line:
            raise SystemExit(f"the engine died during LORA: rc={self.proc.poll()}")
        line = line.rstrip("\n")
        if not line.startswith("LORAOK"):
            raise RuntimeError(f"LORA failed: {line}")

    def _gen_raw(self, ids: list[int], max_new: int) -> tuple[list[int], float]:
        """生成到 EOS 或 `max_new`。`self.last_finish` 是引擎 DONE 行的收尾原因
        （`stop` 命中 EOS / `length` 撞上 max_new / `cancel`），只读不改签名 ——
        评测要靠它区分「答完停」和「上下文满了硬停」，训练侧现有调用方不受影响。"""
        t0 = time.time()
        self.send(f"GEN {max_new} " + ",".join(map(str, ids)))
        out: list[int] = []
        self.last_finish = ""
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise SystemExit(f"the engine died during GEN: rc={self.proc.poll()}")
            line = line.rstrip("\n")
            if line.startswith("T "):
                out.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                p = line.split()
                self.last_finish = p[5] if len(p) > 5 else ""
                return out, (time.time() - t0) * 1000.0
            elif line.startswith("ERR"):
                raise RuntimeError(f"GEN failed: {line}")


def resolve_prefill_chunk(longest_tokens: int, given: str = "") -> str:
    """prompt 路径预留多少 token 的 buffer：**数据里最长的那条**，不是常数。

    `--prefill N` 是 prompt 的**分块大小**，不是长度上限：比 N 长的序列照样读完，只是分成
    多块（`request_chunk = min(N, ceil256(tokens))`）。所以 N 唯一决定的是「一条样本读几块、
    一块预留多大」。在 `--no-prefill-borrow` 下这块 buffer 是**整个会话**的预留（不借专家
    缓存槽，见 BRANCH-NOTES §4.3），于是 N 被 8192 钉死时：300-600 token 的数学数据一块都
    没多读，却按 8192 的大小年年占着专家缓存。取「最长样本向上对齐到 256」两头都占：
    每条样本仍是一块读完（最快），预留只按真实数据大小。

    `given` 非空 = 用户显式给了（数字或 `auto`），照用；最长样本未知（无数据）时交回
    `auto`，让引擎自己挑（no-borrow 下引擎把预留压到 2048，1,826 MiB）。上限 8192：更长就分块读，
    用户可显式抬（128K 数据）。**但抬了不划算**：2026-11-15 实测 128K 上 8192→32768 只把一条
    SCORE 从 45.5 s 压到 43.5 s（−4%），prompt buffer 却是 4× —— 长上下文的钱花在
    「每个 token 走完 48 层」，不在分块次数。"""
    if given:
        return given
    if longest_tokens <= 0:
        return "auto"
    want = (int(longest_tokens) + 255) // 256 * 256
    return str(max(256, min(PREFILL_CHUNK_CAP, want)))


def engine_argv(args, init_adapter: str, extra: list[str] | None = None) -> list[str]:
    """训练引擎的命令行（pipe 与 http 两种 transport 用同一份定义）。

    `--serve` 是引擎的常驻模式；它的命令循环同时收 LORA/SCORE/GEN（src/program/generate.cpp）。

    **专家搬运、KV 档位、显存预留一律不写**：引擎自己的默认就是这三件事的答案
    （`--expert-cache auto` 是 0 → auto、`--kv` 默认 auto、`--vram-reserve-mib` 默认 700；
    见 generate.cpp 的 Options）。不写还多一条：`vram_reserve_given` 保持 false，WDDM 机器上
    引擎会按自己的规则把预留抬到它认为安全的值（generate.cpp:8063）。训练侧只决定
    「哪个 adapter、多大上下文、prompt 分几块」——这三件是数据属性，引擎看不见。

    `--no-prefill-borrow`：借槽 bug 的已知判据（见 BRANCH-NOTES §4.3），SCORE 的数要靠它。
    `--prefill` 取 `resolve_prefill_chunk()` 的结果：数据定，不写死 8192。"""
    argv = [args.engine, "--pack", args.pack, "--native", args.gguf, "--ple-gguf", args.gguf,
            "--expert-profile", args.expert_profile,
            "--prefill", str(getattr(args, "prefill_chunk", "") or "auto"),
            "--spec", "4", "--max-context", str(args.max_context),
            "--serve", "--no-prefill-borrow"]
    # 空串时 `os.path.abspath("")` 是 **cwd**，引擎会拿目录去 open 并拒绝启动（rc=None，
    # 日志只有一行 "LoRA: cannot open <cwd>"）。训练侧历来都传真实路径（init.gguf / --init-adapter），
    # 所以这个洞只在「基座臂」这类探针里露出来 —— 不传就是不加载 adapter，语义上本来就该省掉这个参数。
    if init_adapter:
        argv += ["--lora", os.path.abspath(init_adapter)]
    # `extra`：只给「需要非默认采样行为」的调用方用（目前只有 `data_boost.py`，它要
    # 采样温度来做 MH 提议）。放在最后是因为它**覆盖**前面的默认值，引擎的解析是后写的赢。
    if extra:
        argv += [str(x) for x in extra]
    return argv


def launch_engine(args, log_path: str, init_adapter: str,
                  extra: list[str] | None = None) -> tuple[Engine, float]:
    cmd = engine_argv(args, init_adapter, extra)
    # pipe 直接拉第二个引擎进程不经过 serve 的排队/槽位（那些防的是单进程内的请求），
    # 双驻留实测会打爆主机提交内存、引擎 stall 自杀 —— 拉起前查卡上是否已有驻留。
    import mem_guard
    mem_guard.require_exclusive_gpu(mem_guard.estimate_engine_gib(args.max_context))
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                            encoding="utf-8", errors="replace", bufsize=1,
                            cwd=os.path.dirname(os.path.abspath(args.engine)))
    eng = Engine(proc)
    # 引擎的上下文窗口挂到句柄上：调用方要回答「还能写多少」时只能问它，
    # 自己填一个常数就是人为截断（`data_boost` 的每段预算用它）。
    eng.max_context = int(getattr(args, "max_context", 0) or 0)
    while True:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit(f"the engine refused to start (rc={proc.poll()}), log: {log_path}")
        if line.startswith("READY"):
            return eng, time.time() - t0
        if line.startswith("ERR"):
            raise SystemExit(f"the engine refused to start: {line.strip()}")


class EngineApi(ReplayScoreMixin):
    """HTTP transport：训练请求走常驻 API 服务的 /v1/training/*（server.py 的 TrainBridge）。
    与 Engine 同接口（score/lora/gen），MeZO 循环对 transport 无感。
    会话 open 时服务端先卸载推理引擎（训练与推理互斥），close 后由下一个请求自动拉回。"""

    def __init__(self, base_url: str, argv: list[str]):
        self.base = base_url.rstrip("/")
        self.argv = list(argv)
        # 回环必须绕开代理（strata_client.py 的同款判据）
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        code, body = self._post("/v1/training/session", {"op": "open", "argv": argv}, timeout=600.0)
        if code != 200:
            raise SystemExit(f"training session open failed (HTTP {code}): {self._err(body)}")
        print(f"training session open (pid {body.get('pid')})", flush=True)
        self._open = True

    @staticmethod
    def _err(body: dict) -> str:
        e = body.get("error") if isinstance(body, dict) else None
        return e.get("message", str(body)) if isinstance(e, dict) else str(body)

    def _post(self, path: str, body: dict, timeout: float) -> tuple[int, dict]:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode("utf-8"))
            except (ValueError, OSError):
                return e.code, {}

    def _command(self, line: str, until: tuple[str, ...], timeout: float) -> list[str]:
        if not self._open:
            raise SystemExit("the training session is closed")
        code, body = self._post("/v1/training/command", {"line": line, "until": list(until), "timeout": timeout},
                                timeout=timeout + 30.0)
        if code == 200:
            return body.get("lines", [])
        if code == 502:                              # 引擎 ERR / 桥断：会话可能已被服务端关闭
            if "died" in self._err(body):
                self._open = False
            raise RuntimeError(f"the engine said: {self._err(body)}")
        if code == 409:
            self._open = False
            raise SystemExit(f"training session lost (HTTP 409): {self._err(body)}")
        raise RuntimeError(f"/v1/training/command failed (HTTP {code}): {self._err(body)}")

    def close(self) -> None:
        if self._open:
            self._open = False
            try:
                self._post("/v1/training/session", {"op": "close"}, timeout=30.0)
            except OSError:
                pass

    def _score_raw(self, ids: list[int]) -> tuple[list[float], float]:
        t0 = time.time()
        lines = self._command("SCORE " + ",".join(map(str, ids)), ("SSUM", "ERR"),
                              max(60.0, len(ids) / 300.0 + 60.0))
        slp = [float(p.split()[2]) for p in lines if p.startswith("SL ")]
        for p in lines:
            if p.startswith("SSUM"):
                self._note_ssum(p)
                break
        if not any(p.startswith("SSUM") for p in lines):
            raise RuntimeError("SCORE ended without SSUM")
        return slp, (time.time() - t0) * 1000.0

    def lora(self, path: str) -> None:
        lines = self._command("LORA " + path, ("LORAOK", "ERR"), 300.0)
        if not any(p.startswith("LORAOK") for p in lines):
            raise RuntimeError("LORA ended without LORAOK")

    def _gen_raw(self, ids: list[int], max_new: int) -> tuple[list[int], float]:
        t0 = time.time()
        lines = self._command(f"GEN {max_new} " + ",".join(map(str, ids)), ("DONE", "ERR"), 600.0)
        out = [int(p.split()[1]) for p in lines if p.startswith("T ")]
        if not any(p.startswith("DONE") for p in lines):
            raise RuntimeError("GEN ended without DONE")
        return out, (time.time() - t0) * 1000.0


class EngineHttp(ReplayScoreMixin):
    """HTTP transport：驱动**已经在跑**的推理服务（`POST /v1/engine/score` / `POST /v1/engine/lora`），不卸载它。

    与 `Engine` / `EngineApi` 同接口（score/lora/gen），所以 MeZO 循环对 transport 无感 ——
    三者的差别只在「谁持有 GPU」：

        Engine      自己拉一个训练引擎子进程（pipe；独占）
        EngineApi   让服务端卸掉推理引擎再起训练引擎（`/v1/training/*`；离线，推理中断）
        本类         直接用驻留的推理引擎（不卸载、不重启 ⇒ 推理不中断，即「无感在线」）

    前提（服务端启动时就决定，训练侧补不了）：
      * `--lora <初始 adapter>`：`/v1/engine/lora` 换的是「引擎启动时带着的那套表」的内容，
        形状必须不变 ⇒ rank 全程固定。没有现成 adapter 时用 `mezo/zero_adapter.py` 造一个
        （A = B = 0，带着它服务与不带逐字相同），rank 取 `--rank`（默认 8）；
      * `--no-prefill-borrow`：否则 score 走借槽路径，数值已知会坏（BRANCH-NOTES §4.3）。
        prompt buffer 的**分块大小**（`--prefill`）由数据里最长样本定，见
        `resolve_prefill_chunk()`；显式给 `--prefill-chunk` 就照用。这条会同时影响推理的
        prefill 速度/显存，是「共用引擎」的真实代价，不在训练侧掩盖。

    两个与生俱来的限制（不是 bug，是「共用」的定义）：
      * 每次换表清空所有 K/V：训练期间在跑的对话，下一个请求从 token 0 重读 prompt；
      * 服务端没有「按 ids 生成」的端点 ⇒ `gen` 不可用（`supports_gen = False`），
        eval 的生成检查跳过，只留 logp / answer-ce 两条基线。

    `score` 的长度契约必须与 pipe 逐位一致：引擎的 `SL` 只发 **n-1** 行（`t + 1 < n`，
    src/program/generate.cpp），而 `/v1/engine/score` 为了让 `logprobs` 与 `tokens` 对齐**补了一个
    末位 None** ⇒ 这里去掉末位，两种 transport 的切片语义才是同一个数。
    """

    supports_gen = False

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        # 回环必须绕开代理（strata_client.py 的同款判据）
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @staticmethod
    def _err(body: dict) -> str:
        e = body.get("error") if isinstance(body, dict) else None
        return e.get("message", str(body)) if isinstance(e, dict) else str(body)

    def _post(self, path: str, body: dict, timeout: float) -> tuple[int, dict]:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode("utf-8"))
            except (ValueError, OSError):
                return e.code, {}

    def _score_raw(self, ids: list[int]) -> tuple[list[float], float]:
        t0 = time.time()
        code, body = self._post("/v1/engine/score", {"tokens": [int(t) for t in ids]},
                                max(60.0, len(ids) / 300.0 + 60.0))
        if code != 200:
            raise RuntimeError(f"/v1/engine/score failed (HTTP {code}): {self._err(body)}")
        lps = [0.0 if v is None else float(v) for v in (body.get("logprobs") or [])]
        return lps[:-1], (time.time() - t0) * 1000.0        # 末位是服务端补的对齐位，pipe 不发

    def lora(self, path: str) -> None:
        # 服务端要求「它能读到的文件名」；同一台机器上绝对路径是最不容易错的形式
        code, body = self._post("/v1/engine/lora", {"path": path}, 300.0)
        if code != 200:
            raise RuntimeError(f"/v1/engine/lora failed (HTTP {code}): {self._err(body)}")

    def _gen_raw(self, ids: list[int], max_new: int) -> tuple[list[int], float]:
        raise RuntimeError("gen is not available over the HTTP transports: the server has no "
                           "generate-from-ids endpoint (set supports_gen and guard the caller)")

    def close(self) -> None:
        """什么都不做：推理引擎是别人的，训练结束不能把它关掉 —— 那正是本 transport 的意义。"""
        return None


