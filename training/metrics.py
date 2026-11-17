"""训练指标接口：轮次 / 损失在**训练进行中**就能被外部读到，不必等进程退出。

为什么要有这个文件（用户口径 2026-11-12）：整体架构是 RSI 形式的在线训练 ——
「训练一轮 → 评测 → 决策 → 再采数据 → 再训练」由外部控制器驱动。控制器需要在第 N 轮
结束时立刻拿到该轮的 loss / 指标 / 耗时，才能决定下一轮要不要继续、要不要调 lr；
等整条训练跑完再 parse 人类日志，RSI 就退化成离线批训练。日志文本一改格式就断契约，
所以对外只有一份结构化 schema。

两条通道，同一份数据（都从 `MetricSink.emit` 出去）：

  1) JSONL 文件 `--metrics run.jsonl`：一事件一行，写完立即 flush ⇒ 控制器 `tail -f` 即可。
  2) 本地 HTTP `--metrics-port 8765`：训练进程自己起一个只读线程，不额外拉服务。
        GET /health                  → 是否还在跑 + 最新 seq
        GET /metrics?since=N&limit=  → seq > N 的事件
        GET /summary                 → 按轮聚合（每轮首末 step 的 loss、指标、耗时）
     只绑 127.0.0.1：这是训练进程的内部接口，不对外网暴露。

事件 schema（字段只增不减 —— 控制器按名字读，缺字段当 None）::

    {"seq":1,"t":...,"event":"start","run":"sft-mezo-math","algo":"sft","method":"mezo",
     "rounds":20.0,"steps":320,"lr":1e-4,"data":"...","items":16,"steps_per_epoch":16}
    {"seq":2,"t":...,"event":"step","step":0,"round":0,"loss":3.31,"pj":-1.2e-3,
     "rel":0.0042,"theta_norm":19.8,"lr":1e-4,"ms":520,"tok":187}
    {"seq":3,"t":...,"event":"round","round":1,"step":15,"loss":3.05,"metric":3.05,
     "best":3.05,"wall_s":8.3,"lr":1e-4}
    {"seq":4,"t":...,"event":"eval","round":1,"step":15,"tag":"train","ok":3,"total":16}
    {"seq":5,"t":...,"event":"end","steps_done":320,"wall_s":260,"reason":"done"}

`loss` 一律是「训练步上这个 batch 的目标值」（ce 是 nats/token；dpo 是 −logσ 项；
grpo/luspo 是 −优势·ratio 的平均），口径由算法子类给，跨算法**不可比** —— 曲线只在
同一算法内部看趋势。`metric` 是早停判据（ce 有答案段就是 answer-ce；其余是 −logp(word)）。

零阶线不 import torch：这里只用标准库。
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BUFFER = 50000          # 内存里留最近多少条（文件不截断）


def _plain(v):
    """转成 JSON 能写的类型：numpy 标量 / 数组在这里一次性收口。

    调用方传的都是 numpy 结果（pj、rel、‖θ‖），`json.dumps` 遇到 np.float32 直接抛错；
    各调用点各写一次 `float(...)` 迟早漏一个，所以收口到 emit 里。
    """
    if isinstance(v, (str, bool)) or v is None:
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return None if v != v else v          # NaN → null（JSON 没有 NaN）
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    try:                                       # numpy 标量 / 0 维数组
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return str(v)


class MetricSink:
    """一个 run 的指标出口。`path` 与 `port` 都可以为空 = 什么都不做（默认不改变录音行为）。"""

    def __init__(self, path: str = "", port: int = 0, run: str = ""):
        self.run = run or "run"
        self.path = str(path or "")
        self._lock = threading.Lock()
        self._seq = 0
        self._buf: list[dict] = []
        self._fh = None
        self._srv = None
        self._thread = None
        self._running = True
        self._t0 = time.time()
        self._rounds: dict[int, dict] = {}
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "w", encoding="utf-8")
        if port:
            self._start_http(int(port))
        self.emit("open", run=self.run, pid=__import__("os").getpid())

    # ---- 写 ---------------------------------------------------------------
    def emit(self, event: str, **fields) -> dict:
        rec = {"seq": 0, "t": round(time.time(), 3), "event": event}
        rec.update({k: _plain(v) for k, v in fields.items()})
        with self._lock:
            self._seq += 1
            rec["seq"] = self._seq
            self._buf.append(rec)
            if len(self._buf) > MAX_BUFFER:
                del self._buf[:len(self._buf) - MAX_BUFFER]
            if self._fh is not None:
                self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._fh.flush()            # RSI 控制器 tail 的是「已落盘」的行
            r = rec.get("round")
            if event == "step" and isinstance(r, int):
                agg = self._rounds.setdefault(r, {"round": r, "steps": 0, "loss_sum": 0.0,
                                                  "loss_n": 0, "first_loss": None,
                                                  "last_loss": None, "t0": rec["t"],
                                                  "ms_sum": 0.0})
                agg["steps"] += 1
                if rec.get("loss") is not None:
                    agg["loss_sum"] += rec["loss"]
                    agg["loss_n"] += 1
                    if agg["first_loss"] is None:
                        agg["first_loss"] = rec["loss"]
                    agg["last_loss"] = rec["loss"]
                if rec.get("ms") is not None:
                    agg["ms_sum"] += rec["ms"]
            elif event in ("round", "eval") and isinstance(r, int):
                self._rounds.setdefault(r, {"round": r, "steps": 0, "loss_sum": 0.0,
                                            "loss_n": 0, "first_loss": None,
                                            "last_loss": None, "t0": rec["t"],
                                            "ms_sum": 0.0}).update(
                    {k: rec[k] for k in ("metric", "best", "ok", "total", "wall_s", "tag")
                     if k in rec})
        return rec

    # 便捷封装：调用点读起来就是「一步 / 一轮 / 一次评测」
    def start(self, **meta) -> dict:
        return self.emit("start", run=self.run, **meta)

    def step(self, **f) -> dict:
        return self.emit("step", **f)

    def rnd(self, **f) -> dict:
        return self.emit("round", **f)

    def eval(self, **f) -> dict:
        return self.emit("eval", **f)

    def end(self, **f) -> dict:
        return self.emit("end", **f)

    # ---- 读（HTTP 侧用） ---------------------------------------------------
    def since(self, seq: int = 0, limit: int = 0) -> list[dict]:
        with self._lock:
            out = [r for r in self._buf if r["seq"] > seq]
        return out[:limit] if limit else out

    def summary(self) -> dict:
        with self._lock:
            rounds = []
            for r in sorted(self._rounds):
                a = dict(self._rounds[r])
                n = a.pop("loss_n")
                a["loss_mean"] = round(a.pop("loss_sum") / n, 6) if n else None
                steps = a["steps"]
                a["ms_mean"] = round(a.pop("ms_sum") / steps, 1) if steps else None
                rounds.append(a)
            return {"run": self.run, "seq": self._seq, "running": self._running,
                    "wall_s": round(time.time() - self._t0, 1), "rounds": rounds}

    # ---- HTTP --------------------------------------------------------------
    def _start_http(self, port: int) -> None:
        sink = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a):        # 不往训练日志里灌访问行
                return

            def _send(self, obj, code=200):
                body = json.dumps(obj, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                from urllib.parse import parse_qs, urlparse
                u = urlparse(self.path)
                q = parse_qs(u.query)
                try:
                    since = int(q.get("since", ["0"])[0])
                    limit = int(q.get("limit", ["0"])[0])
                except ValueError:
                    since, limit = 0, 0
                if u.path == "/health":
                    last = sink.since(sink._seq - 1)
                    self._send({"ok": True, "run": sink.run, "seq": sink._seq,
                                "running": sink._running,
                                "last": last[0] if last else None})
                elif u.path == "/metrics":
                    self._send({"run": sink.run, "seq": sink._seq,
                                "events": sink.since(since, limit)})
                elif u.path == "/summary":
                    self._send(sink.summary())
                else:
                    self._send({"error": "GET /health | /metrics?since=N | /summary"}, 404)

        self._srv = ThreadingHTTPServer(("127.0.0.1", port), H)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True,
                                        name="metrics-http")
        self._thread.start()

    @property
    def port(self) -> int:
        return self._srv.server_address[1] if self._srv else 0

    def close(self) -> None:
        with self._lock:
            self._running = False
        if self._srv is not None:
            self._srv.shutdown()
            self._srv.server_close()
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


# ---- 读取端：给控制器 / 事后分析共用的两个函数 -------------------------------

def read_metrics(path: str, since: int = 0) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("seq", 0) > since:
                out.append(r)
    return out


def rounds_of(path: str) -> list[dict]:
    """从 JSONL 还原「每轮」记录（优先用显式 round 事件，没有就从 step 聚合）。

    这是画图的唯一取数口径：图上的每个点必须来自这里，不能画图时再算一遍。
    """
    events = read_metrics(path)
    explicit = [e for e in events if e["event"] == "round"]
    if explicit:
        by = {e["round"]: dict(e) for e in explicit}
        for e in events:                        # 把同一轮的 eval 结果挂上去
            if e["event"] == "eval" and e.get("round") in by:
                by[e["round"]].setdefault("eval", []).append(
                    {k: e.get(k) for k in ("tag", "ok", "total", "step")})
        return [by[k] for k in sorted(by)]
    agg: dict[int, dict] = {}
    for e in events:
        if e["event"] != "step" or not isinstance(e.get("round"), int):
            continue
        a = agg.setdefault(e["round"], {"round": e["round"], "steps": 0, "loss_sum": 0.0,
                                        "n": 0, "last_loss": None, "t0": e["t"]})
        a["steps"] += 1
        if e.get("loss") is not None:
            a["loss_sum"] += e["loss"]
            a["n"] += 1
            a["last_loss"] = e["loss"]
    for a in agg.values():
        a["loss_mean"] = round(a.pop("loss_sum") / a.pop("n"), 6) if a.get("n") else None
    return [agg[k] for k in sorted(agg)]


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="读训练指标 JSONL：轮次摘要")
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true", help="原样打印轮次 JSON")
    a = ap.parse_args()
    rows = rounds_of(a.path)
    if a.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0
    print(f"{'round':>5} {'steps':>6} {'loss_mean':>10} {'metric':>9} {'best':>9} "
          f"{'eval':>9} {'ms/step':>8}")
    for r in rows:
        ev = r.get("eval") or [{}]
        e = ev[-1]
        evs = f"{e['ok']}/{e['total']}" if e.get("total") else ""
        print(f"{r['round']:5d} {r.get('steps', 0):6d} {str(r.get('loss_mean', '')):>10} "
              f"{str(r.get('metric', '')):>9} {str(r.get('best', '')):>9} {evs:>9} "
              f"{str(r.get('ms_mean', '')):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
