"""投影采样（projection sampling）—— 把 SFT 数据「搬到」模型自己会说的话上。

来源：Karan / Chen / Du, *Finetuning with Sampling: SFT Learns Better Than You Think*
(arXiv:2610.02140)，参考实现 `aakaran/finetuning-with-sampling` 的 `boost_math.py`。
**变的是数据，不是目标函数**：交叉熵、学习率、LoRA 挂点一个字都不改，改的只是
「专家那条轨迹该长什么样」。

## 为什么这条线对我们对症

反传 SFT 在这批数学数据上的实测（2026-11-18，5 组配置）不是「学得慢」，是**方向错了**：
每条样本的 CE 从 3.67 降到 1.77，同一批题的 pass@1 却从 22/24 掉到 21/24、L5 从 6 掉到 3。
CE 降而降不动正确率，说明梯度主要花在「让模型改口说另一种腔调」上 —— 专家轨迹是
MATH 的参考解答，用词、分段、`$...$` 的写法都与基座自己的输出不同。模型学会了那种写法
（答案更短更模板化、eos 从 35/60 拉到 105/108），却没学会解题。

这正是论文的出发点：专家轨迹（off-policy）带着基座拿不到的信息，但它的**分布**离基座太远，
普通的 teacher-forced CE 会把模型往那条远离自身的分布上拉。论文不改 loss，而是把每条
轨迹投影到「与基座 KL 最小、且信息不变」的那一片：

    p_C(x) ∝ p(x) · 1(x ∈ C)

`p` 是基座分布在**同一条提问**上的分布，`C` 是「与专家轨迹等价」的那一族轨迹。实现上用
Metropolis–Hastings：提议核 κ 的构造方式是「把专家解答放进上下文，让基座自己接着写」，
于是每条候选天然落在 `C` 里（信息保持），而似然一步比一步高（KL 单调不增）。

## 落到我们这套引擎上的三个具体决定

1. **评价口径 p(x|q) 用「提问」当上下文，不用提议模板。** 论文的 `get_logprobs_vLLM` 也是
   这样：taken 的是基座在**原始提问**下对这条轨迹的似然。提议模板里塞了解答，那是 κ 的
   上下文，不是 p 的上下文。两者混起来就等于拿「有答案提示的似然」去比「没提示的似然」。
2. **接受判据用「均值 logprob 变大就换」**（论文 App. C.2 明说这是省一次前向的近似，比完整
   MH 更激进地往高似然区走，但 Fig. 5 显示 KL 仍单调下降）。逐 token 求和会让「更长的轨迹」
   系统性吃亏，所以两边都取**平均**。
3. **等价类靠答案判据兜底。** 论文的提议模板里写着「final answer exactly matches」，
   实测仍有 ~5% 生成判错；我们这边直接用 `bench/math_extract.answer_hit`（全项目唯一判据）
   复核一遍，判错就**退回原轨迹**并在记录里标 `fallback`。宁可少一条 boost，也不许把
   一条答错的轨迹当训练目标 —— 那是把「SFT 学会答错」写进数据里。

## 开关

`training/backward/sft.py --data-boost auto|on|off`（默认 auto）只负责**读**已经 boost 过的
数据；生成 boost 数据是这一步（`data_boost.py`）。分开的理由是显存：boost 要独占一张卡跑
引擎，训练也要独占，两者不能同时在卡上 —— 见 MEMORY 的「GPU 单任务铁律」。
同一条链路由 `training/backward/pipeline.py --boost` 串起来。

    python training/backward/data_boost.py --data train.jsonl --out train.boost.jsonl \
        --engine build-v0213-sync/strata.exe --pack <pack> --gguf <model.gguf> \
        --expert-profile data/expert-profile.bin --tokenizer <pack>/tokenizer

    python training/backward/data_boost.py --data train.jsonl --dry-run   # 只量原轨迹有多 off-policy

## 为什么 blocks=1（以及为什么默认从空串起跑）

论文用 `B=32, T=1856, N=10`。我们的轨迹只有 150–450 token，照抄会出现两个实测问题
（2026-11-18，同一批 3 条 easy 数据）：

* `blocks=8 / jump=96` 时，每段 `GEN` 都要从「解答 + 空 partial」重新起手，于是同一句
  `We need answer user's request...` 被写了 8 遍，拼出 768 token 的复读文本 —— 判分居然
  还能过（`answer_hit` 取最靠后的标记），但那是**不能拿去训练的数据**。`blocks=1` 一次
  连续生成，复读消失。
* 反过来从专家轨迹起跑（`--start expert`）+ MCMC 编辑，24 步里接受 **0** 次：提议模板要求
  「用自己的话写」，模型的实际行为是**逐字照抄**，抄出来的东西似然当然不变 ⇒ 没有可爬的坡。
  所以默认走冷启动；`--start expert` 保留给「只要轻微改写」的场合。

单块冷启动的实测（3 条）：`lp −1.09 → −0.45`（+0.64 nats/token），4.5–11.6 s/条，
3/3 判对。24 条约 3 分钟 —— 这就是一次性的预处理成本。

## 怎么区分「这份数据到底增强过没有」

三个标记缺一不可。理由很实际：boost 有成本（独占卡 3 分钟/24 条）、有风险（生成的轨迹判错、
复读、越改越远），而它的产物长得和原始数据几乎一样——`{prompt, boosted}` 与
`{prompt, solution}` 在训练脚本眼里都是「一条提问 + 一段解答」。一旦混了，A/B 的两组就再也
分不开，而且会反复花掉卡去做没有坡可爬的二次投影。

1. **产物自带身份，且拒绝二次投影。** 产物第一行是 `_meta`，之后每行都有
   `boosted` / `source` / `lp_before` / `lp_after` / `changed`；原始训练行只有
   `prompt` / `solution` / `gold`。`detect_provenance()` 据此判 `raw` / `boosted` / `mixed`。
   `data_boost.py` 在 `boosted` 输入上**直接退出**（`--allow-reboost` 才放行）：二次投影是把
   「模型自己的改写」再改写一遍，lp 已经在 −0.5 附近、没有坡可爬，每改一次都在丢信息。
2. **逐条判「有没有真的变」。** `changed` 为真要求同时满足三条：归一化后文本不同、与专家的
   字符相似度 < 0.98、平均 logprob 提升 ≥ `--min-effect`（默认 0.02 nats/token）。任一条不满足
   就把目标**退回原轨迹**并记 `noop` 原因（`identical` / `near_identical` / `tiny_gain`）——
   「改了但几乎没改」不该占用 `boosted` 这个名字，更不该被算进「本轮数据增强过」。
3. **训练日志自报身份。** `sft.py --data-boost` 打印
   `已增强 N/M（实质变更 K，等价保留 J，退回 F）`；`--data-boost off` 而旁边躺着产物时明说
   「未使用」。跑完一轮不看 loss 先看这一行——它决定这轮能不能和上一轮比。
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]          # training/
for _p in (_ROOT, _ROOT / "common", _ROOT / "bench", _ROOT.parent / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 答案判据全项目只有这一份（`sft.py` / `math_probe` 也是它）。用裸模块名 import 是为了
# 与 `sft.py` 里的 `from math_extract import answer_hit` 落到**同一个**模块对象上 ——
# `bench.math_extract` 与 `math_extract` 会各自加载一次，虽然函数是纯的，但那是没必要
# 埋下的第二个身份。
from math_extract import _ANSWERED, _BOXED, answer_hit  # noqa: E402


def _last_marker_span(text: str) -> int | None:
    """文本里最后一个答案标记的**起始下标**；没有就 None。

    刻意复用 `math_extract` 的那两个正则而不是另写一套：判分与「答案行从哪儿开始」必须是
    同一个定义，否则护住的尾巴和判分读的位置可能差一个 `$`（`\\boxed{5}` 里外各一份）。
    """
    s = text or ""
    cands = [m.start() for m in _ANSWERED.finditer(s)]
    cands += [m.start() for m in _BOXED.finditer(s)]
    return max(cands) if cands else None


# 「几乎没改」的相似度门槛。0.98 是刻意取高的：轨迹动辄 800 字符，换一种写法早就低于它；
# 而只把 `，` 换成 `,`、多两个空格这类排版级改动会轻松超过 0.98。
SAME_RATIO = 0.98

# 绝对改动量判据：对不上的字符 ≤ max(NEAR_ABS_MIN, NEAR_REL × 长度) 也算「只动了排版」。
NEAR_ABS_MIN = 4
NEAR_REL = 0.01

# 「算不算真增强」的增益门槛（nats/token）。实测 24 条的逐条增益最小是 0.184，所以 0.02 不会
# 误杀正常结果；而它挡掉的是「严格大于 0 就接受」放进来的浮点噪声级增益。
DEFAULT_MIN_EFFECT = 0.02


def _norm(text: str) -> str:
    """只折叠空白，**不**折叠标点、大小写、`$`、花括号。

    「等价改写」和「无意义改写」的区别就在这里：多几个空格、换行不同是排版，不是增强；
    而 `5` 换成 `5.0`、`\boxed{5}` 换成 `Answer: 5` 都是真变化，不能一起折叠掉——折叠得
    太狠会把「只是重打了空格」判成 changed，那正是要防的那种无意义增强。
    """
    return " ".join((text or "").split())


def _sha8(text: str) -> str:
    """原文的短指纹，只用来**校验配对**，不做安全用途。"""
    return hashlib.sha256((text or "").encode()).hexdigest()[:8]


def _similar(a: str, b: str) -> float:
    """两段文本的字符相似度（0–1）。粗判够用，且对几百字符的轨迹开销可忽略。"""
    na, nb = _norm(a), _norm(b)
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _diff_chars(a: str, b: str) -> int:
    """两段文本**有几个字符对不上**（不相等的 opcode 里取较长的一边）。"""
    na, nb = _norm(a), _norm(b)
    sm = difflib.SequenceMatcher(None, na, nb)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes()
               if tag != "equal")


def _near_identical(a: str, b: str) -> bool:
    """「只动了排版」→ True。

    单看相似度不够：短文本（`Answer: 4` 这种只有几十字符）换个标点、改个连接词，
    比值就掉到 0.92 了，而它显然没变什么。所以加一条**绝对改动量**判据：
    对不上的字符 ≤ max(4, 1%) 也算没变 —— 800 字符的轨迹里改 8 个字符必然是排版级的。
    两条判据任一成立就算「几乎没改」。
    """
    na, nb = _norm(a), _norm(b)
    if not na and not nb:
        return True
    if difflib.SequenceMatcher(None, na, nb).ratio() >= SAME_RATIO:
        return True
    return _diff_chars(na, nb) <= max(NEAR_ABS_MIN, int(NEAR_REL * max(len(na), len(nb))))


def _effect(expert: str, boosted: str, lp_before: float, lp_after: float,
            min_effect: float) -> tuple[bool, str]:
    """这次投影算不算「真的增强」→ `(changed, noop_reason)`。

    三条判据是并联的，任一条不过就算 noop（reason 取第一个命中的，便于统计归因）：

    * `identical`      —— 归一化后逐字相同。模型把专家轨迹照抄了一遍（`--start expert` 下
      的常态），似然不变，等于没做。
    * `near_identical` —— 只动了排版/标点/连接词（两条判据见 `_near_identical`）。
      信息与分布都没动，「改写」是假的。
    * `tiny_gain`     —— 增益 < `min_effect`。这类最隐蔽：`lp > lp_cur` 严格大于就接受，
      于是浮点噪声（+1e-6）也能过；平均值报出来是正的，逐条看却是零。
    """
    if _norm(boosted) == _norm(expert):
        return False, "identical"
    if _near_identical(expert, boosted):
        return False, "near_identical"
    if not (lp_after - lp_before >= min_effect):
        return False, "tiny_gain"
    return True, ""

# 提议模板。与论文 `SEQ_PROMPT_MATH` 同一骨架，三处按我们的数据改：
#   * `{FORMAT}` 由题面自己决定收尾格式（题面写「最后一行 `Answer: <integer>`」就照那个），
#     **不写死 `\boxed{}`** —— 模板若强制换一种收尾，就是又教了模型一种漂移，而这次要做的
#     恰恰是消除漂移；
#   * 明确「partial 可能是空的，那就从头写」：我们的 block 循环第一段 RESPONSE 就是空串，
#     实测模型看到「解答已经在上面」会直接吐 EOS（1 个 token 就停），整条轨迹作废；
#   * 「不要逐字抄解答」+「必须写出收尾行」两句都要留：只有前者会漏掉 `Answer:` 行（判分
#     读不到），只有后者会鼓励照抄（那样 KL 一点没降，boost 等于没做）。
SEQ_PROMPT = """You are an expert mathematician given a math problem, its solution, and an initial, partial response. Carefully study the solution, identifying what reasoning or steps are already provided, and then continue the partial response. Ensure your response is logically consistent with the solution and leads to a complete and correct final answer.

Task:

{PROBLEM}

This is an example solution to the problem:

{SOLUTION}

Here is a partial response:

{RESPONSE}

Starting with the partial response, continue the response in your own words, including the thinking process. Do not copy the wording of the example solution verbatim; express the same reasoning your own way. Ensure the final answer exactly matches that of the provided solution. The response MUST end with the required final-answer line, because only that line is graded: {FORMAT}
"""

# `partial` 为空时不能沿用上面那个模板。实测（2026-11-18）：上下文里已经放着完整解答、
# 又要求「继续这段（空的）partial」，模型会判定无事可做，**第一个 token 就是 `<|im_end|>`**
# —— 3 次试跑里 2 条这样，`tok 1` 的轨迹当然判不到答案。
# 所以 k=0 这一支换成「照着解答自己写一遍完整解答」：同一件事（信息保持），但不再有
# 「继续一个不存在的东西」这个自相矛盾的指令。
SEQ_PROMPT_FRESH = """You are an expert mathematician. Below is a math problem and an example solution to it. Study the example solution, then write your own complete step-by-step solution to the problem. Use your own wording instead of copying the example. Your final answer must match the example solution exactly, and your response MUST end with the required final-answer line, because only that line is graded: {FORMAT}

Problem:

{PROBLEM}

Example solution:

{SOLUTION}

Your solution:
"""


def format_hint(prompt: str) -> str:
    """题面要求的收尾格式 —— 从题面本身抄，别在模板里另立一套。

    题面写「最后一行恰好是 `Answer: <integer>`」就要求那一行；题面什么都没说就要求
    `\\boxed{}`（MATH 的标准收尾，`math_extract` 两种都认）。**两者都不许省**：轨迹没有
    收尾行 ⇒ `answer_hit` 判不到 ⇒ 这条 boost 白跑（试跑第一版 2/2 全废就是这么来的）。
    """
    p = (prompt or "")
    for line in p.splitlines():
        low = line.lower()
        if "answer" in low and ("last line" in low or "exactly" in low or "form" in low):
            return f"end with a line of exactly the form the task asks for ({line.strip()})"
    return r"present the final answer within \boxed{}"


@dataclass
class BoostCfg:
    """一次投影采样的参数。默认值按我们的成本尺子定（内存 `速度尺子`）。"""

    blocks: int = 1          # 分几段自回归扩展。默认 1（见下方「为什么 blocks=1」）
    steps: int = 4           # MH 步数（论文 10；每步一次全序列 SCORE，最贵）
    max_new: int = 700       # 轨迹 token 上限（安全阀，正常由模型自己写 EOS 收尾）
    temperature: float = 0.6  # 提议的采样温度（论文 0.6）
    require_hit: bool = True   # 生成轨迹必须判对才收（等价类兜底）
    min_gain: float = 0.0      # 接受所需的最小平均 logprob 提升（nats/token）
    min_effect: float = DEFAULT_MIN_EFFECT  # 判「算不算真增强」的增益门槛（**与 min_gain 不同**）
    seed: int = 0
    start: str = "empty"       # empty = 从空串生成整条（论文原样）；expert = 从专家轨迹起跑

    def jump(self) -> int:
        return max(1, self.max_new // max(1, self.blocks))


@dataclass
class BoostRow:
    """一条数据的 boost 记录。`text` 是最终要拿去训练的目标。"""

    text: str
    source: str = "boosted"          # boosted / original（fallback 或 noop）
    fallback: str = ""               # 退回原轨迹的原因（判错/空生成/超预算 —— 都是**失败**）
    noop: str = ""                   # 跑成功了但没有实质变化的原因（**不是失败**，是白跑）
    changed: bool = False            # 目标文本与专家轨迹是否**实质不同**（见 _effect）
    generated: str = ""              # 模型实际写出来的那条（fallback/noop 时 text 是原轨迹，这条留着排错）
    lp_before: float = float("nan")  # 原轨迹在 p(x|q) 下的平均 logprob
    lp_after: float = float("nan")   # boost 结果的平均 logprob
    accepts: int = 0
    attempts: int = 0
    hit: bool = False
    n_tok: int = 0
    secs: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def gain(self) -> float:
        return self.lp_after - self.lp_before


def _mean_lp(eng, prefix: list[int], trace: list[int]) -> float:
    """基座在 `prefix` 之后写 `trace` 的平均 logprob（nats/token）。

    切片口径与 `common/encode.py` 同一个：引擎 `SL[t] = log p(ids[t+1])`，所以 `trace`
    第一个 token（绝对下标 `len(prefix)`）的 logprob 落在 `slp[len(prefix)-1]`。
    两边各写一遍「−1 还是 0」正是探针读数对不上的典型来源。
    """
    if not trace:
        return float("nan")
    slp, _ms = eng.score(list(prefix) + list(trace))
    lo = max(len(prefix) - 1, 0)
    seg = slp[lo:lo + len(trace)]
    if len(seg) != len(trace):
        raise RuntimeError(f"SCORE 长度不符：位置 {len(seg)} vs trace {len(trace)}"
                           f"（prefix {len(prefix)}，SL 共 {len(slp)}）")
    return float(sum(seg)) / len(seg)


class Booster:
    """把「引擎 + tokenizer + 渲染器」包成投影采样要用的两个动作。

    只依赖 `gen(ids, max_new)` 与 `score(ids)`，与 `common/runtime.py` 三种 transport 同接口，
    所以既能用独立引擎（pipe），也能用 `--serve-url` 的 training session。
    """

    def __init__(self, eng, tok, render, cfg: BoostCfg, eos: int | None = None):
        self.eng = eng
        self.tok = tok
        self.render = render
        self.cfg = cfg
        self.eos = eos
        self.rng = random.Random(cfg.seed)

    # ---- 两个原语 -------------------------------------------------------
    def prompt_ids(self, q: str) -> list[int]:
        """一条「用户提问」→ prompt ids（**必须过渲染器**）。"""
        return list(self.tok.encode(self.render(q), parse_special=True))

    def _ids(self, text: str, *, special: bool = True) -> list[int]:
        """`special=True`：这段文本是**渲染过的 prompt**（含 `<|im_start|>` 等），必须按特殊
        token 编；`special=False`：这是模型要写的正文（专家轨迹），与 `sft.py` 的
        `tok(a, add_special_tokens=False)` 同一个口径 —— 正文里出现 `<|xx|>` 字样不能变成特殊 token。"""
        return list(self.tok.encode(text, parse_special=bool(special)))

    def propose(self, question: str, solution: str, partial: str, ext_len: int) -> tuple[list[int], str]:
        """κ：把解答放进上下文，让基座接着 partial 往下写 `ext_len` 个 token。

        **提议 prompt 也必须 `render()`**。2026-11-18 试跑踩到的坑：直接 `tok.encode(text)`
        编出来的是一段没有 `<|im_start|>assistant` 回合的裸文本，模型看到「话已经说完了」，
        第一个 token 就吐 `<|im_end|>` —— 8 条样本全部 `tok 1`、全部判错，而日志里只看到
        「生成失败」，看不出是渲染契约漏了一步。同一段文本经 `render` 后是 283 token，
        生成 96 token 一切正常。

        返回 `(新 token, finish)`。`finish` 是引擎 DONE 行的收尾原因（`stop` = 自己写出了
        `<|im_end|>`，`length` = 撞上 ext_len）—— 轨迹该有多长由模型自己定，撞 `length`
        的尾巴是「话没说完」，拿它判分只会得到一条永远不含 `Answer:` 的废样本。
        """
        if ext_len <= 0:
            return [], ""
        text = (SEQ_PROMPT if partial else SEQ_PROMPT_FRESH).format(
            PROBLEM=question, SOLUTION=solution, RESPONSE=partial,
            FORMAT=format_hint(question))
        out, _ms = self.eng.gen(self._ids(self.render(text), special=True), ext_len)
        return [int(t) for t in out], str(getattr(self.eng, "last_finish", "") or "")

    def mean_lp(self, prefix: list[int], trace: list[int]) -> float:
        return _mean_lp(self.eng, prefix, trace)

    # ---- 一条样本 -------------------------------------------------------
    def _tail_guard(self, trace: list[int]) -> int:
        """末尾「答案行」占多少 token —— MCMC 编辑点不许落进去。

        为什么必须护住（2026-11-18 实测）：`Answer: N` 这一行是**等价类的见证**，也是判分
        唯一看的地方。让它成为编辑目标只有一种结果 —— 被换掉、判分失败、整条白跑。而它
        本身是固定格式（不是散文），重写它没有任何「更像自己说的话」的收益。护住之后
        编辑只作用于推理过程，正是要投影的那一段。
        """
        text = self._dec(trace)
        got = _last_marker_span(text)
        if got is None:
            return 0
        tail = text[got:]
        return len(self._ids(tail, special=False)) if tail else 0

    def run(self, row: dict, *, gold: str, requirement) -> BoostRow:
        question = row["prompt"]
        expert = row.get("solution") or row.get("answer") or ""
        if not expert:
            return BoostRow(text="", source="original", fallback="no_expert_trace")

        t0 = time.time()
        prefix = self.prompt_ids(question)
        cfg = self.cfg
        jump = cfg.jump()
        expert_ids = self._ids(expert, special=False)

        if cfg.start == "expert":
            # 从专家轨迹起跑（`--start expert`）。**不是默认值**：实测 24 步接受 0 次 ——
            # 提议模板要求「用自己的话」，模型的实际行为是逐字照抄，抄出来的似然与原轨迹
            # 相等 ⇒ 没有可爬的坡，整条 boost 等于空转。留着它是给「只要轻微改写」的场合。
            trace: list[int] = list(expert_ids)
            lp_cur = self.mean_lp(prefix, trace)
        else:
            trace = []
            lp_cur = float("-inf")

        accepts = attempts = 0
        fin = ""
        for _b in range(cfg.blocks):
            if cfg.start == "empty":
                new, fin = self.propose(question, expert, self._dec(trace), jump)
                if not new:
                    break
                trace += new
                lp_cur = self.mean_lp(prefix, trace)
            for _s in range(cfg.steps):
                if len(trace) < 2:
                    break
                guard = self._tail_guard(trace)
                hi = max(1, len(trace) - guard)
                idx = self.rng.randrange(hi)
                need = len(trace) - idx
                prop_tail, _f = self.propose(question, expert, self._dec(trace[:idx]), need)
                if not prop_tail:
                    continue
                attempts += 1
                prop = trace[:idx] + prop_tail
                lp = self.mean_lp(prefix, prop)
                if lp > lp_cur + cfg.min_gain:
                    trace, lp_cur = prop, lp
                    accepts += 1
            # 冷启动才有「收尾」这回事：模型自己写出回合结束就停；撞 length 是话没说完，
            # 继续下一段。热启动的轨迹一直在显式答案行上，没有这个问题。
            if cfg.start == "empty" and (fin == "stop" or
                                         (self.eos is not None and self.eos in trace)):
                if self.eos is not None and self.eos in trace:
                    trace = trace[:trace.index(self.eos) + 1]
                break

        lp_before = self.mean_lp(prefix, expert_ids)
        text = self._dec(trace)
        hit = answer_hit(text, gold, require_marker=requirement(text))
        changed, noop = _effect(expert, text, lp_before, lp_cur, cfg.min_effect)
        res = BoostRow(text=text, generated=text, lp_before=lp_before, lp_after=lp_cur,
                       accepts=accepts, attempts=attempts, hit=hit, changed=changed,
                       n_tok=len(trace), secs=time.time() - t0)
        # 三级判定，**顺序不能换**：先判「生成的到底对不对」（判错是数据事故，绝不能进训练），
        # 再判「跑了但什么都没换」（白花卡，但不污染数据），最后才是接受。
        if cfg.require_hit and not hit:
            res.text = expert
            res.source = "original"
            res.fallback = "generated_trace_failed_judge"
            res.changed = False
        elif not text.strip():
            res.text = expert
            res.source = "original"
            res.fallback = "empty_generation"
            res.changed = False
        elif not changed:
            # 判对了、也跑完了，但目标与专家轨迹实质相同 —— **退回原轨迹**。
            # 为什么不保留：既没有收益（似然没动），又带进来一次额外的模型改写（可能引入
            # 复读、格式漂移）。这类行必须与 `fallback` 分开统计，否则 `boosted 24/24` 会
            # 把「一条都没变」也报成成功。
            res.text = expert
            res.source = "original"
            res.noop = noop
        return res

    def _dec(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return self.tok.decode(list(ids)) if hasattr(self.tok, "decode") else ""


# ---------------------------------------------------------------- 数据 / 缓存


def _marker_requirement(prompt: str) -> bool:
    """题面要不要 `Answer:` 标记 —— 与 `sft.py --answer-strict auto` 同一条判据：
    题面自己写明「最后一行 Answer:」就必须命中标记，否则回退会变成掷骰子。"""
    p = (prompt or "")
    return ("Answer:" in p) or ("answer:" in p)


def cache_key(cfg: BoostCfg, rows: list[dict], model_tag: str) -> str:
    """boost 结果的指纹：数据内容 + 模型 + 超参。命中就能整段跳过（这是「一次性成本」的落地）。"""
    h = hashlib.sha256()
    h.update(json.dumps({"model": model_tag, "cfg": asdict(cfg)},
                        sort_keys=True, ensure_ascii=False).encode())
    for r in rows:
        h.update((r.get("prompt", "") + "\x00" + (r.get("solution") or r.get("answer") or "")
                  + "\x00" + str(r.get("gold", ""))).encode())
        h.update(b"\x01")
    return h.hexdigest()[:16]


def boost_rows(rows: list[dict], booster: Booster, *, gold_of, log=print,
               budget_secs: float = 0.0, resume: list[dict] | None = None,
               show: int = 0) -> tuple[list[dict], dict]:
    """逐条 boost。超出 `budget_secs` 就停，剩下的行原样保留并标 `skipped_budget`。

    「跑不完就少 boost 几条、不要整段失败」是刻意的：boost 是一次性预处理，中断后能从
    cache 续上；而训练脚本等到一半才发现数据没生成好，代价是一整轮实验。
    """
    done = [dict(r) for r in (resume or [])]
    # 续跑可能接的是**旧产物**（没有 `changed` 字段）：不补齐的话，这些行会被 summarize 当成
    # 「没增强」，续跑一次就凭空少报 N 条。补齐用生成时的口径（`source=="boosted"`）。
    for _r in done:
        if "changed" not in _r:
            _r["changed"] = _r.get("source") == "boosted"
    out = list(done)
    t0 = time.time()
    for i, row in enumerate(rows):
        if i < len(done):
            continue
        if budget_secs and (time.time() - t0) > budget_secs:
            for rest in rows[i:]:
                expert = rest.get("solution") or rest.get("answer") or ""
                out.append({"prompt": rest["prompt"], "boosted": expert,
                            "source": "original", "fallback": "skipped_budget",
                            "noop": "", "changed": False,
                            "expert_sha8": _sha8(expert), "expert_len": len(expert)})
            log(f"boost: 预算 {budget_secs:.0f}s 用尽，余 {len(rows) - i} 条原样保留")
            break
        gold = str(gold_of(row))
        r = booster.run(row, gold=gold, requirement=_marker_requirement)
        expert = row.get("solution") or row.get("answer") or ""
        rec = {"prompt": row["prompt"], "boosted": r.text, "source": r.source,
               "fallback": r.fallback, "noop": r.noop, "changed": bool(r.changed),
               "lp_before": r.lp_before, "lp_after": r.lp_after,
               "gain": r.gain, "accepts": r.accepts, "attempts": r.attempts,
               "hit": r.hit, "n_tok": r.n_tok, "secs": round(r.secs, 1),
               "gold": gold, "generated": r.generated,
               # 原文指纹：`apply_to_sft` 靠它确认「这份产物是用**这一份**解答算出来的」。
               # 只比 prompt 不够 —— 同一批题重出了解法时 prompt 一模一样，会静默错配。
               "expert_sha8": _sha8(expert), "expert_len": len(expert)}
        out.append(rec)
        tag = ("changed" if r.changed else ("noop:" + r.noop if r.noop else "fallback"))
        log(f"  [{i + 1}/{len(rows)}] {tag:16s} hit={int(bool(r.hit))} "
            f"lp {r.lp_before:+.3f}→{r.lp_after:+.3f}（{r.gain:+.3f}）"
            f" acc {r.accepts}/{r.attempts} tok {r.n_tok:4d} {r.secs:5.1f}s"
            + (f"  ⚠ {r.fallback}" if r.fallback else ""))
        if show:
            # 排错要看**模型真写出来的那条**，不是退回后的原轨迹（两者在 fallback 时不同）
            flat = " ".join((r.generated or r.text).split())
            log(f"      {flat[:show]}{'…' if len(flat) > show else ''}")
    return out, summarize(out)


def summarize(out: list[dict]) -> dict:
    """一份能直接进报告的账 —— **三类别分开数**，这是「防止无意义增强」的记账口径：

    * `boosted`  —— 真的换了目标（`changed=True`）。只有这一类算「这一轮数据增强过」。
    * `noop`     —— 跑完了但目标实质没变（照抄 / 排版级改动 / 增益低于门槛），已退回原文。
    * `fallback` —— 失败（判错 / 空生成 / 超预算），同样退回原文。

    `fallback` 以前算的是「不是 boosted 的都算退回」，加了 noop 之后那样算会把「一条都没变、
    但也什么都没坏」报成失败。两者混起来，`boosted 0/24 fallback 24` 会被读成「全炸了」，
    而真相可能是「模型照抄了 24 次、数据原封不动」—— 那是完全不同的处置。
    """
    n_boosted = sum(1 for r in out if r.get("source") == "boosted" and r.get("changed"))
    n_noop = sum(1 for r in out if r.get("noop"))
    n_fallback = sum(1 for r in out if r.get("fallback"))
    boosted = [r for r in out if r.get("source") == "boosted" and r.get("changed")]
    gains = [r["gain"] for r in boosted
             if isinstance(r.get("gain"), float) and r["gain"] == r["gain"]]
    gains_sorted = sorted(gains)
    return {
        "rows": len(out),
        "boosted": n_boosted,
        "noop": n_noop,
        "fallback": n_fallback,
        "untouched": len(out) - n_boosted - n_noop - n_fallback,
        "noop_reasons": {k: sum(1 for r in out if r.get("noop") == k)
                         for k in sorted({r.get("noop") for r in out if r.get("noop")})},
        "reasons": {k: sum(1 for r in out if r.get("fallback") == k)
                    for k in sorted({r.get("fallback") for r in out if r.get("fallback")})},
        "hit": sum(1 for r in out if r.get("hit")),
        # 有效增强率：真的换了目标的行 / 总行数。报告里先看这个数，再看 gain。
        "effective_rate": round(n_boosted / len(out), 3) if out else 0.0,
        "lp_before": round(sum(r["lp_before"] for r in boosted) / len(boosted), 4) if boosted else None,
        "lp_after": round(sum(r["lp_after"] for r in boosted) / len(boosted), 4) if boosted else None,
        "mean_gain": round(sum(gains) / len(gains), 4) if gains else None,
        # 平均值会被一条大增益带偏，最小值与中位数才是「有没有白跑」的证据。
        "min_gain": round(gains_sorted[0], 4) if gains_sorted else None,
        "p50_gain": round(gains_sorted[len(gains_sorted) // 2], 4) if gains_sorted else None,
        "mean_accept_rate": round(sum(r["accepts"] / r["attempts"] for r in boosted
                                      if r.get("attempts")) / max(1, len(boosted)), 3) if boosted else None,
        "mean_tok": round(sum(r["n_tok"] for r in boosted) / len(boosted), 1) if boosted else None,
        "secs": round(sum(r.get("secs", 0.0) for r in out), 1),
    }


# ---------------------------------------------------------------- 读取 / 落盘


def read_rows(path: str) -> list[dict]:
    return read_input(path)[1]


def read_input(path: str) -> tuple[dict, list[dict]]:
    """读输入数据 → (meta, rows)。**跳过 `_meta` 行**。

    跳过这一步是真 bug 级的：把 boost 产物当 `--data` 喂进来时第一行是 `{"_meta": ...}`，
    它没有 `prompt`，后面 `row["prompt"]` 直接 KeyError —— 而报错点离原因隔了几十行。
    """
    meta, rows = {}, []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            if i == 0 and "_meta" in obj:
                meta = obj["_meta"]
            else:
                rows.append(obj)
    return meta, rows


def _is_boost_rec(row: dict) -> bool:
    """这一行是不是 boost 产物的记录（而不是原始训练行）。

    判据取「有 `boosted` 字段」+「有 `source` 或 `lp_before`」：原始训练行是
    `{prompt, solution|answer, gold}`，不可能同时带这两个字段；而手工拼的数据里即便有个叫
    `boosted` 的字段，也不会带 `lp_before`。
    """
    return ("boosted" in row) and ("source" in row or "lp_before" in row)


def detect_provenance(rows: list[dict], meta: dict | None = None) -> dict:
    """这份数据增强过没有 —— 训练侧和生成侧都靠它自报身份。

    返回 `state`：
      * `raw`     —— 一条 boost 记录都没有，是原始数据；
      * `boosted` —— 全部都是 boost 记录（正常产物）；
      * `mixed`   —— 一半一半（手工拼过，或者产物没覆盖全批）。**这种情况最危险**：
        拿去训练时一半目标是投影过的、一半是专家原文，两组 A/B 从此不可比。
    """
    hits = [r for r in rows if _is_boost_rec(r)]
    if not hits:
        state = "raw"
    elif len(hits) == len(rows):
        state = "boosted"
    else:
        state = "mixed"
    # 老产物（加了 `changed` 字段之前生成的）没有这个键：退回用 `source` 判，
    # 并把 `legacy=True` 报出去 —— 否则 `changed 0` 会被误读成「全都白跑」。
    legacy = bool(hits) and all("changed" not in r for r in hits)
    if legacy:
        changed = sum(1 for r in hits if r.get("source") == "boosted")
    else:
        changed = sum(1 for r in hits if r.get("changed"))
    return {
        "state": state,
        "rows": len(rows),
        "boost_rows": len(hits),
        "changed": changed,
        "noop": sum(1 for r in hits if r.get("noop")),
        "fallback": sum(1 for r in hits if r.get("fallback")),
        "legacy": legacy,
        "key": (meta or {}).get("key", ""),
        "lp_before": ((meta or {}).get("stats") or {}).get("lp_before"),
        "lp_after": ((meta or {}).get("stats") or {}).get("lp_after"),
    }


def peek_state(path: str) -> str:
    """只看第一行判断这是不是 boost 产物 —— 训练前这道检查不该为了 10 万行再读一遍文件。

    产物第一行必然是 `_meta`（`write_out` 保证），所以这一行的判定就足够可靠：
    拿到 `_meta` 就是产物；拿到一条带 `boosted`/`lp_before` 的行就是手工拼的混合物；
    其余当原始数据。
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "_meta" in obj:
                return "boosted"
            return "mixed" if _is_boost_rec(obj) else "raw"
    return "raw"


def provenance_line(prov: dict, *, verb: str = "data") -> str:
    """一行话，直接打进训练日志 —— 事后对照实验靠它区分两组。"""
    st = prov.get("state")
    if st == "raw":
        return f"{verb}: 原始数据（未增强）"
    tag = "（旧产物，无 changed 字段）" if prov.get("legacy") else ""
    head = "⚠ 混合数据 " if st == "mixed" else ""
    return (f"{verb}: {head}已增强 {prov['changed']}/{prov['rows']} 条{tag}"
            f"　等价保留 {prov['noop']}　退回 {prov['fallback']}"
            f"　key {prov['key'] or '?'}")


def gold_of_row(row: dict):
    """`gold` 优先，退回 `answer`（`answer` 在这批数据里是 CoT，取它会是整段文字）。"""
    if "gold" in row and str(row["gold"]).strip():
        return row["gold"]
    if "answer" in row and str(row["answer"]).strip():
        return row["answer"]
    return ""


def write_out(path: str, rows: list[dict], meta: dict) -> None:
    """落到 `<path>`：第一行是 meta（可读、可做指纹），之后每条一行。

    为什么不是纯 jsonl 数据：拿到一份 boost 数据最先要看的是「这份是哪个模型、用什么参数
    boost 的、多少条退回了」—— 散落在日志里就没人看。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def default_boost_path(data_path: str) -> str:
    """`--data a.jsonl` 对应的 boost 产物路径。约定俗成放在同一目录，训练侧不用额外参数。"""
    return str(data_path) + ".boost.jsonl"


def apply_to_sft(data: list[tuple[str, str]], path: str, *, mode: str = "auto",
                 log=print) -> list[tuple[str, str]]:
    """把 boost 过的目标换进 `sft.py` 的 `[(prompt, answer)]`。

    **prompt 必须逐条逐字相同**才换：boost 产物是按 prompt 索引对上的，若两份数据不是同
    一批题，静默按行号替换就会把 A 题的解答配给 B 题 —— 这种错在 loss 曲线上完全看不出来
    （照样平滑下降），只有生成答案会莫名其妙地对不上题。所以这里对不上就报错退出。

    没找到产物时：`auto` 打一行 WARN 继续用原数据（默认行为，不阻断），`on` 直接退出
    （用户明说要 boost，就不该静默降级）。
    """
    if mode == "off":
        return data
    meta, rows = ({}, [])
    if Path(path).is_file():
        meta, rows = read_out(path)
    if not rows:
        msg = (f"--data-boost {mode}: 没有找到 boost 产物 {path}\n"
               "  先生成（需要独占一张卡跑引擎，训练前把引擎关掉）：\n"
               f"    python training/backward/data_boost.py --data <原始数据> "
               f"--out {path} --engine <strata.exe> --pack <pack> --gguf <model.gguf> "
               "--expert-profile data/expert-profile.bin")
        if mode == "on":
            raise SystemExit(msg)
        log(f"WARN: {msg}")
        return data
    if len(rows) != len(data):
        raise SystemExit(f"{path}: {len(rows)} 条 boost 记录 vs 数据 {len(data)} 条 —— "
                         "两份不是同一批数据（用过 --limit？），拒绝按行号替换")
    for i, ((q, a), r) in enumerate(zip(data, rows)):
        if r.get("prompt") != q:
            raise SystemExit(f"{path}: 第 {i} 条 prompt 与 --data 不一致，拒绝替换\n"
                             f"  数据: {q[:80]!r}\n  boost: {str(r.get('prompt'))[:80]!r}")
        # 同一批题重出了解法时 prompt 一模一样，只比 prompt 会静默把 A 题的投影结果配给
        # B 题的新解法。`expert_sha8` 是产物生成时记下的原文指纹，能挡住这一路。
        want = r.get("expert_sha8")
        if want and _sha8(a) != want:
            raise SystemExit(f"{path}: 第 {i} 条 prompt 对上了、但专家轨迹变了"
                             f"（指纹 {_sha8(a)} vs 产物 {want}）—— 这份产物不是用当前 --data "
                             "算出来的，请重跑 data_boost.py")
    # **替换与否由这里重算，不信产物里的 `changed`。** 两个理由：
    #   * 旧产物根本没有这个字段（`--min-effect` 是后加的），只认字段等于老数据全部不换；
    #   * 产物是被人工改过／拷来拷去的文件，`changed` 可能和后面对不上的文本并存。
    # 重算用的是产物里已经存好的 `lp_before/lp_after`，不需要再碰引擎。
    m_eff = float((((meta or {}).get("cfg") or {}).get("min_effect", DEFAULT_MIN_EFFECT)))
    take: list[bool] = []
    for (q, a), r in zip(data, rows):
        b = r.get("boosted") or ""
        lb, la = r.get("lp_before"), r.get("lp_after")
        if not b:
            take.append(False)                     # 空目标：无脑用原轨迹
        elif lb is None or la is None:
            # 记录不全（超预算、或产物被裁过）：退回生成时的判定字段，别在这里瞎猜
            take.append(bool(r.get("changed", r.get("source") == "boosted")))
        else:
            take.append(_effect(a, b, float(lb), float(la), m_eff)[0])
    out: list[tuple[str, str]] = []
    for (q, a), r, t in zip(data, rows, take):
        out.append((q, (r.get("boosted") or a) if t else a))
    prov = detect_provenance(rows, meta)
    n_take = sum(take)
    log(f"data-boost: 用 {path}（key {(meta or {}).get('key', '?')}）")
    log("  " + provenance_line(prov, verb="数据身份"))
    if meta and "min_effect" not in ((meta.get("cfg") or {})):
        log(f"  note: 产物生成时还没有 min_effect 判定，本次按 {m_eff:g} nats/token 重算")
    if n_take != prov["changed"]:
        log(f"  WARN: 本次重算替换 {n_take} 条 vs 产物记账 changed {prov['changed']} 条 "
            f"（noop {prov['noop']} / fallback {prov['fallback']}）—— "
            "产物可能是旧规则生成的或被人手改过")
    st = (meta or {}).get("stats") or {}
    ch_before = sum(len(a) for _, a in data) / max(1, len(data))
    ch_after = sum(len(b) for _, b in out) / max(1, len(out))
    if st:
        _extra = f"，min {st['min_gain']}" if st.get("min_gain") is not None else ""
        log(f"  平均 p(x|q) {st.get('lp_before')} → {st.get('lp_after')} nats/token"
            f"（{st.get('mean_gain')}{_extra}）  接受率 {st.get('mean_accept_rate')}")
    log(f"  目标长度 {ch_before:.0f} → {ch_after:.0f} 字符/条"
        f"　实际替换 {n_take}/{len(rows)} 条")
    return out


def read_out(path: str) -> tuple[dict, list[dict]]:
    """读回 `write_out` 的产物 → (meta, rows)。与 `read_input` 同一件事，保留名字给调用方。"""
    return read_input(path)


# ---------------------------------------------------------------- CLI


def _sampling_flags(args) -> list[str]:
    """提议温度只能从引擎命令行给（`GEN max_new ids` 没有 per-request 的采样参数）。

    `--seed` 在引擎里同时是「切出 greedy」的开关（`generate.cpp`：`--seed` 那一支会置
    `greedy=false`），所以温度要靠它才生效。**同一个进程只 seed 一次、RNG 随每次 GEN 往前走**
    —— 若哪次发现所有提议逐字相同，那就是 RNG 被按请求重置了，得回到这里查。
    """
    return ["--seed", str(int(args.seed)),
            "--temperature", f"{float(args.temperature):g}",
            "--top-p", f"{float(args.top_p):g}"]


def _build_engine(args, log):
    """起引擎（pipe，独占 GPU）或接已有 serve 的 training session。两路都返回同接口对象。"""
    from common.chat import load_template, load_tokenizer
    from common.runtime import EngineApi, engine_argv, launch_engine

    tok = load_tokenizer(args.pack)
    render = load_template(args.pack, enable_thinking=True, system=args.system_prompt,
                           reasoning_effort=args.reasoning_effort or None)
    # **路径必须先绝对化**：`launch_engine` 用 `cwd=dirname(exe)` 起引擎，所以引擎眼里的
    # `../Strata-data/packs/x` 是相对 build 目录的，不是相对调用者的。`serve.server` 那边
    # 因为 config 里写了 `"cwd": "."` 才没暴露这个差别；pipe 这条路没有 config，只能在这里定死。
    ns = argparse.Namespace(engine=os.path.abspath(args.engine), pack=os.path.abspath(args.pack),
                            gguf=os.path.abspath(args.gguf),
                            expert_profile=os.path.abspath(args.expert_profile)
                            if args.expert_profile else "",
                            max_context=args.max_context,
                            prefill_chunk=args.prefill_chunk, rank=args.rank,
                            alpha=float(args.alpha), seed=args.seed)
    if args.serve_url:
        argv = engine_argv(ns, args.lora, _sampling_flags(args))
        eng = EngineApi(args.serve_url, argv)
        log(f"engine: 走 {args.serve_url} 的 training session（会先卸载推理引擎）")
        return eng, tok, render
    eng, load_s = launch_engine(ns, args.log, args.lora, _sampling_flags(args))
    log(f"engine up in {load_s:.0f}s  ctx {args.max_context}（独占）")
    return eng, tok, render


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="投影采样：把 SFT 数据搬到模型自己会说的话上")
    ap.add_argument("--data", required=True, help="输入 jsonl：{prompt, solution|answer, gold?}")
    ap.add_argument("--out", default="", help="输出（默认 <data>.boost.jsonl）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（先小批试跑）")
    ap.add_argument("--blocks", type=int, default=BoostCfg.blocks)
    ap.add_argument("--steps", type=int, default=BoostCfg.steps)
    ap.add_argument("--max-new", type=int, default=BoostCfg.max_new)
    ap.add_argument("--temperature", type=float, default=BoostCfg.temperature,
                    help="提议采样温度（论文 0.6）；经引擎的 --temperature 生效")
    ap.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    ap.add_argument("--min-gain", type=float, default=BoostCfg.min_gain, metavar="NATS",
                    help="**单步 MCMC 接受**所需的最小平均 logprob 提升（>0 更保守）")
    ap.add_argument("--min-effect", type=float, default=DEFAULT_MIN_EFFECT, dest="min_effect",
                    metavar="NATS",
                    help="判「这条算不算真增强」的增益门槛（默认 0.02）：低于它、或与专家"
                         "轨迹相似度 ≥0.98、或归一化后相同，就标 noop 并退回原轨迹。"
                         "与 --min-gain 不同：那个管单步是否接受，这个管整条有没有意义")
    ap.add_argument("--allow-reboost", action="store_true", dest="allow_reboost",
                    help="允许在已经 boost 过的数据上再跑一遍（默认拒绝：二次投影没有坡可爬）")
    ap.add_argument("--no-require-hit", dest="require_hit", action="store_false",
                    help="不强制生成轨迹判对（默认强制；关掉等于放弃等价类兜底）")
    ap.add_argument("--budget-secs", type=float, default=0.0, metavar="S",
                    help="总时长上限；用尽后剩余行原样保留（0 = 不限）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只量原轨迹在基座下的平均 logprob（不生成），用来看这份数据有多 off-policy")
    ap.add_argument("--start", choices=("expert", "empty"), default=BoostCfg.start,
                    help="empty（默认）= 从空串生成整条，论文原样；"
                         "expert = 从专家轨迹起跑做 MCMC 编辑（实测接受 0 次：提议会逐字照抄，"
                         "似然不动，除非只要轻微改写，否则别用）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=0, metavar="N",
                    help="每行打印生成文本的前 N 个字符（调模板时用）")
    ap.add_argument("--resume", action="store_true", help="从已存在的 --out 续跑")
    ap.add_argument("--force", action="store_true",
                    help="即便 --out 的指纹与本次参数一致也重算（默认命中即复用）")
    ap.add_argument("--stats", default="", help="把统计写到这个 json（默认和 --out 同名 .stats.json）")

    ap.add_argument("--engine", default="", help="strata.exe")
    ap.add_argument("--pack", default="", help="packs/<model>")
    ap.add_argument("--gguf", default="", help="模型 gguf（--native）")
    ap.add_argument("--expert-profile", default="", dest="expert_profile")
    ap.add_argument("--max-context", type=int, default=8192, dest="max_context")
    ap.add_argument("--prefill-chunk", default="auto", dest="prefill_chunk")
    ap.add_argument("--lora", default="", help="空 = 基座（boost 通常就用基座）")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--serve-url", default="", dest="serve_url",
                    help="改成接已有 serve 的 /v1/training/session（会卸载推理引擎）")
    ap.add_argument("--log", default="data_boost.engine.log")
    ap.add_argument("--system-prompt", default="", dest="system_prompt")
    ap.add_argument("--reasoning-effort", default="", dest="reasoning_effort")
    a = ap.parse_args(argv)

    def log(m):
        print(m, flush=True)

    out_path = a.out or (a.data + ".boost.jsonl")
    in_meta, rows = read_input(a.data)
    prov = detect_provenance(rows, in_meta)
    # **先验身份，再干任何活（包括起引擎）**。二次投影是最典型的「无意义增强」：
    # 输入已经是「模型自己写得顺」的那一版，lp 在 −0.5 附近，再投一次只会在模型自己的
    # 改写上再改写一遍 —— 没有坡可爬，每改一次都在丢信息，还要白占一张卡 3 分钟。
    if prov["state"] != "raw" and not a.allow_reboost:
        lp = (f"lp {prov['lp_before']} → {prov['lp_after']}"
              if prov.get("lp_before") is not None else "lp 未知（产物无 _meta）")
        raise SystemExit(
            f"{a.data}: 这份数据**已经是投影采样的产物**"
            f"（{prov['changed']}/{prov['rows']} 条实质变更，{lp}），拒绝二次投影。\n"
            "  二次投影是把「模型自己的改写」再改写一遍：似然已经在 −0.5 附近、没有坡可爬，\n"
            "  每改一次都在丢信息，而且会白占一张卡。\n"
            "  要重做：拿**原始数据**（只有 prompt / solution|answer / gold）并加 --force。\n"
            "  确实要叠加多重投影就加 --allow-reboost（不建议；此时 --dry-run 的读数也无意义）。")
    if prov["state"] == "mixed":
        log(f"WARN: {a.data} 里只有 {prov['boost_rows']}/{prov['rows']} 条是 boost 记录 —— "
            "混合数据训练出来的两组实验不可比，建议先归一成一份")
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        raise SystemExit(f"{a.data}: 空文件")
    cfg = BoostCfg(blocks=a.blocks, steps=a.steps, max_new=a.max_new,
                   temperature=a.temperature, require_hit=a.require_hit,
                   min_gain=a.min_gain, min_effect=a.min_effect,
                   seed=a.seed, start=a.start)
    if cfg.max_new % cfg.blocks:
        log(f"note: max_new {cfg.max_new} 不是 blocks {cfg.blocks} 的整数倍，"
            f"实际每段 {cfg.jump()} tok（共 {cfg.jump() * cfg.blocks}）")

    if not (a.engine and a.pack and a.gguf):
        raise SystemExit("需要 --engine/--pack/--gguf（或用 --serve-url 接已有 serve）")

    done: list[dict] = []
    if a.resume and Path(out_path).is_file():
        _, done = read_out(out_path)
        log(f"resume: 已有 {len(done)} 条，跳过")
    # 指纹命中就直接返回 —— **在起引擎之前**。boost 是一次性预处理，重跑同一份数据
    # 不该再占一次卡（每次 20 s 启动 + 逐条生成）。要重算就 --force。
    key = cache_key(cfg, rows, a.gguf + "|" + (a.lora or "base"))
    if Path(out_path).is_file() and not a.force:
        old_meta, _ = read_out(out_path)
        if old_meta.get("key") == key:
            log(f"boost: {out_path} 已是同一份数据 + 同一套参数的产物（key {key}），直接复用。"
                "要重算加 --force")
            return 0
        log(f"boost: 已有产物 {out_path}，但指纹不符"
            f"（旧 {old_meta.get('key', '?')} vs 新 {key}）→ 重算"
            + ("（旧产物生成时还没有 min_effect 判定）"
               if "min_effect" not in (old_meta.get("cfg") or {}) else ""))

    eng = None
    try:
        eng, tok, render = _build_engine(a, log)
        from common.encode import eos_id as _eos
        booster = Booster(eng, tok, render, cfg, eos=_eos(tok))
        if a.dry_run:
            stats: dict = {"rows": 0, "secs": 0.0}
            lps = []
            t_dry = time.time()
            for i, r in enumerate(rows):
                p = booster.prompt_ids(r["prompt"])
                expert = r.get("solution") or r.get("answer") or ""
                lp = booster.mean_lp(p, booster._ids(expert, special=False))
                lps.append(lp)
                log(f"  [{i + 1}/{len(rows)}] lp(expert|q) = {lp:+.4f} nats/token")
            if lps:
                stats = {"rows": len(lps), "lp_mean": round(sum(lps) / len(lps), 4),
                         "lp_min": round(min(lps), 4), "lp_max": round(max(lps), 4),
                         "secs": round(time.time() - t_dry, 1)}
            log(f"dry-run: {stats}  —— lp 越接近 0 说明专家轨迹本来就在模型的分布里，"
                "boost 没有空间；-1 以下是明显 off-policy，值得投影")
            log("dry-run: 输入已确认是**原始数据**（未增强），这个读数才代表「专家 vs 基座」的距离")
            p_dry = Path(a.stats or (out_path + ".dryrun.json"))
            p_dry.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
            log(f"  -> {p_dry}")
            return 0

        key = cache_key(cfg, rows, a.gguf + "|" + (a.lora or "base"))
        log(f"boost: {len(rows)} 条  blocks={cfg.blocks} steps={cfg.steps} "
            f"jump={cfg.jump()} max_new={cfg.max_new} require_hit={cfg.require_hit}  key={key}")
        out, stats = boost_rows(rows, booster, gold_of=gold_of_row, log=log,
                                budget_secs=a.budget_secs, resume=done, show=a.show)
    finally:
        if eng is not None:
            eng.close()

    meta = {"key": key, "data": a.data, "cfg": asdict(cfg), "gguf": a.gguf,
            "lora": a.lora or "(base)", "stats": stats, "when": time.strftime("%F %T")}
    write_out(out_path, out, meta)
    sp = a.stats or (out_path + ".stats.json")
    Path(sp).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"boost done: {stats['boosted']}/{stats['rows']} 条真增强"
        f"（有效 {stats['effective_rate'] * 100:.0f}%）　无变化 {stats['noop']} {stats['noop_reasons']}"
        f"　退回 {stats['fallback']} {stats['reasons']}")
    if stats["boosted"]:
        log(f"  平均 lp {stats['lp_before']}→{stats['lp_after']}（{stats['mean_gain']}，"
            f"min {stats['min_gain']} / p50 {stats['p50_gain']}）  用时 {stats['secs']:.0f}s")
    log(f"  -> {out_path}")
    log(f"  -> {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
