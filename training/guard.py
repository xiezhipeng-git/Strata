"""训练护栏的纯逻辑 —— 不 import torch，no-torch CI 档直接可测。

两件护栏对应同一类事故：**训练 10 条同构样本，无关能力跟着坏**（2026-10-30 实测：
8 epochs 后算术 holdout 答错、无关提问被插名）。归因是「梯度把共享投影往捷径方向拖」，
护栏从两头掐：

1. **token 选择**（`select_mask`）：只有基座觉得意外（ref CE 高）的 token 才进 loss。
   基座本来就会的 token 拿不到梯度 —— 它们不需要学，学它们只会通过共享投影产生连带位移。
   这是 Rho-1（选择性语言建模）的思路用在 RLHF 数据上的版本。
2. **能力门**（`gate_verdict` / `should_hard_stop`）：训练中周期性评估
   「身份学会多少（id_delta）」与「能力漂移多少（cap_drift）」，二者一起决定：
   - reject：能力越界（cap_drift > tol）—— 这个状态不配做 best；
   - accept：身份比已知最好更好且能力没越界 —— 快照参数（keep-best）；
   - hold：没有更好 —— 不动。
   cap_drift 超过硬顶（cap_stop）就立即停训，最后恢复 best 快照再存盘。

阈值语义都是**严格大于**：边界值不选、不越界。选择零个 token 的样本整条跳过，
不算进 loss、也不算进 early-stop 的滑动窗口（否则 loss=0 会假触发早停）。
"""

from __future__ import annotations


def select_mask(ref_ce: list[float], thresh: float, first: int = 0) -> list[bool]:
    """ref CE 高于阈值的 token 才进 loss（严格大于；空列表返回空）。

    ref_ce 是**训练起点策略**（LoRA 即 B=0 的基座；UniLoRA 是初始随机扰动下的模型）
    对答案段每个 token 的交叉熵。阈值单位 nats：1.0 ≈ 基座给这个 token 的概率低于 e^-1。

    `first=N`（N>0）再砍一刀：**每条样本只留前 N 个选中的 token**。这是「闸门课程」。

    判据（2026-11-14 实测，`idC` 3000 步）：`thresh=1.0` 时 233/505 个 token 进 loss，
    其中真正卡住生成的是追加段的**第一个** token（在「…大型语言模型。」之后接「当前」，
    当时测得 θ₀ CE 12.3；后来修掉英文行的合并 token 后，12 条闸门是 6.05–19.43、
    中位 10.87）—— 它只占选中量的 5%。零阶每步只有一个标量 pj，于是那 95% 的
    「闸门之后」的 token 主导了方向：追加段均值 5.30 → 3.90，而 12 条闸门的 CE
    中位数 **8.70 → 9.08（反而升了）**，生成逐字未变。

    更本质的是：闸门没开时，生成路径**走不到**闸门之后的那些位置，那部分梯度是纯浪费。
    所以只对「生成时真正会访问到、且当前答错」的头几个 token 求导。
    """
    mask = [c > thresh for c in ref_ce]
    if first and first > 0:
        kept = 0
        for i, on in enumerate(mask):
            if on:
                kept += 1
                if kept > first:
                    mask[i] = False
    return mask


def select_all(ref_ce: list[float], thresh: float = 1.0) -> tuple[list[bool], list[int]]:
    """「全段进 loss + 难位置加权」：mask 全 True，返回 `(mask, hot)`。

    `hot` 是 ref CE 严格大于 `thresh` 的下标，交给 `span_weights(mask, hot, W)` 加权。

    为什么需要这个口径（2026-11-15 实测）：
    - `excess`（只留 CE>1.0 的 31/530 = 5.8%）跑 2000 步：answer-CE 0.3024→0.4894，
      生成变长成「通过不断的技术创新与优化…」—— 没被选中的 499 个位置无人约束，
      更新把它们一起带跑。
    - `off`（全段等权）跑 12000→20000 步：CE 卡在 0.27~0.29 不动，句号后第一个 token
      的 logp 只从 −1.93 挪到 −1.90，而 EOS 从 −2.24 抬到 −1.75 —— 难位置被 500 个
      便宜 token 稀释到 1/530 的零阶差分权重，压不动。
    两者交集就是这里：每一位都有梯度（不漂），难位置权重高（不被摊薄）。
    """
    n = len(ref_ce)
    hot = [i for i, c in enumerate(ref_ce) if c > thresh]
    return [True] * n, hot


def select_novel(ref_ce: list[float], mass: float = 0.85, k: int = 0,
                 grow: int = 0, fill: int = 0, thresh: float = 0.0) -> list[bool]:
    """按「起点最意外」选 token —— 与内容无关的通用口径（好奇心 / 差异最大化）。

    和 `select_mask` 的区别是**排序**而不是阈值：

    - `select_mask(thresh)` 只看「够不够意外」，选中的量由数据自己决定 —— 身份数据上
      25 个 token 全进 loss，目标词那 5 个只占 20%，一步一个标量 pj 的预算被摊薄
      （2026-11-15 实测 1500 步：answer-CE 3.29→2.67、探针 +15.6 nats，生成改的却是
      「独立开发→自主研发」这类前缀同义词 —— 梯度全花在便宜的地方）。
    - 这里按 ref CE **降序**累加，只留贡献了 `mass` 比例总 surprise 的那些位置。
      模型本来就会的 token（CE≈0）永远进不去，不管它是「的」「，」还是一个名字。
      这不是身份专用：换数学题、代码、工具轨迹，同一行代码选出的都是「模型最不会的那段」。

    `fill=G` 把**相邻两个选中位置之间**的空隙填满（间距 ≤ G+1 时），但不向簇外扩。
    这是关键的一刀：差异大的位置往往成片出现，而片内会夹着几个模型已经会的过渡 token
    （2026-11-15 实测 `xiezhipeng-git` 的 5 个 token里 `eng` 的 ref CE 只有 0.09，
    因为「ie(z)」之后接「ng」几乎是必然）。`grow` 想覆盖它们就会把簇外的低 CE 前缀
    一起拖进来；`fill` 只补内部，既不让链断开，也不让预算漏到前缀上。
    `grow=R` 是向两侧各扩张 R 位（默认 0，一般不需要）。
    `k>0` 再叠一层硬上限（默认 0=只按 mass 截）。
    `thresh>0` 先滤掉「起点就对」的位置（与 `--select-thresh` 同义）。
    """
    n = len(ref_ce)
    idx = [i for i, c in enumerate(ref_ce) if c > thresh]
    if not idx:
        return [False] * n
    idx.sort(key=lambda i: (-ref_ce[i], i))
    if k and k > 0:
        idx = idx[:k]
    elif mass > 0.0:
        tot = sum(ref_ce[i] for i in idx)
        if tot <= 0:
            return [False] * n
        acc, keep = 0.0, []
        for i in idx:
            keep.append(i)
            acc += ref_ce[i]
            if acc >= mass * tot:
                break
        idx = keep
    mask = [False] * n
    for i in idx:
        mask[i] = True
    if fill and fill > 0:
        s = sorted(idx)
        for a, b in zip(s, s[1:]):
            if 1 < b - a <= fill + 1:
                for j in range(a + 1, b):
                    mask[j] = True
    g = max(0, grow)
    if g:
        for i in idx:
            for j in range(max(0, i - g), min(n - 1, i + g) + 1):
                mask[j] = True
    return mask


def select_cluster(ref_ce: list[float], extra: int = 1, ctx: int = 1) -> list[bool]:
    """**最大的那一个差异簇** + 若干个「段外独立高地」——通用，与内容无关。

    为什么不是「所有 CE 高的点」（`select_novel` 的 mass 口径）：那会把预算摊到多个簇上。
    2026-11-15 同一份身份数据、同一起点、同为 1500 步：

    | 口径 | 进 loss 的位置 | 结果 |
    |---|---|---|
    | 闸门 + 目标词（人工指定，8 位） | 8 | probe +34.3 nats，core **1/5** |
    | surprise mass .45 + fill 3（12 位） | 12 | 0/5 |

    差别不在位数，在**这些位是不是同一件事**：把版本名、项目名一起塞进来，一步一个
    标量 pj 就被三件事分裂了。所以先回答「哪个区域最值得学」，再单独处理「还有没有
    一堵孤立的墙」。

    中心化再求最大子段（Kadane）：ref CE 全是正数，不减去逐行均值的话「最大子段」
    恒等于整个答案段。减去均值之后，只有持续高于本题平均难度的区域才会累积正分，
    边界也就落在真实的事实段落上。

    `extra`：主簇之外，按 ref CE 从高到低补几个孤立高点（每个 ±`ctx` 给一点上下文）。
    身份数据上主簇是「，由 xiezhipeng-git 通过 strata」（因为名字和项目名相邻），
    `extra=1` 补的正是追加段第一个 token「当前」—— 它是一堵孤立的高墙，不补就永远
    开不了口。
    """
    n = len(ref_ce)
    if n == 0:
        return []
    mean = sum(ref_ce) / n
    best, cur, s, bs, be = float("-inf"), 0.0, 0, 0, 0
    for i, c in enumerate(ref_ce):
        w = c - mean
        if cur <= 0.0:
            cur, s = w, i
        else:
            cur += w
        if cur > best:
            best, bs, be = cur, s, i
    if best <= 0.0:                     # 完全均匀：退化成单点
        bs = be = max(range(n), key=lambda i: ref_ce[i])
    mask = [False] * n
    for j in range(bs, be + 1):
        mask[j] = True
    extra = max(0, extra)
    c = max(0, ctx)
    if extra:
        added = 0
        for i in sorted(range(n), key=lambda i: -ref_ce[i]):
            if added >= extra:
                break
            if mask[i]:
                continue
            for j in range(max(0, i - c), min(n - 1, i + c) + 1):
                mask[j] = True
            added += 1
    return mask


def pick_layers(layers: list[int], zo_layers: int = 0,
                zo_range: str = "") -> tuple[list[int], str]:
    """从全层里选出零阶更新的作用层：`--zo-range L0,L1` 区段 / `--zo-layers K` 末 K 层。

    为什么需要「区段」而不只是「末 K 层」：**能改什么由层决定，不由步数决定**。
    2026-11-15 核到的历史配置：`training/common/lora.py:adapter_layout` 每层只挂**一个**
    矩阵（QSA 层 `attn_output`、GDN 层 `ssm_out`，都是把已算好的 hidden 投回残差流），
    而 `--zo-layers K` 只取 `layers[-K:]` ⇒ 这次之前**每一个** run 能动的都只是
    末 1–2 层的输出投影（rank4 时 d=69632，占全模型 rank4 的 3%）。
    输出投影只能给词表加一个「与上下文无关、与位置无关」的常数偏置：这解释得了
    probe logp 一路降（某个 token 的 logit 整体抬高），解释不了「在位置 t 说 X、t+1 说 Y」
    —— 后者要改中间层 MLP 写进 KV 的事实，而那几层从未被放开。
    事实召回集中在中间层，所以先要能选中间层，再谈步数。

    `zo_range` 语法：`"20,34"`（含两端）/ `"20"`（单层）/ `"20,34,4"`（第 20–34 层里每隔
    4 层取一个，用于「大 d 但只放一部分」）。两者都给 ⇒ ValueError，不猜优先级。

    返回 (选中层升序, 一句话人读描述)。
    """
    full = sorted(int(l) for l in layers)
    if not full:
        raise ValueError("layers 为空")
    if zo_layers and zo_range:
        raise ValueError("--zo-layers 与 --zo-range 不能同时给（一个说末 K 层、一个说区段）")
    if zo_layers:
        k = int(zo_layers)
        if k < 0:
            raise ValueError(f"--zo-layers 必须 ≥ 0，收到 {k}")
        if k == 0:
            return full, f"全 {len(full)} 层"
        if k > len(full):
            raise ValueError(f"--zo-layers {k} 超过模型层数 {len(full)}")
        keep = full[-k:]
        return keep, f"末 {len(keep)} 层（{keep[0]}..{keep[-1]}）"
    if not zo_range:
        return full, f"全 {len(full)} 层"
    parts = [p for p in str(zo_range).replace(":", ",").split(",") if p.strip()]
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"--zo-range 要写成 L0,L1 或 L0,L1,步长，收到 {zo_range!r}") from None
    if len(nums) == 1:
        lo = hi = nums[0]
        step = 1
    elif len(nums) == 2:
        lo, hi = nums
        step = 1
    elif len(nums) == 3:
        lo, hi, step = nums
    else:
        raise ValueError(f"--zo-range 最多三段（L0,L1,步长），收到 {zo_range!r}")
    if step < 1:
        raise ValueError(f"--zo-range 步长必须 ≥ 1，收到 {step}")
    if lo > hi:
        lo, hi = hi, lo
    if lo < full[0] or hi > full[-1]:
        raise ValueError(f"--zo-range {lo}..{hi} 越界（模型层 {full[0]}..{full[-1]}）")
    keep = [l for l in full if lo <= l <= hi and (l - lo) % step == 0]
    if not keep:
        raise ValueError(f"--zo-range {zo_range!r} 没选中任何层")
    span = f"{keep[0]}..{keep[-1]}" if len(keep) > 1 else f"{keep[0]}"
    every = "" if step == 1 else f"，每隔 {step} 层"
    return keep, f"{len(keep)} 层（{span}{every}）"


def block_window(layers: list[int], step: int, k: int) -> list[int]:
    """第 `step` 步动哪 `k` 层：在 `layers` 上按步轮转，`k >= len(layers)` 时等于全部。

    为什么要这个：零阶单方向探测的信噪比 ∝ 1/√d（d 是可训维数），而「身份改写要改多少东西」
    又要求覆盖到足够多的层 —— 两者在「一次全动」里直接矛盾，实测也是这个走向
    （2026-11-15 层扫描：末 2 层 400 步 answer-ce −0.59，全 48 层 rank4 只 −0.09，
    全 48 层 rank16 跑 1200 步也只 −0.39，且生成全是「通义千问」）。
    块坐标把矛盾拆开：**每步的 d 小**（方向估得准），**多步轮转**（最终覆盖全层）。

    轮转按 `step * k` 取起点而不是随机抽样：抽样会让同一层连续几步都被动、别的层饿着，
    轮转保证任意连续 `ceil(L/k)` 步里每层恰好被动一次。

    动量在这里会打折：同一层的两次更新相隔 `ceil(L/k)` 步，`momentum ** ceil(L/k)`
    衰减掉。这是块坐标的固有代价，不是 bug —— 想保住动量就调小 k 或调大 momentum。
    """
    full = list(layers)
    if not full or k <= 0:
        return full
    L = len(full)
    if k >= L:
        return full
    start = (int(step) * int(k)) % L
    return [full[(start + i) % L] for i in range(k)]


def find_token_span(texts: list[str], needle: str) -> tuple[int, int] | None:
    """在「逐 token decode 出来的文本」里定位 needle，返回 (首 token, 末 token) 闭区间。

    为什么按文本而不是按 token id 匹配：BPE 会把目标词的首字符和前面的字符合并
    （2026-11-15 实测 `xiezhipeng-git` 单独编码是 `['x','iez','hip','eng','-git']`，
    但在「，由 x」后面第一位是 `' x'`，id 不同），按 id 匹配会整条找不到。
    """
    full = "".join(texts)
    ch = full.find(needle)
    if ch < 0:
        return None
    end_ch = ch + len(needle)
    first = last = None
    acc = 0
    for j, t in enumerate(texts):
        b, e = acc, acc + len(t)
        if e > ch and first is None:
            first = j
        if b < end_ch:
            last = j
        acc = e
    if first is None or last is None:
        return None
    return first, last


def span_mask(n_ce: int, span: tuple[int, int], lead: int = 0) -> list[bool]:
    """把「答案段的 token 区间」换成「CE 数组的下标区间」。

    口径（2026-11-24 与 `answer_seg` 一同修正）：CE 数组现在是**答案 token 对齐**的 ——
    `ce[j]` 就是答案第 j 个 token 的 logp（首位含在内，由 prompt 最后一位预测）。所以
    token 区间 `[i0, i1]` 直接就是 CE 区间 `[i0, i1]`，没有平移。

    `lead` 往前多吃几个 token：只 push 名字本身时，模型并不知道名字前面该接什么，
    所以要带上「，由」这两个 token；lead=0 就是纯名字。
    """
    i0, i1 = span
    lo = max(0, i0 - max(0, lead))
    hi = min(n_ce - 1, i1)
    mask = [False] * n_ce
    for i in range(lo, max(lo, hi) + 1):
        if i < n_ce:
            mask[i] = True
    return mask


def span_weights(mask, hot, weight: float) -> list[float] | None:
    """整句都留在 loss 里，只把 `hot` 那几位乘上 `weight` —— 与 ``seg[sel]`` 对齐。

    为什么不能用截断：2026-11-15 三组对照，同起点同 lr 同墙钟 1500–1800 步：

    | 口径 | answer-CE | 生成 |
    |---|---|---|
    | 整句 25 位（`s2`） | 2.67（−0.62） | 干净，但事实一个字没有 |
    | 闸门+「，由」+名字 8 位（`w1`） | 3.20（−0.08） | `由由由由…` 自环 |
    | 闸门+名字 6 位（`w3`） | 4.54（**+1.27**） | `当前当前…` / `-git-git-git…` |

    窄口径确实把名字的 logp 抬起来了（「，由 _」处 ` x` 从 −7.48 到 −0.52，概率 59%），
    但**没被监督的位置就没有约束**：生成走到 `由` / `当前` / `-git` 之后，argmax 就是它
    自己，于是自环。整句监督能防住自环，却把梯度摊平到 25 个 token 上（事实只占 20%）。

    加权是两者的交集：每一位都有梯度（不会自环），事实那几位权重高（不会被摊薄）。
    权重按 ``Σw`` 归一（`objectives` 里已经这么算），所以乘常数不放大步长。
    """
    n = sum(1 for v in mask if v)
    if n == 0 or weight <= 1.0:
        return None
    hot = set(hot or ())
    w = []
    for i, on in enumerate(mask):
        if on:
            w.append(weight if i in hot else 1.0)
    return w


def gate_weights(n_sel: int, weight: float, gate_n: int = 2) -> list[float] | None:
    """「整句监督 + 闸门加权」的权重，长度 = 选中 token 数（与 ``seg[sel]`` 对齐）。

    与 `select_mask(..., first=N)` 的**截断**是两条路：这里所有选中的 token 都留在
    loss 里，只把**前 gate_n 个**（追加句起头的闸门及其后一位）乘上 `weight`。
    闸门因此能拿到足够的零阶差分权重，而 N+1 之后的 token 仍然有梯度。

    实测依据（2026-11-14）：`select-first 2`（4000 步，idM）把闸门 CE 中位
    10.87→2.82，但生成塌成「当前当前当前…」；rank 4（idR4）闸门全开到 0.14，
    逐 token CE 却是 0.1 / 0.0 / **7.1 / 10.0** —— 切口正好在 N=2 之后。
    自由生成走到第 3 位时那里从来没有 loss，只能回落到刚学会的 token。
    截断是开门/诊断工具，不是学整句的方法。

    `weight <= 0` 或 `gate_n <= 0` 或没选中 token 时返回 None = 不加权的旧口径。
    调用方按 Σw 归一，所以 weight 只改变相对权重、不放大 loss 标量（零阶是纯 SGD，
    loss 标量就是步长；见 LUSPO 的分母事故）。
    """
    if weight <= 0 or gate_n <= 0 or n_sel <= 0:
        return None
    return [float(weight) if i < gate_n else 1.0 for i in range(n_sel)]


def expand_to_labels(sel: list[bool], n_labels: int, plen: int) -> list[bool]:
    """把「答案段每 token 的选择」对齐到 labels 位置（labels = full_ids[1:]）。

    labels 的前 plen-1 个位置是 prompt 段（恒 False），其后逐位对应答案 token。
    长度对不上说明编码/选择不是同一份 —— 抛错而不是带病往下训。
    """
    n_ans = n_labels - (plen - 1)
    if len(sel) != n_ans:
        raise ValueError(f"selection length {len(sel)} != answer tokens {n_ans} "
                         f"(n_labels={n_labels}, plen={plen})")
    return [False] * (plen - 1) + list(sel)


GDN_CHUNK = 64   # fla torch_chunk_gated_delta_rule 的 chunk；qwen4_exp 调用不覆盖，走默认


def split_full(full: list[int], plen: int) -> tuple[list[int], list[int]]:
    """frozen-prefix 反传的序列切分：返回 (prefix, tail)。

    切点 = 最后一个 prompt token（full[plen-1]）**向下对齐到 GDN chunk 边界**。
    两个约束叠加：

    * 梯度约束：full[plen-1]（最后 prompt token）必须在 tail 里 —— 答案首 token
      的 logits 由它的隐状态算出，切过它第一个答案 token 就没有梯度；
    * 数值约束（2026-10-31 diag 实测）：GDN 层的 chunked delta rule 按 chunk
      结合。全序列前向的 chunk 是 [0,64,128,...] 划分，带缓存尾段前向从切点
      重新分 chunk —— 切点不在 64 的倍数上，同一个尾段在两边落在**内容不同
      的 chunk**里，浮点重结合的差异经 48 层累积成 0.1~0.45 nats 的 CE 差。
      对齐后两边每个 chunk 的初始态与内容逐位一致，diag 验证差归零。

    对齐向下取，代价是至多 GDN_CHUNK-1 个 prompt token 进梯度段（无 loss，
    只是带梯度流过），正确性不受影响。切完的几何：

        prefix = full[:s]        # s = (plen-1) // GDN_CHUNK * GDN_CHUNK
        tail   = full[s:]        # [对齐段内 prompt..., 最后prompt token, *答案, EOS]

    tail 里答案 token 的偏移 = (plen-1) - len(prefix)，调用方用它在 tail_logprobs
    的输出里剪出答案段。s 可能为 0（prompt 不足一个 chunk）：prefix 为空列表，
    prefix_cache 返回 None，tail_logprobs 退化为普通全前向。
    """
    if plen < 1 or plen > len(full) - 1:
        raise ValueError(f"plen={plen} 越界（len(full)={len(full)}）："
                         "答案段至少要剩 1 个 token（EOS 也算）")
    s = (plen - 1) // GDN_CHUNK * GDN_CHUNK
    return full[:s], full[s:]


def gate_verdict(objective_delta: float, cap_drift: float,
                 best_objective: float | None, cap_tol: float,
                 gen_ok: bool = True) -> str:
    """能力门的判定：'reject' | 'accept' | 'hold'。

    `objective_delta` 是**本次训练的目标指标**（越大越好），由调用方决定取哪个数：
    身份改写 = 目标词 logp 的增量；事实注入 / 数学题 = heldout CE 的下降量
    （传 `base_held_ce - held_ce`）。门是共用的，目标不是 —— 把目标写死在门里，
    任何非身份任务都会「永远 hold、退回未训参数」，看着像算法无效，其实是判据选错。

    - **生成体检不过就 reject**（gen_ok=False）：teacher-forced 的 cap CE 量不出
      生成退化 —— 2026-10-30 实测，CE 只涨 0.225 nats（容差内），贪心解码却已坍缩成
      「xiezhipeng-gitiezhipeng…」复读环。生成体检（distinct-3gram + EOS）补这个盲区。
    - cap_drift 超过 cap_tol 也 reject —— 哪怕目标学得再好也不配做 best
      （我们不要「学会了目标、坏了能力」的 checkpoint）；
    - 否则目标增益严格大于已知最好才 accept（首个评估 best_objective=None 直接 accept）。
    """
    if not gen_ok or cap_drift > cap_tol:
        return "reject"
    if best_objective is None or objective_delta > best_objective:
        return "accept"
    return "hold"


def should_hard_stop(cap_drift: float, cap_stop: float) -> bool:
    """能力漂移越过硬顶就立即停训（> 严格大于）。"""
    return cap_drift > cap_stop


# ---- 生成体检 ----

GEN_MAX_NEW = 24          # 每条生成探针贪心生成的 token 上限（训练路径 decode ~2 tok/s，别贪长）
GEN_MIN_DISTINCT3 = 0.5   # distinct-3gram 低于它 = 复读环（"ababab" 级别的退化）


def distinct_n(ids: list[int], n: int = 3) -> float:
    """生成序列的 distinct-ngram 占比；短于 n 视为 1.0（没有重复可言）。"""
    if len(ids) < n:
        return 1.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def gen_probe_ok(text_ids: list[int], eos_hit: bool,
                 min_distinct3: float = GEN_MIN_DISTINCT3,
                 max_new: int = GEN_MAX_NEW) -> bool:
    """一条生成是否「形式上没退化」。

    判据分两档，因为「撞窗口」与「复读」是两件事：

    * distinct-3gram 不达标 ⇒ 直接失败。这是复读环的直接指标，也是唯一能抓到
      `We will write We will We need We We We will` 这类中后段崩坏的量。
    * 命中 EOS ⇒ 通过（正常收尾）。
    * 没命中 EOS 但**跑满了窗口** ⇒ 通过：thinking 模型的收尾段在几千 token 之外，
      24 token 的探针永远到不了（2026-11-06 实测 12/12 探针 eos=0、d3=1.0）。
      旧判据要求提前出 EOS，导致 `gen_ok` 恒 False、`gate_verdict` 永远 reject、
      keep-best 从未 accept 过任何 checkpoint —— 崩了也没人回退，等于没有保护网。
    * 没命中 EOS 且**没跑满窗口**（生成器提前停了）⇒ 失败：提前停又不是收尾，
      说明落到了异常 token 上。
    """
    d3 = distinct_n(text_ids, 3)
    if d3 < min_distinct3:
        return False
    if eos_hit:
        return True
    return len(text_ids) >= max_new


def no_word_leak(text: str, word: str) -> bool:
    """cap 探针的生成文本里不允许出现身份词（内容级检查，形式检查的补集）。

    v5-off 实测的洞：step 16 的 cat 探针生成 `'xiezhipeng-git xiezhip…'` ——
    distinct3=0.8、有 EOS，形式全过，但 cat 的问题答出了名字。形式体检量不出这个。
    """
    if not word:
        return True
    return word not in text
