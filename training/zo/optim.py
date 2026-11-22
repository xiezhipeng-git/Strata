"""零阶（MeZO）估计器 —— 只有「怎么从标量损失得到 Δθ」，别的什么都不做。

边界（2026-11-14 用户要求「各司其职」后立的规矩）：

    这个文件不许出现：数据、chat 模板、tokenizer、引擎、SCORE、评测、
    探针、metrics、lr 调度、早停、存盘。它只认三样东西 ——
    θ（LoRA 因子字典）、一个 `loss_fn(th) -> float` 的标量函数、学习率。

    `loss_fn` 由训练器提供（它知道怎么把 th 写进引擎并 SCORE）；本模块对
    「损失是怎么算出来的」一无所知，对「θ 存在哪」也不关心（纯 numpy 字典）。

数学（MeZO, arXiv 2305.17333，算法 1）：

    z ~ N(0, I)，ĝ = (L(θ+εz) − L(θ−εz)) / (2ε) · z，θ ← θ − lr·ĝ

    dirs=K 时取 K 个独立方向平均（方差降 √K，代价 2K 次前向）；
    momentum 把跨步的 ĝ 累积起来（单方向噪声里那部分「一致分量」才留得下）。

可选的 residual-feedback 单点估计（`one_point=True`，默认关）：

    pj = (L(θ+εz) − L_prev) / ε，  L_prev = 上一步采到的 L(θ_prev+εz_prev)

每步只要 **1 次前向**（首步多一次基线，摊到整个 run 上可忽略），代价是偏差
`(L(θ) − L_prev)/ε` 与更大的方差 —— 它用上一步的扰动点代理本步的中心损失。
这是 SPSA 的 residual-feedback 变体，不是 MeZO 原文的两点估计；开启后 `loss_mid`
报的是扰动点损失而非中心损失。默认关闭，历史数字逐位不变。

结构化方向采样（`subspace=K`，K>0 开启；依据：InferZO/LoZO 与梯度低维性文献）：

    均匀 z 与真梯度 ∇L 的对齐度 ~ 1/√d（d ≈ 1e6 时 ≈ 0.001）—— 范数几乎全
    浪费在无关坐标上。两种模式（`subspace_mode`）：

    **lozo**（默认；InferZO 的方向算法，忠实移植）：

        LoZO: "Enhancing Zeroth-Order Fine-Tuning for Language Models with
        Low-Rank Structures" (ICLR 2025, arXiv:2410.07698)；
        InferZO: "Inference-Native Zeroth-Order Optimization for LLMs"
        (arXiv:2605.28760，官方实现 github.com/playeriv65/InferZO，
        其 `LOZODirectionProvider`/`SubZeroDirectionProvider` 即此处的
        参数空间等价形式)。

        随机正交基 Q ∈ R^(k×d) 初始化生成，每 `subspace_refresh`（ν，LoZO 默认
        50）步懒刷新；每步 z 从 span(Q) 采：

            z = Σ_i c_i q_i + α·ξ，  c ~ N(0, I_k)，ξ ~ N(0, I_d)，
            α = noise·√k/√d，  z ← z·(√d/‖z‖)

        LoZO 的 W 空间扰动 ΔW = ε·U·Vᵀ（V 持久懒刷新、U 每步重采）在我们的
        LoRA-only 设定下的等价形式：θ 参数空间的持久随机子空间 + 每步新系数。
        rank→d 时 CLT 收敛回 dense 高斯 ZO（InferZO 论文的论证）。无自适应
        判据、无 loss 收集 —— 简单、确定、与 InferZO released-LoZO 对齐。

    **adaptive**（旧行为，实验判负后保留作对照）：

        基从「loss_mid 创新低」步的梯度估计 ĝ 收集（Gram-Schmidt、近共线拒绝、
        环形容量）。2026-11-16 实验判负：batch-rows 1 下 loss 锯齿使判据失效
        （修复 EMA 基线后仍比均匀慢 8×）。

    对齐度升到 ~1/√k；α·ξ（默认 noise=0.2）保留子空间外的探索。范数重标定到
    √d 与各向同性的期望范数一致 ⇒ ε 与 lr 的语义在三种模式间不变。
    subspace=0 时所有路径逐位不变。
"""

from __future__ import annotations

import numpy as np


def theta_norms(theta: dict, layers: list, facs: tuple[str, ...] = ("a", "b")):
    """(‖θ‖, ‖θ‖ 只算参与更新的因子)。

    为什么要两个口径：`--zo-factors b` 下 A 是冻结的随机基，‖A‖ 就是不动的常数，
    而 B 从 0 长起。只用全范数时 `rel = ‖lr·v‖/‖θ‖` 被那个常数压掉约 3 个数量级，
    `rel > 0.05` 的发散守卫等于关闭。判步长要看**能动的那个范数**。
    `facs` 含 ("a","b") 时两个口径相同（历史数字靠这一支复现）。
    """
    full = moving = 0.0
    for l in layers:
        for _w, (fa, fb) in theta[l].items():
            a2 = float(np.sum(fa * fa))
            b2 = float(np.sum(fb * fb))
            full += a2 + b2
            moving += (a2 if "a" in facs else 0.0) + (b2 if "b" in facs else 0.0)
    return float(np.sqrt(full)), float(np.sqrt(moving))


def rel_cap_would_selflock(facs: tuple[str, ...], norm_moving0: float) -> bool:
    """`rel_cap` 在这个 (更新因子, 起点可动范数) 上是否**自锁**。

    自锁的定义是「上限 ∝ 自己要长的那个量」：`--zo-factors b` 下上限 = rel×‖B‖，而 ‖B‖ 从 0 起、
    只能靠步长长大 ⇒ 第一步把 ‖B‖ 抬到 s₁，第二步上限就是 rel×s₁（比 s₁ 小两个数量级），
    之后永远 ≈ rel×‖B‖。实测（2026-11-14 idH）rel 0.01 把每步钉在 1.4e-4，17 步 ‖B‖ 没长。

    `--zo-factors ab` 不会：‖A‖ 是随机初始化基、不动，分母有常数基线。
    这个判定必须显式报错而不是打印警告 —— 症状是「训练不报错也不动」，看一眼日志根本看不出来。
    """
    return "a" not in facs and norm_moving0 <= 0.0


def theta_guard_exceeded(norm: float, norm0: float, max_rel: float) -> bool:
    """`--theta-max-rel` 的停机判据：‖θ‖ > max_rel × ‖θ₀‖。

    为什么需要它（2026-11-22 identity v6/time 4000 步实测）：零阶的位移是**纯累积**的，
    `--zo-step-max` 只限制单步、不限制总量，`step_max` 一旦被钉死（本例 4000/4000 步），
    θ 就以恒定速度随机游走。‖θ‖ 2.82→5.85（2.07×）时生成掉进重复环，2.96× 时保持集
    48 题从 44 掉到 3 —— 而训练侧的 loss 曲线全程在降，**没有任何一个读数会提前报警**。
    这条是唯一在「已损坏」之前就能触发的读数。

    0 / 非正 / NaN 一律不触发：给了非法值绝不能把训练停在第 0 步。
    """
    try:
        max_rel, norm, norm0 = float(max_rel), float(norm), float(norm0)
    except (TypeError, ValueError):
        return False
    if not (max_rel > 0 and norm0 > 0 and norm == norm):
        return False
    return norm > norm0 * max_rel


class Mezo:
    """零阶优化器：持超参，跑一步，原地改 θ 与动量，返回读数。

    参数全部是**算法本身的参数**，没有一个是训练器的：
      eps      差分步长（论文的 ε，本实现单位是 LoRA 的 B 空间）
      dirs     每步采几个独立方向（K）
      momentum 方向动量的衰减；0 = 纯 SGD
      facs     参与更新的因子（"b" / ("a","b")）
      rel_cap  每步位移 ≤ rel_cap × ‖θ_能动‖；0 = 不裁
      step_max 每步位移 ≤ step_max（绝对量）；0 = 不裁
      one_point 用 residual-feedback 单点估计（每步 1 次前向）；False = 原文两点
      subspace 结构化方向容量 K；0 = 关（均匀 z，历史行为逐位不变）
      subspace_noise 子空间外各向同性噪声占比（0.2 = 噪声能量为子空间分量的 4%）

    两个上限都是**尺度归一化**，不是算法的一部分：零阶的 `Δθ = lr·pj·z` 里
    ‖z‖ = √d，所以同一串 lr 在不同 d 下含义完全不同。MeZO 原文用 lr 1e-7 是因为
    它的 d ≈ 1e10；本项目的 d 小 4 个数量级，直接照搬 lr 会偏大 100 倍。
    给出 step_max 就把「一步走多远」从 lr 里解耦出来。
    """

    def __init__(self, layout: dict, layers: list, *, eps: float = 1e-3,
                 dirs: int = 1, momentum: float = 0.9, facs: tuple[str, ...] = ("b",),
                 rel_cap: float = 0.0, step_max: float = 0.0,
                 one_point: bool = False,
                 subspace: int = 0, subspace_noise: float = 0.2,
                 subspace_mode: str = "lozo", subspace_refresh: int = 50):
        self.layout = layout
        self.layers = list(layers)
        self.eps = float(eps)
        self.dirs = max(1, int(dirs))
        self.momentum = float(momentum)
        self.facs = tuple(facs)
        self.rel_cap = float(rel_cap or 0.0)
        self.step_max = float(step_max or 0.0)
        # 上一步的裁剪读数（诊断用：`mezo` 靠它统计「上限钉死了多少步」）
        self.clamped = False
        self.step_pre = 0.0
        self.scale = 1.0
        self.one_point = bool(one_point)
        self.subspace = max(0, int(subspace))
        self.subspace_noise = float(subspace_noise)
        if self.subspace and subspace_mode not in ("lozo", "adaptive"):
            raise ValueError(f"subspace_mode 必须是 lozo|adaptive，得到 {subspace_mode!r}")
        self.subspace_mode = str(subspace_mode)
        self.subspace_refresh = max(1, int(subspace_refresh))
        self._lozo_since_refresh = 0             # 距上次基刷新的步数（lozo 模式）
        self.prev_loss: float | None = None      # residual-feedback 的上一次扰动点损失
        # 结构化方向状态：正交基（每条是与 z 同布局的单位向量）+ 已见最低 loss_mid
        self._basis: list[dict] = []
        self._best_loss: float | None = None
        self.allow_push = True                   # 基收集闸门：per-item 基线未建立前关
        self._seg: list[tuple] | None = None     # (l, which, 因子名) 展平顺序缓存
        self._d: int | None = None               # 采样维数（只算被采样的因子）
        self._shape_cache: dict = {}             # f"{l}|{which}|{a/b}" -> ndarray（形状模板）

    # ---- 状态 ---------------------------------------------------------------
    def zeros(self, theta: dict) -> dict:
        """与 θ 同形状的动量状态（零初始化）。"""
        return {l: {w: (np.zeros_like(a), np.zeros_like(b))
                    for w, (a, b) in theta[l].items()} for l in self.layers}

    # ---- 结构化方向 ----------------------------------------------------------
    def _segments(self, theta: dict) -> list[tuple]:
        """展平顺序：(l, which, "a"/"b")，只含**被采样的**因子；首调用时按 θ 形状缓存
        （形状模板同时进 `_shape_cache`，供 `_unflatten`/`_push_basis` 在任何时点重建）。"""
        if self._seg is None:
            self._seg = []
            if self._shape_cache is None:        # push 可能先于结构化 sample 发生
                self._shape_cache = {}
            for l in self.layers:
                for which, (a, b) in theta[l].items():   # 全模块：每个挂点一段（out-only 时恰一段）
                    self._shape_cache[f"{l}|{which}|a"] = a   # 两个因子都存形状：
                    self._shape_cache[f"{l}|{which}|b"] = b   # 补零分支要给未采样因子建零阵
                    if "a" in self.facs:
                        self._seg.append((l, which, "a"))
                    if "b" in self.facs:
                        self._seg.append((l, which, "b"))
        return self._seg

    def _flatten(self, z: dict) -> np.ndarray:
        return np.concatenate([z[l][w][0 if f == "a" else 1].ravel()
                               for l, w, f in self._seg]).astype(np.float32)

    def _unflatten(self, vec: np.ndarray) -> dict:
        z, off = {}, 0
        for l, w, f in self._seg:
            arr = self._shape_cache[f"{l}|{w}|{f}"]
            n = arr.size
            piece = vec[off:off + n].reshape(arr.shape).astype(np.float32)
            off += n
            pa, pb = z.setdefault(l, {}).get(w, (None, None))
            z[l][w] = (piece, pb) if f == "a" else (pa, piece)
        for l in self.layers:                    # 补零：未采样因子保留形状
            for which in self.layout[l]:         # 全模块：每个挂点补齐（层级陷阱见下）
                # 注意层级：z 的键是层号，z[l] 的键才是权重名 —— 错一层会把整个
                # layer 条目覆盖成 (None, None)，刚拆出来的 piece 全部丢掉。
                z.setdefault(l, {}).setdefault(which, (None, None))
                pa, pb = z[l][which]
                if pa is None or pb is None:
                    a = self._shape_cache[f"{l}|{which}|a"]
                    b = self._shape_cache[f"{l}|{which}|b"]
                    z[l][which] = ((pa if pa is not None else np.zeros_like(a)),
                                   (pb if pb is not None else np.zeros_like(b)))
        return z

    def _basis_flat(self) -> np.ndarray:
        """正交基矩阵 (k, d)，按展平顺序拼起来。"""
        return np.stack([self._flatten(q) for q in self._basis])

    def _maybe_refresh_lozo(self, theta: dict, rng) -> None:
        """LoZO 懒刷新（arXiv:2410.07698 的 ν 间隔）：基空或满 refresh 步时
        重采随机正交基。基的来源是**随机**而非 loss 判据 —— InferZO 的
        LOZODirectionProvider（GaussianV 持久 + GaussianU 每步重采）在
        本模块参数空间设定下的等价形式。每个刷新周期恰好消耗同一 rng 流。"""
        self._segments(theta)
        if self._d is None:
            self._d = sum(self._shape_cache[f"{l}|{w}|{f}"].size
                          for l, w, f in self._segments(theta))
        if self._basis and self._lozo_since_refresh < self.subspace_refresh:
            self._lozo_since_refresh += 1
            return
        d = max(self._d, 1)
        k = min(self.subspace, d)
        G = rng.standard_normal((d, k)).astype(np.float32)
        Q, _ = np.linalg.qr(G)                   # 列正交：Q.T @ Q = I_k
        self._basis = [self._unflatten(np.ascontiguousarray(Q[:, i]).astype(np.float32))
                       for i in range(k)]
        self._lozo_since_refresh = 0

    def _push_basis(self, g_flat: np.ndarray, theta: dict) -> bool:
        """把梯度估计 ĝ（已展平）经 Gram-Schmidt 投进基；近共线则拒绝。返回是否收下。

        `theta` 只用于建立形状模板缓存（首次 push 可能早于任何结构化 sample）。"""
        if self._basis is None:                  # 跨步状态转手缺省防御
            self._basis = []
        self._segments(theta)
        n0 = float(np.linalg.norm(g_flat))
        if not np.isfinite(n0) or n0 < 1e-12:
            return False
        new = g_flat / n0
        for q in self._basis:
            new = new - float(np.dot(new, self._flatten(q))) * self._flatten(q)
        n1 = float(np.linalg.norm(new))
        if n1 < 0.05:                            # 与现有基几乎共线：不带新信息
            return False
        self._basis.append(self._unflatten((new / n1).astype(np.float32)))
        if len(self._basis) > self.subspace:
            self._basis.pop(0)                   # 环形：挤最旧
        return True

    # ---- 一步 ---------------------------------------------------------------
    def sample(self, theta: dict, rng) -> dict:
        """采一个方向。subspace>0 且基非空时从子空间采（lozo=随机懒刷新基，
        adaptive=历史有效梯度收集基，见模块文档），否则各向同性 N(0, I)
        （历史路径逐位不变）；不在 `facs` 里的因子补零。"""
        if not (self.subspace and self._basis):
            z = {}
            for l in self.layers:
                z[l] = {}
                for which, (a, b) in theta[l].items():   # 全模块：每个挂点一个方向（out-only 时恰一个，rng 流不变）
                    za = (rng.standard_normal(a.shape).astype(np.float32)
                          if "a" in self.facs else np.zeros_like(a))
                    zb = (rng.standard_normal(b.shape).astype(np.float32)
                          if "b" in self.facs else np.zeros_like(b))
                    z[l][which] = (za, zb)
            return z

        if self._d is None:                      # 首次：按 θ 形状算采样维数
            self._d = sum(self._shape_cache[f"{l}|{w}|{f}"].size
                          for l, w, f in self._segments(theta))
        k = len(self._basis)
        d = max(self._d, 1)
        # 子空间分量：c ~ N(0, I_k) 张成；噪声分量 α·ξ，α = noise·√k/√d
        c = rng.standard_normal(k).astype(np.float32)
        z_sub = (self._basis_flat().T @ c).astype(np.float32)
        alpha = self.subspace_noise * np.sqrt(k / d)
        xi = rng.standard_normal(d).astype(np.float32)
        z_raw = z_sub + alpha * xi
        nrm = float(np.linalg.norm(z_raw))
        if not np.isfinite(nrm) or nrm < 1e-12:
            z_raw = xi                           # 退化：退回纯各向同性
            nrm = float(np.linalg.norm(z_raw))
        z_raw = z_raw * (np.sqrt(d) / nrm)       # 范数钉在 √d：ε/lr 语义与均匀模式一致
        return self._unflatten(z_raw)

    def step(self, theta: dict, vel: dict, lr: float, loss_fn, rng,
             step_max: float | None = None):
        """跑一步，原地改 theta/vel，返回 (pj_vals, ‖θ‖, rel, loss_mid, ‖θ_能动‖)。

        `loss_mid = mean_q (L(θ+εz_q) + L(θ−εz_q))/2`：**零额外开销**的中心损失估计
        —— 差分本来就算过这两个点，取平均即 L(θ) 的 O(ε²) 估计。轮次曲线画它而不是画
        pj，是因为 pj 是方向导数，可正可负、随方向抖动。

        `step_max`：本步的绝对位移上限，缺省用实例值。**训练器每步传进来**是为了让
        lr 曲线能作用到上限上 —— 上限一旦先钉死，`scale` 会把位移重归一化回同一个
        范数，lr 衰减被完全吃掉（见 `clamped` 与 `mezo._zo_step` 的注释）。
        """
        theta_lo = {l: dict(theta[l]) for l in self.layers}   # 只取参与更新的层

        acc = {l: {w: (np.zeros_like(a), np.zeros_like(b))
                   for w, (a, b) in theta_lo[l].items()} for l in self.layers}
        pj_vals: list[float] = []
        mids: list[float] = []

        # lozo 模式：基的懒刷新按**步**计数（dirs 个方向共用同一周期）。
        if self.subspace and self.subspace_mode == "lozo":
            self._maybe_refresh_lozo(theta_lo, rng)

        for _ in range(self.dirs):
            z = self.sample(theta_lo, rng)
            th_plus = {l: {w: (theta_lo[l][w][0] + self.eps * z[l][w][0],
                               theta_lo[l][w][1] + self.eps * z[l][w][1])
                           for w in theta_lo[l]} for l in self.layers}
            if self.one_point:
                # residual feedback：只采 +ε 一点，上一步的扰动点代理 L(θ)。
                lp = loss_fn(th_plus)
                base = (self.prev_loss if self.prev_loss is not None
                        else loss_fn(theta_lo))       # 首步一次性基线
                pj = (lp - base) / self.eps
                mids.append(lp)
                self.prev_loss = lp
            else:
                th_minus = {l: {w: (theta_lo[l][w][0] - self.eps * z[l][w][0],
                                    theta_lo[l][w][1] - self.eps * z[l][w][1])
                                for w in theta_lo[l]} for l in self.layers}
                lp = loss_fn(th_plus)
                lm = loss_fn(th_minus)
                pj = (lp - lm) / (2 * self.eps)
                mids.append(0.5 * (lp + lm))
            pj_vals.append(pj)
            for l in self.layers:
                for which, (za, zb) in z[l].items():     # 全模块：每个被扰动的挂点都收梯度
                    pa, pb = acc[l][which]
                    acc[l][which] = (pa + pj * za, pb + pj * zb)

        # 结构化方向（仅 adaptive 模式）：只有 loss_mid 创新低的步才把本步的
        # ĝ 估计收进正交基。lozo 的基是随机懒刷新的，不收 loss 判据方向 ——
        # 混收会把随机基污染成自适应基，两种模式的判据语义就都丢了。
        # ĝ = mean(pj·z) 是「这步实测有用的方向」；无效/变差的步不污染基。
        # allow_push=False（per-item 基线未建立的首轮）时既不收基也不记 best——
        # 否则首个低损失样本会把 _best_loss 钉在假低位，之后全是「非创新低」。
        # `_segments(theta_lo)` 要在首次 push 前调用——形状模板在那时才建立。
        if (self.subspace and self.subspace_mode == "adaptive"
                and self.allow_push and mids):
            lm = float(np.mean(mids))
            if self._best_loss is None or lm < self._best_loss:
                self._best_loss = lm
                self._segments(theta_lo)
                g_dict = {l: dict(acc[l]) for l in self.layers}
                self._push_basis(self._flatten(g_dict), theta_lo)

        # 动量：vel ← m·vel + mean(ĝ)
        for l in self.layers:
            for which, (pa, pb) in acc[l].items():
                va, vb = vel[l][which]
                vel[l][which] = (self.momentum * va + pa / self.dirs,
                                 self.momentum * vb + pb / self.dirs)

        # 尺度上限裁的是**施加的位移**，不是动量本身 —— 动量里跨步累积的方向要留着。
        norm_pre, mov_pre = theta_norms(theta_lo, self.layers, self.facs)
        step_pre = 0.0
        for l in self.layers:
            for which, (va, vb) in vel[l].items():
                step_pre += float(np.sum((lr * va) ** 2) + np.sum((lr * vb) ** 2))
        step_pre = step_pre ** 0.5
        scale = 1.0
        sm = self.step_max if step_max is None else float(step_max)
        if self.rel_cap > 0 and mov_pre > 0 and step_pre > self.rel_cap * mov_pre:
            scale = self.rel_cap * mov_pre / step_pre
        if sm > 0 and step_pre > sm:
            scale = min(scale, sm / step_pre)
        self.clamped = sm > 0 and step_pre > sm
        self.step_pre = step_pre
        self.scale = scale

        for l in self.layers:
            for which, (a, b) in theta_lo[l].items():
                va, vb = vel[l][which]
                theta_lo[l][which] = (a - scale * lr * va, b - scale * lr * vb)

        # 写回（--zo-layers 下 theta 可能还含未训层，原样保留）
        for l in self.layers:
            theta[l] = theta_lo[l]

        norm, mov = theta_norms(theta_lo, self.layers, self.facs)
        # rel 报**实际施加**的位移（裁过就按裁后算）。起点 B=0 时可动范数为 0：记 NaN
        # （= JSON null），不记 0 —— 0 会被读成「没走」。
        rel = (scale * step_pre) / mov_pre if mov_pre > 0 else float("nan")
        loss_mid = float(np.mean(mids)) if mids else None
        return pj_vals, norm, rel, loss_mid, mov
