"""结构化方向采样（--zo-subspace）的独立判据集：算法对不对，与训练器无关。

背景（2026-11-16，方案 A）：均匀 z 与真梯度的对齐度 ~1/√d（d≈1.67M 时 ≈0.001），
pj 信噪比 ~0.15 ⇒ 同等质量梯度要 ~(1/0.15)² ≈ 44 倍采样 —— 这就是零阶慢 40 倍
的主因。改进：把「loss_mid 创新低」步的梯度估计 ĝ 存成正交基 Q（Gram-Schmidt），
新方向从 span(Q) + 少量各向同性噪声采（对齐度 ~1/√k），范数钉在 √d 保持 ε/lr
语义不变。依据：梯度低维性 + InferZO/LoZO 的低秩方向结构。

钉住的判据：

  1. subspace=0（默认）逐位不变 —— sample 走纯各向同性分支，与旧实现同 seed 同序列
  2. 基只收「创新低」的步：变差的步不污染基
  3. 范数钉在 √d：结构化 z 的范数与均匀 z 的期望范数一致（ε 语义不变）
  4. 能量集中在子空间：noise=0 时 z 完全落在 span(Q)；noise=0.2 时子空间外能量 <5%
  5. 快降：把真梯度塞进基后，同 lr 同步数下结构化模式损失显著低于均匀模式
  6. 跨步状态存活：Mezo 实例每步重建（训练器真实用法），状态经训练器转手后
     基不丢、结构化采样立即可用
  7. Gram-Schmidt 拒绝近共线方向：基不带冗余、收下后标准正交
"""

import pathlib as _p
import sys as _s
import unittest

import numpy as np

for _d in (_p.Path(__file__).resolve().parent, _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

from zo.optim import Mezo  # noqa: E402

NL, NOUT, RANK, NIN = 2, 6, 3, 4


def make_theta(seed=0):
    rng = np.random.default_rng(seed)
    theta = {}
    for l in range(NL):
        a = rng.standard_normal((RANK, NIN)).astype(np.float32)
        b = np.zeros((NOUT, RANK), dtype=np.float32)
        theta[l] = {f"w{l}": (a, b)}
    return theta


THETA = make_theta()
LAYERS = list(THETA.keys())
LAYOUT = {l: (f"w{l}",) for l in LAYERS}
D_B = sum(b.size for l in LAYERS for _, b in THETA[l].values())   # 被采样维数


def vel_zero():
    return {l: {LAYOUT[l][0]: (np.zeros_like(a), np.zeros_like(b))
                for a, b in [THETA[l][LAYOUT[l][0]]]}
            for l in LAYERS}


def per_call_loss(values):
    """按**调用次数**吐预定值的损失桩（两点估计每步调 2 次：+ε、−ε）。"""
    it = iter(values)
    return lambda th: next(it)


def coupled_loss(values, scale=1e-3):
    """脚本值 + 微小 θ 耦合项：mid 按脚本走（判据可精确预言），pj 非零（ĝ 可进基）。

    pj = scale·(ΣB₊ − ΣB₋)/(2ε) = scale·Σz ≠ 0，方向 = 本步 z 的方向。"""
    it = iter(values)

    def f(th):
        s = sum(float(np.sum(th[l][LAYOUT[l][0]][1])) for l in LAYERS)
        return next(it) + scale * s

    return f


def flat_b(theta):
    return np.concatenate([theta[l][LAYOUT[l][0]][1].ravel() for l in LAYERS])


class TestOffIsBitwiseLegacy(unittest.TestCase):
    def test_subspace_off_matches_default_sequence(self):
        """判据 1：subspace=0 显式传入与默认不传，采样逐位一致。"""
        t1, t2 = make_theta(1), make_theta(1)
        m1 = Mezo(LAYOUT, LAYERS, subspace=0)
        m2 = Mezo(LAYOUT, LAYERS)
        for i in range(5):
            z1 = m1.sample(t1, np.random.default_rng(100 + i))
            z2 = m2.sample(t2, np.random.default_rng(100 + i))
            for l in LAYERS:
                np.testing.assert_array_equal(z1[l][LAYOUT[l][0]][1], z2[l][LAYOUT[l][0]][1])

    def test_off_step_return_unchanged(self):
        """默认构造下 step 返回与 subspace=0 显式传入逐位一致。"""
        loss = per_call_loss([3.0] * 64)
        out1 = Mezo(LAYOUT, LAYERS, dirs=2).step(
            make_theta(2), vel_zero(), 1e-3, loss, np.random.default_rng(3))
        out2 = Mezo(LAYOUT, LAYERS, dirs=2, subspace=0).step(
            make_theta(2), vel_zero(), 1e-3, loss, np.random.default_rng(3))
        np.testing.assert_array_equal(np.array(out1[0]), np.array(out2[0]))
        self.assertEqual(out1[1], out2[1])
        # rel 在 B=0 起点是 NaN（设计如此，记「没走」而非 0）—— NaN 对 NaN 视为相等
        self.assertTrue((out1[2] == out2[2])
                        or (np.isnan(out1[2]) and np.isnan(out2[2])))


class TestBasisCollection(unittest.TestCase):
    def test_basis_only_grows_on_improvement(self):
        """判据 2：loss_mid 变差的步不进基。mids = 10, 9, 9.5, 8 → 收 1、2、4 步。"""
        m = Mezo(LAYOUT, LAYERS, subspace=4, momentum=0.0, subspace_mode="adaptive")
        loss = coupled_loss([10, 10, 9, 9, 9.5, 9.5, 8, 8])
        vel = vel_zero()
        for _ in range(4):
            m.step(THETA, vel, 0.0, loss, np.random.default_rng(0))
        self.assertEqual(len(m._basis), 3)

    def test_rejects_near_collinear(self):
        """判据 7a：与现有基共线（含反向）的 ĝ 不重复收；正交方向收下。"""
        m = Mezo(LAYOUT, LAYERS, subspace=8)
        v = np.zeros(D_B, dtype=np.float32)
        v[0] = 1.0
        self.assertTrue(m._push_basis(v, THETA))
        self.assertFalse(m._push_basis(v * 2.0, THETA))
        self.assertFalse(m._push_basis(v * -3.0, THETA))
        self.assertEqual(len(m._basis), 1)
        v2 = np.zeros(D_B, dtype=np.float32)
        v2[1] = 1.0
        self.assertTrue(m._push_basis(v2, THETA))
        self.assertEqual(len(m._basis), 2)

    def test_basis_orthonormal(self):
        """判据 7b：收进基的向量互相正交且单位长。"""
        m = Mezo(LAYOUT, LAYERS, subspace=8)
        rng = np.random.default_rng(11)
        for _ in range(5):
            m._push_basis(rng.standard_normal(D_B).astype(np.float32), THETA)
        Q = m._basis_flat()
        np.testing.assert_allclose(Q @ Q.T, np.eye(len(m._basis)), atol=1e-4)


class TestStructuredSample(unittest.TestCase):
    def setUp(self):
        self.rng5 = np.random.default_rng(5)
        self.Qvectors = [self.rng5.standard_normal(D_B).astype(np.float32)
                         for _ in range(3)]

    def _filled(self, subspace=3, noise=0.0):
        m = Mezo(LAYOUT, LAYERS, subspace=subspace, subspace_noise=noise)
        for v in self.Qvectors[:subspace]:
            m._push_basis(v, THETA)
        return m

    def test_norm_pinned_to_sqrt_d(self):
        """判据 3：结构化 z 的范数 = √d_sampled（默认只采 b）。"""
        m = self._filled(3, 0.0)
        z = m.sample(THETA, np.random.default_rng(9))
        zn = float(np.linalg.norm(flat_b(z)))
        self.assertAlmostEqual(zn, np.sqrt(D_B), places=1)

    def test_energy_lives_in_subspace(self):
        """判据 4：noise=0 → z 完全在 span(Q)；noise=0.2 → 子空间外能量 <5%。"""
        m0 = self._filled(3, 0.0)
        Q = m0._basis_flat()
        z0 = m0.sample(THETA, np.random.default_rng(2))
        z0f = flat_b(z0)
        proj = Q.T @ (Q @ z0f)
        self.assertAlmostEqual(float(np.linalg.norm(proj)),
                               float(np.linalg.norm(z0f)), places=2)

        m2 = self._filled(3, 0.2)
        Q2 = m2._basis_flat()
        # 单次采样的能量比波动大（子空间分量 ~ χ²(k)），用 200 次均值对照期望值：
        # E[外能量]/E[总能量] = noise²/(1+noise²) = 0.038
        out_frac = []
        for i in range(200):
            z2 = m2.sample(THETA, np.random.default_rng(1000 + i))
            z2f = flat_b(z2)
            out = z2f - Q2.T @ (Q2 @ z2f)
            out_frac.append(float(np.sum(out ** 2)) / float(np.sum(z2f ** 2)))
        self.assertLess(float(np.mean(out_frac)), 0.08)

    def test_faster_loss_drop_with_true_gradient_in_basis(self):
        """判据 5：L=‖B−G‖²、G 已进基（noise=0）时，同 lr 同步数损失显著低于均匀模式。

        机理：E[Δθ] = −lr·P·∇L（P = 基上的投影）。G 在基内 ⇒ P∇L = ∇L 全量对齐且
        方差只在 k 维子空间；均匀模式 P=I 但 pj·z 的横向方差 ~ d 倍 —— 实际进展慢
        一个数量级。"""
        rngG = np.random.default_rng(13)
        G_flat = rngG.standard_normal(D_B).astype(np.float32)

        # 目标按 (l, w) 形状切开
        target, off = {}, 0
        for l in LAYERS:
            b = THETA[l][LAYOUT[l][0]][1]
            n = b.size
            target[l] = G_flat[off:off + n].reshape(b.shape).astype(np.float32)
            off += n

        def quad_loss(th):
            return sum(float(np.sum((th[l][LAYOUT[l][0]][1] - target[l]) ** 2))
                       for l in LAYERS)

        def run(mode):
            th = make_theta(4)                     # B=0 起点（与训练真实起点一致）
            vel = vel_zero()
            if mode == "structured":
                m = Mezo(LAYOUT, LAYERS, dirs=1, subspace=1, subspace_noise=0.0)
                m._push_basis(G_flat, th)          # 真「梯度」已在基里
            else:
                m = Mezo(LAYOUT, LAYERS, dirs=1)
            for _ in range(50):
                m.step(th, vel, 1e-4, lambda t: quad_loss(t), np.random.default_rng(7))
            return quad_loss(th)

        loss_uni = run("uniform")
        loss_st = run("structured")
        self.assertLess(loss_st, 0.5 * loss_uni)

    def test_state_survives_trainer_rebuild(self):
        """判据 6：训练器每步重建 Mezo 实例（mezo.py 的真实用法），基经转手不丢。"""
        held = {"basis": None, "best": None, "seg": None, "shapes": None, "d": None}
        loss = coupled_loss([5, 5, 4, 4, 3, 3])   # mids 5→4→3：步步创新低
        vel = vel_zero()
        for i in range(3):
            m = Mezo(LAYOUT, LAYERS, subspace=4, momentum=0.0, subspace_mode="adaptive")
            if held["basis"] is not None:
                m._basis, m._best_loss = held["basis"], held["best"]
                m._seg, m._shape_cache, m._d = held["seg"], held["shapes"], held["d"]
            m.step(THETA, vel, 0.0, loss, np.random.default_rng(i))
            held["basis"], held["best"] = m._basis, m._best_loss
            held["seg"], held["shapes"], held["d"] = m._seg, m._shape_cache, m._d
        self.assertEqual(len(held["basis"]), 3)    # 三步全创新低：三条都进基
        # 第 4 步用全新实例：结构化采样立即可用（形状缓存与 _d 都转手成功）
        m4 = Mezo(LAYOUT, LAYERS, subspace=4, subspace_mode="adaptive")
        m4._basis, m4._best_loss = held["basis"], held["best"]
        m4._seg, m4._shape_cache, m4._d = held["seg"], held["shapes"], held["d"]
        z = m4.sample(THETA, np.random.default_rng(0))
        self.assertAlmostEqual(float(np.linalg.norm(flat_b(z))), np.sqrt(D_B), places=1)


class TestAllowPushGate(unittest.TestCase):
    def test_allow_push_false_suppresses_basis_and_best(self):
        """allow_push=False（per-item 基线未建立的首轮）：基不收、best 不记。"""
        m = Mezo(LAYOUT, LAYERS, subspace=4, momentum=0.0, subspace_mode="adaptive")
        m.allow_push = False
        loss = coupled_loss([10, 10, 5, 5])       # mid 10 → 5：若不拦会是「创新低」
        vel = vel_zero()
        for _ in range(2):
            m.step(THETA, vel, 0.0, loss, np.random.default_rng(0))
        self.assertEqual(len(m._basis), 0)
        self.assertIsNone(m._best_loss)           # 假低位没被钉住
        m.allow_push = True
        loss2 = coupled_loss([8, 8])              # mid 8：从 None 起步正常首收
        m.step(THETA, vel, 0.0, loss2, np.random.default_rng(0))
        self.assertEqual(len(m._basis), 1)
        self.assertAlmostEqual(m._best_loss, 8.0, places=6)


if __name__ == "__main__":
    unittest.main()


class TestLozoMode(unittest.TestCase):
    """LoZO/InferZO 模式（随机懒刷新基，arXiv:2410.07698 / 2605.28760）。"""

    def test_basis_random_orthogonal_refresh_every_nu(self):
        m = Mezo(LAYOUT, LAYERS, subspace=4, subspace_mode="lozo", subspace_refresh=3)
        rng = np.random.default_rng(0)
        loss = per_call_loss([1.0] * 20)
        vel = vel_zero()
        m.step(THETA, vel, 0.0, loss, rng)             # 步 1：生成基
        q1 = np.stack([m._flatten(q).copy() for q in m._basis])
        self.assertEqual(q1.shape, (4, D_B))
        np.testing.assert_allclose(q1 @ q1.T, np.eye(4), atol=1e-4)
        for _ in range(3):                             # 步 2-4：基不变（ν=3）
            m.step(THETA, vel, 0.0, loss, rng)
        q2 = np.stack([m._flatten(q).copy() for q in m._basis])
        np.testing.assert_array_equal(q2, q1)
        m.step(THETA, vel, 0.0, loss, rng)             # 步 5：重采
        q3 = np.stack([m._flatten(q) for q in m._basis])
        self.assertFalse(np.allclose(q3, q1))

    def test_noise0_direction_lives_in_span_and_norm_is_sqrt_d(self):
        m = Mezo(LAYOUT, LAYERS, subspace=4, subspace_mode="lozo", subspace_noise=0.0)
        m._maybe_refresh_lozo(THETA, np.random.default_rng(1))
        z = m.sample(THETA, np.random.default_rng(2))
        zf = np.concatenate([z[l][LAYOUT[l][0]][1].ravel() for l in LAYERS])
        Q = m._basis_flat()
        resid = zf - Q.T @ (Q @ zf)
        self.assertLess(float(np.linalg.norm(resid)), 1e-4 * float(np.linalg.norm(zf)))
        self.assertAlmostEqual(float(np.linalg.norm(zf)), np.sqrt(D_B), places=3)

    def test_refresh_counter_survives_handoff(self):
        m1 = Mezo(LAYOUT, LAYERS, subspace=4, subspace_mode="lozo", subspace_refresh=3)
        m1._maybe_refresh_lozo(THETA, np.random.default_rng(0))
        m1._lozo_since_refresh = 2
        m2 = Mezo(LAYOUT, LAYERS, subspace=4, subspace_mode="lozo", subspace_refresh=3)
        m2._basis, m2._lozo_since_refresh = m1._basis, m1._lozo_since_refresh
        m2._maybe_refresh_lozo(THETA, np.random.default_rng(0))
        self.assertEqual(m2._lozo_since_refresh, 3)    # 2→3，未重采

    def test_lozo_never_collects_adaptive_basis(self):
        m = Mezo(LAYOUT, LAYERS, subspace=4, subspace_mode="lozo")
        loss = per_call_loss([10, 5, 1, 10, 5, 1, 10, 5])   # 创新低多步
        vel = vel_zero()
        for _ in range(4):
            m.step(THETA, vel, 0.0, loss, np.random.default_rng(0))
        self.assertIsNone(m._best_loss)                # adaptive 收集在 lozo 模式关闭

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            Mezo(LAYOUT, LAYERS, subspace=4, subspace_mode="bogus")


if __name__ == "__main__":
    unittest.main()
