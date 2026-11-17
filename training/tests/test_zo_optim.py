"""零阶估计器的独立判据集：算法本身对不对，与训练器无关。

为什么单独钉（2026-11-14）：`_zo_step` 原先埋在一个 2000 行的训练器里，任何一个
「评测口径 / 模板 / 数据」问题都会让人怀疑到算法上，反过来也一样 —— 三个月里
每次都只能整体重跑才能判断「是算法错了还是尺子错了」。抽出 `training/zo/optim.py`
后，算法有 6 条可以离线秒过的判据：

  1. 线性损失 → pj 精确可算（估计量没有符号/系数错）
  2. 二次损失 → cos(ĝ, ∇L) == |cos(z, ∇L)|（零阶估计的方向性质，逐位级）
  3. rel == ‖实际位移‖ / ‖位移前的可动范数‖（步长读数的定义）
  4. step_max 可加：N 步总位移按 √N 增长并可规划
  5. momentum 跨步累积（vel 是状态，不是每步清零）
  6. 与旧 `EngineMezo._zo_step` 同 seed 同输入**逐位一致**（抽取没改数值）

第 6 条是回归锁：抽取重构只许搬家，不许改数。
"""

import argparse
import pathlib as _p
import sys as _s
import unittest

import numpy as np

for _d in (_p.Path(__file__).resolve().parent, _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

from zo.optim import Mezo, theta_norms       # noqa: E402


def make_theta(b_scale=0.0, rank=2, n_in=8, n_out=4):
    a = np.full((rank, n_in), 0.5, dtype=np.float32)
    b = np.full((n_out, rank), b_scale, dtype=np.float32)
    return {0: {"ssm_out": (a, b)}}


LAYOUT = {0: ("ssm_out", 8, 4)}


def vel_new_template(theta):
    """与 θ 同形状的零动量状态。"""
    return {l: {w: (np.zeros_like(a), np.zeros_like(b)) for w, (a, b) in theta[l].items()}
            for l in theta}


class LinearLoss:
    """L(θ) = slope·ΣB（对 B 线性）⇒ pj = slope·Σ(εz)/ε = slope·Σz，可精确断言。"""

    def __init__(self, slope=1.0):
        self.slope = slope
        self.calls = 0

    def __call__(self, th):
        self.calls += 1
        return self.slope * float(np.sum(th[0]["ssm_out"][1]))


class TestEstimatorMath(unittest.TestCase):
    def test_linear_pj_is_exact(self):
        """pj 必须等于 (L+ − L−)/2ε，没有漏系数 2、没有符号反。"""
        th = make_theta(0.0)
        opt = Mezo(LAYOUT, [0], eps=1e-3, dirs=1, momentum=0.0, facs=("b",))
        vel = opt.zeros(th)
        loss = LinearLoss(slope=1.0)
        rng = np.random.default_rng(11)
        z = opt.sample({0: dict(th[0])}, np.random.default_rng(11))   # 复现同一方向

        pj, _n, _r, _m, _mov = opt.step(th, vel, 0.0, loss, np.random.default_rng(11))
        want = float(np.sum(z[0]["ssm_out"][1]))
        self.assertAlmostEqual(pj[0], want, places=3, msg=f"pj {pj[0]} != Σz {want}")
        self.assertEqual(loss.calls, 2, "一步必须只调两次损失（±ε 各一次）")

    def test_quadratic_direction_identity(self):
        """二次损失 L = ½‖b − c‖² ⇒ ĝ ∝ (c−b) 方向；cos(ĝ,∇L) == |cos(z,∇L)|。

        这是零阶估计量的锐利判据（2026-10-29 test_zo 同一条）：它对每个方向逐个成立，
        不依赖平均、不依赖学习率，任何符号/缩放错误都会当场破掉。
        """
        c = np.arange(8, dtype=np.float32).reshape(4, 2) / 4.0
        grad_dir = lambda th: th[0]["ssm_out"][1] - c     # ∇L = b − c

        def L(th):
            d = th[0]["ssm_out"][1] - c
            return 0.5 * float(np.sum(d * d))

        for seed in (1, 2, 3):
            th = make_theta(0.0)
            opt = Mezo(LAYOUT, [0], eps=1e-3, dirs=1, momentum=0.0, facs=("b",))
            vel = opt.zeros(th)
            z = opt.sample({0: dict(th[0])}, np.random.default_rng(seed))
            pj, *_ = opt.step(th, vel, 0.0, L, np.random.default_rng(seed))
            g_hat = pj[0] * z[0]["ssm_out"][1]
            g_true = grad_dir(make_theta(0.0))
            cos = lambda u, v: float(np.sum(u * v) /
                                     (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30))
            self.assertAlmostEqual(cos(g_hat, g_true), abs(cos(z[0]["ssm_out"][1], g_true)),
                                   places=5, msg=f"seed {seed}: cos 恒等式不成立")

    def test_zero_lr_does_not_move_theta(self):
        th = make_theta(0.01)
        b0 = th[0]["ssm_out"][1].copy()
        opt = Mezo(LAYOUT, [0], eps=1e-3, dirs=1, momentum=0.0, facs=("b",))
        opt.step(th, opt.zeros(th), 0.0, LinearLoss(), np.random.default_rng(0))
        np.testing.assert_array_equal(th[0]["ssm_out"][1], b0)


class TestScale(unittest.TestCase):
    def test_rel_identity(self):
        th = make_theta(0.01)
        b0 = th[0]["ssm_out"][1].copy()
        opt = Mezo(LAYOUT, [0], eps=1e-3, dirs=1, momentum=0.0, facs=("b",))
        _pj, _n, rel, _m, mov = opt.step(th, opt.zeros(th), 5e-5, LinearLoss(),
                                         np.random.default_rng(7))
        d = th[0]["ssm_out"][1].astype(np.float64) - b0.astype(np.float64)
        want = float(np.linalg.norm(d)) / float(np.linalg.norm(b0))
        self.assertLess(abs(rel - want) / want, 1e-5)
        self.assertAlmostEqual(mov, float(np.linalg.norm(th[0]["ssm_out"][1])), places=6)

    def test_nan_at_zero_b(self):
        th = make_theta(0.0)
        opt = Mezo(LAYOUT, [0], eps=1e-3, dirs=1, momentum=0.0, facs=("b",))
        _pj, _n, rel, _m, mov = opt.step(th, opt.zeros(th), 5e-5, LinearLoss(),
                                        np.random.default_rng(0))
        self.assertTrue(rel != rel, "起点可动范数为 0 ⇒ rel 无定义，必须是 NaN")
        self.assertGreater(mov, 0.0)

    def test_step_max_is_a_per_step_budget(self):
        """绝对上限是**每步**预算，总位移与预算成线性 ⇒ 可以规划。

        判据分三条：
          1. 任何一步的位移都不越界（严格）；
          2. N 步的总位移 ≤ N×预算（不越界），且明显小于无预算时的量级；
          3. 预算减半 ⇒ 总位移也大致减半。相对上限（--zo-rel）做不到第 3 条：
             它按当前范数的百分比放行，B 从 0 起时是**复利**，16.8→54→153→366 那个
             300 步曲线就是这么来的（2026-11-14 实测）。
        """
        def run(n, cap):
            th = make_theta(0.0)
            opt = Mezo(LAYOUT, [0], eps=1e-3, dirs=1, momentum=0.0,
                       facs=("b",), step_max=cap)
            vel = opt.zeros(th)
            rng = np.random.default_rng(5)
            worst = 0.0
            for _ in range(n):
                b0 = th[0]["ssm_out"][1].copy()
                opt.step(th, vel, 1.0, LinearLoss(), rng)      # lr 取 1，让上限说话
                worst = max(worst, float(np.linalg.norm(
                    th[0]["ssm_out"][1].astype(np.float64) - b0.astype(np.float64))))
            return worst, float(np.linalg.norm(th[0]["ssm_out"][1]))

        w, d_full = run(200, 1e-2)
        _w, d_half = run(200, 5e-3)
        self.assertLessEqual(w, 1e-2 * 1.000001, f"有单步越界：{w}")
        self.assertLessEqual(d_full, 1e-2 * 200)               # 不越界（平凡但必要）
        self.assertAlmostEqual(d_half / d_full, 0.5, delta=0.1,
                               msg=f"预算减半但总位移比 {d_half / d_full:.2f} —— 不可线性规划")

    def test_momentum_accumulates(self):
        """同方向两遍：无动量时第二步位移 == 第一步；有动量时第二步更大。"""
        def steps(mom):
            th = make_theta(0.0)
            opt = Mezo(LAYOUT, [0], eps=1e-3, dirs=1, momentum=mom, facs=("b",))
            vel = opt.zeros(th)
            out, prev = [], 0.0
            for i in range(2):
                opt.step(th, vel, 1e-2, LinearLoss(), np.random.default_rng(0))  # 同方向
                cur = float(np.linalg.norm(th[0]["ssm_out"][1]))
                out.append(cur - prev)
                prev = cur
            return out

        flat, mom = steps(0.0), steps(0.9)
        self.assertAlmostEqual(flat[0], flat[1], places=6, msg="无动量时两步必须一样")
        self.assertGreater(mom[1], mom[0] * 1.5, "动量没有累积")


class TestBitwiseParityWithOldImpl(unittest.TestCase):
    """抽取重构的回归锁：同一 seed、同一损失桩，新旧实现必须给出同样的 θ。"""

    def test_same_theta_same_pj(self):
        import mezo as legacy

        class Stub(legacy.EngineMezo):
            def __init__(self, dirs=3, momentum=0.9, zo_rel=0.0, step_max=0.0):
                self.args = argparse.Namespace(dirs=dirs, eps=1e-3, momentum=momentum,
                                               zo_factors="b", zo_rel=zo_rel,
                                               zo_step_max=step_max)
                self.layout = dict(LAYOUT)

            def loss_at(self, th, batch):
                return 1.0 * float(np.sum(th[0]["ssm_out"][1])) + 0.5 * float(
                    np.sum(th[0]["ssm_out"][1] ** 2))

        for kw in (dict(), dict(zo_rel=1e-3), dict(step_max=1e-2), dict(momentum=0.0)):
            old = Stub(**kw)
            th_old = make_theta(0.01)
            opt = Mezo(LAYOUT, [0], eps=1e-3, dirs=old.args.dirs,
                       momentum=old.args.momentum, facs=("b",),
                       rel_cap=kw.get("zo_rel", 0.0), step_max=kw.get("step_max", 0.0))
            th_new = make_theta(0.01)
            vel_old = dict(vel_new_template(th_old))
            vel_new = opt.zeros(th_new)
            for i in range(3):
                a = old._zo_step(th_old, vel_old, np.random.default_rng(100 + i),
                                 [], [0], 5e-5)
                b = opt.step(th_new, vel_new, 5e-5, lambda t: old.loss_at(t, []),
                             np.random.default_rng(100 + i))
                self.assertEqual(a[0], b[0], f"{kw} step {i}: pj 列不一致 {a[0]} vs {b[0]}")
                self.assertAlmostEqual(a[1], b[1], places=9)
                self.assertEqual(str(a[2]), str(b[2]), f"{kw} step {i}: rel 不一致")
                self.assertAlmostEqual(a[3], b[3], places=9)
                self.assertAlmostEqual(a[4], b[4], places=9)
            np.testing.assert_array_equal(th_old[0]["ssm_out"][1], th_new[0]["ssm_out"][1],
                                          err_msg=f"{kw}: 走完 3 步 θ 必须逐位相同")


class TestOnePoint(unittest.TestCase):
    """residual-feedback 单点估计：每步 1 次前向，默认关闭时行为逐位不变。"""

    def test_one_forward_per_step_after_the_first(self):
        th = make_theta(0.0)
        opt = Mezo(LAYOUT, [0], eps=1e-3, dirs=1, momentum=0.0, facs=("b",),
                   one_point=True)
        vel = opt.zeros(th)
        loss = LinearLoss(slope=1.0)
        rng = np.random.default_rng(7)
        for _ in range(3):
            opt.step(th, vel, 0.0, loss, rng)
        # 首步多一次中心基线，之后每步 1 次 ⇒ 4；两点估计是 6。
        self.assertEqual(loss.calls, 4, "单点估计除首步外每步只能调一次损失")

    def test_default_is_two_point(self):
        th = make_theta(0.0)
        opt = Mezo(LAYOUT, [0], eps=1e-3, dirs=1, momentum=0.0, facs=("b",))
        self.assertFalse(opt.one_point)
        loss = LinearLoss(slope=1.0)
        opt.step(th, opt.zeros(th), 0.0, loss, np.random.default_rng(7))
        self.assertEqual(loss.calls, 2, "默认必须仍是原文两点估计")

    def test_first_step_pj_matches_two_point_on_a_linear_loss(self):
        """线性损失下 `(L(θ+εz)−L(θ))/ε` 与 `(L+−L−)/2ε` 都等于 slope·Σz。"""
        rng_seed = 23
        z = Mezo(LAYOUT, [0], eps=1e-3, facs=("b",)).sample(
            make_theta(0.0), np.random.default_rng(rng_seed))
        want = float(np.sum(z[0]["ssm_out"][1]))
        for one in (False, True):
            th = make_theta(0.0)
            opt = Mezo(LAYOUT, [0], eps=1e-3, dirs=1, momentum=0.0, facs=("b",),
                       one_point=one)
            pj, *_ = opt.step(th, opt.zeros(th), 0.0, LinearLoss(1.0),
                              np.random.default_rng(rng_seed))
            self.assertAlmostEqual(pj[0], want, places=3,
                                   msg=f"one_point={one}: pj {pj[0]} != Σz {want}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
