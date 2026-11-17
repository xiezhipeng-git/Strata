"""零阶步长的两个读数：‖θ‖ 与 rel 的分母必须是**能动的那个范数**。

为什么单独钉这一条（2026-11-14 GRPO 复盘）：`--zo-factors b` 下 A 是冻结的随机基，
‖A‖ 是不变的常数（本模型 19.59），B 从 0 长起。旧代码把 `rel = ‖lr·v‖/‖θ全‖` 当
相对步长，于是：

  * GRPO 跑完 78 步，日志上 ‖θ‖ 19.5919 → 19.5920、rel ~1e-5，看上去「一步没走」，
    而 npz 里 ‖b‖ 已经是 0.071（bmax 3.6e-4）；
  * `rel > 0.05` 的发散守卫按旧口径要 276 倍才响 —— 实际等于关闭；
  * `--zo-rel` 的上限裁同样按 276 倍放水。

本文件把 rel 钉成「实际施加的位移 / 施加前的可动范数」这个恒等式，并用一个线性
损失的桩把 pj 变成可精确预期的常数。
"""

import argparse
import pathlib as _p
import sys as _s
import unittest

import numpy as np

for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

import mezo


class Stub(mezo.EngineMezo):
    """只留下 _zo_step 用得到的三样：args / layout / loss_at。"""

    def __init__(self, zo_factors="b", zo_rel=0.0, slope=1.0):
        self.args = argparse.Namespace(dirs=1, eps=1e-3, momentum=0.0,
                                       zo_factors=zo_factors, zo_rel=zo_rel)
        self.layout = {0: ("ssm_out", 8, 4)}
        self.slope = slope

    def loss_at(self, th, batch):
        # 对 b 线性：L = s·Σb ⇒ pj = (L+ − L−)/(2ε) = s·Σz_b，可精确断言
        return self.slope * float(np.sum(th[0]["ssm_out"][1]))


def make_theta(b_scale=0.0, rank=2, n_in=8, n_out=4):
    a = np.full((rank, n_in), 0.5, dtype=np.float32)
    b = np.full((n_out, rank), b_scale, dtype=np.float32)
    return {0: {"ssm_out": (a, b)}}


def make_vel(theta):
    return {l: {w: (np.zeros_like(a), np.zeros_like(b)) for w, (a, b) in theta[l].items()}
            for l in theta}


class NormSplit(unittest.TestCase):
    def test_full_vs_moving(self):
        theta = make_theta(b_scale=0.01)
        full, mov = mezo.theta_norms(theta, [0], ("b",))
        self.assertAlmostEqual(mov, float(np.linalg.norm(theta[0]["ssm_out"][1])), places=6)
        self.assertGreater(full, 10 * mov, "本用例里 A 才是范数来源，全范数必须远大于 ‖b‖")

    def test_ab_keeps_historical_definition(self):
        theta = make_theta(b_scale=0.01)
        full, mov = mezo.theta_norms(theta, [0], ("a", "b"))
        self.assertAlmostEqual(full, mov, places=9, msg="--zo-factors ab 下两个口径必须相同")


class RelIsTheAppliedStep(unittest.TestCase):
    def test_zero_b_reports_nan_not_zero(self):
        """起点 B=0：可动范数是 0，比值无定义 ⇒ NaN（0 会被读成「没走」）。"""
        st = Stub()
        theta, vel = make_theta(), make_vel(make_theta())
        pj, norm, rel, _lm, mov = st._zo_step(theta, vel, np.random.default_rng(0), [], [0], 5e-5)
        self.assertTrue(rel != rel, f"起点 rel 应为 NaN，实际 {rel}")
        self.assertGreater(mov, 0.0, "返回的是**施加后**的 |b|，起点走完一步就不该是 0")
        self.assertGreater(float(np.abs(theta[0]["ssm_out"][1]).sum()), 0.0, "更新没落到 b 上")
        self.assertGreater(norm, 0.0)

    def test_rel_equals_applied_over_moving(self):
        """恒等式：rel == ‖Δb‖ / ‖b_施加前‖。"""
        st = Stub()
        theta = make_theta(b_scale=0.01)
        vel = make_vel(theta)
        b0 = theta[0]["ssm_out"][1].copy()
        lr = 5e-5
        pj, _norm, rel, _lm, mov = st._zo_step(theta, vel, np.random.default_rng(7),
                                               [], [0], lr)
        d = theta[0]["ssm_out"][1].astype(np.float64) - b0.astype(np.float64)
        want = float(np.linalg.norm(d)) / float(np.linalg.norm(b0))
        # float32 累加 + 精确除法，只差 1e-6 相对量级；断言用相对误差
        self.assertLess(abs(rel - want) / want, 1e-5, f"rel {rel} != ‖Δb‖/‖b0‖ {want}")
        self.assertAlmostEqual(mov, float(np.linalg.norm(theta[0]["ssm_out"][1])), places=6)

    def test_old_denominator_was_two_orders_off(self):
        """同一份更新：旧口径（全范数当分母）会把 rel 压小到 read 不出来。

        这里 A 的范数比 b 大 ~30 倍，所以新 rel 至少是旧 rel 的 10 倍 ——
        在真模型上这个倍数是 276（‖A‖ 19.59 / ‖b‖ 0.071）。
        """
        st = Stub()
        theta = make_theta(b_scale=0.01)
        vel = make_vel(theta)
        _pj, _n, rel_new, _lm, _mov = st._zo_step(theta, vel, np.random.default_rng(7),
                                                  [], [0], 5e-5)
        theta2 = make_theta(b_scale=0.01)
        vel2 = make_vel(theta2)
        b0 = theta2[0]["ssm_out"][1].copy()
        st._zo_step(theta2, vel2, np.random.default_rng(7), [], [0], 5e-5)
        d2 = theta2[0]["ssm_out"][1].astype(np.float64) - b0.astype(np.float64)
        rel_old = float(np.linalg.norm(d2)) / mezo.theta_norms(theta2, [0], ("b",))[0]
        self.assertGreater(rel_new, 10 * rel_old)


class ZoRelCap(unittest.TestCase):
    def test_cap_actually_binds(self):
        """--zo-rel 现在按可动范数裁：同一 lr 下开上限必须把位移真的压下来。"""
        free = Stub(zo_rel=0.0)
        capped = Stub(zo_rel=1e-3)
        outs = []
        for st in (free, capped):
            theta = make_theta(b_scale=0.01)
            vel = make_vel(theta)
            b0 = theta[0]["ssm_out"][1].copy()
            _pj, _n, rel, _lm, _mov = st._zo_step(theta, vel, np.random.default_rng(3),
                                                  [], [0], 1.0)
            d = theta[0]["ssm_out"][1].astype(np.float64) - b0.astype(np.float64)
            outs.append((float(np.linalg.norm(d)), rel))
        (d_free, rel_free), (d_cap, rel_cap) = outs
        self.assertGreater(d_free, d_cap * 10, "上限没起作用（旧口径下这是常态）")
        self.assertLessEqual(rel_cap, 1e-3 * 1.000001)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ------------------------------------------------------------------ 两个目标的 loss 尺度必须可比

class OnePointAcrossSteps(unittest.TestCase):
    """训练器每步新建 Mezo 实例时，residual-feedback 的跨步状态不能丢。

    2026-11-15 实测：`prev_loss` 只存在优化器实例里，而 `_zo_step` 每步重建实例，
    于是每步都被当成首步、重新算一次中心基线 —— 声称的「每步 1 次前向」仍是 2 次，
    还额外背上单点估计的偏差（800 步 260 s vs 两点 276 s，CE 降幅腰斩）。
    """

    def _run(self, one: bool, steps: int = 4) -> int:
        st = Stub()
        st.args.zo_one_point = one
        calls = [0]
        orig = st.loss_at

        def counted(th, batch):
            calls[0] += 1
            return orig(th, batch)

        st.loss_at = counted
        theta = make_theta(0.0)
        vel = make_vel(theta)
        for i in range(steps):
            st._zo_step(theta, vel, np.random.default_rng(100 + i), [], [0], 0.0)
        return calls[0]

    def test_one_point_keeps_state_across_rebuilt_optimizers(self):
        # 首步 2 次（基线 + 扰动点），其后每步 1 次 ⇒ 4 步共 5 次。
        self.assertEqual(self._run(True), 5)

    def test_two_point_is_untouched(self):
        self.assertEqual(self._run(False), 8)


class _FakeEng:
    """SCORE 只回一份固定的 per-token logp：`old` 与它逐位相等 ⇒ ratio 恒 1，
    于是 loss 只剩分母的差别，可以精确比「同一份数据上两个目标谁大谁小」。"""

    def __init__(self, lp):
        self.lp = lp

    def score(self, seq):
        return list(self.lp[:len(seq)]), 0.0


class _Zo(mezo.GrpoMezo):
    def __init__(self, objective, lp, nseq=4, nseg=1000):
        self.objective = objective
        self.args = argparse.Namespace(clip=0.2, group_size=nseq)
        self.eng = _FakeEng(lp)
        self.plus_path = ""              # loss_at 会 swap(th, self.plus_path)；桩里丢掉
        self.swap = lambda th, path: None
        self._items = []
        for i in range(nseq):
            seq = list(range(nseg + 8))
            self._items.append({"seq": seq, "start": 8,
                                "old": np.zeros(nseg, dtype=np.float32),
                                "nseg": nseg, "reward": 1.0 if i else 0.0})
        for it in self._items:            # old = 前向值 ⇒ ratio = 1
            it["old"] = np.asarray(lp[8:8 + nseg], dtype=np.float32)

    def batch_at(self, step):
        return self._items


class LossScaleAcrossObjectives(unittest.TestCase):
    """零阶线没有 Adam：loss 的**标量**直接就是步长，所以两个目标的 loss 尺度必须可比。

    2026-11-14 实测（AIME 104 条轨迹，lr 5e-5 两条算法相同）：
      grpo  pj ~1e-2 … 5e-1，78 步 |b| 0 → 0.071（权重相对位移 ~1e-5，等于没动）；
      luspo pj ~1.5e3 … 2.4e3，**一步** |b| 0 → 74.5、第 17 步 139.7，rel 1.58 ⇒ adapter 打飞。
    差别正好是序列长度 T（grpo 除 token 数、luspo 除序列数 G）。反传线有 Adam 把这个
    统一因子吸收掉，零阶没有，于是「两条算法同 lr」在零阶线上不成立。
    """

    def test_token_denominator_for_both(self):
        lp = np.linspace(-0.5, -0.1, 1008)
        g = abs(_Zo("grpo", lp).loss_at(None, _Zo("grpo", lp).batch_at(0)))
        z = _Zo("luspo", lp)
        l = abs(z.loss_at(None, z.batch_at(0)))
        self.assertGreater(g, 0.0)
        # 修好之后两者是同一个 ~ |mean(adv)| 量级（差在 clip/norm 的细节，不该到倍数量级）
        self.assertLess(max(g, l) / min(g, l), 2.0,
                        f"grpo {g:.4f} vs luspo {l:.4f}：分母口径没统一（差 {max(g,l)/min(g,l):.0f}×）")

    def test_direction_unchanged(self):
        """统一分母是**全局标量**，不改变哪条轨迹被抬、哪条被压：把 adv 翻个符号，loss 只翻符号。"""
        lp = np.linspace(-0.5, -0.1, 1008)
        a = _Zo("luspo", lp)
        base = a.loss_at(None, a.batch_at(0))
        b = _Zo("luspo", lp)
        for i, it in enumerate(b.batch_at(0)):
            it["reward"] = 1.0 - it["reward"]      # 1,0,1,1 → 0,1,0,0：优势整体取反
        self.assertAlmostEqual(a.loss_at(None, a.batch_at(0)), -base, places=6)
        self.assertAlmostEqual(base, -b.loss_at(None, b.batch_at(0)), places=6)
