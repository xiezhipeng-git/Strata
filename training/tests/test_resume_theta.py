"""resume 的 θ 必须是**全层**骨架，只把 npz 里训过的层盖上去。

踩过的坑（`--zo-layers 1` + `--resume`）：npz 只存被训过的层（这里只有第 47 层），
直接拿它当 self.theta，会在 `swap(self.theta, final.gguf)` 处炸 `KeyError: 0` ——
`write_adapter` 按 layout 全层写，引擎 `lora_update` 也要求张量集合全程不变。
更隐蔽的是引擎其实已经起来了（init.gguf 是单独铺全层写的），所以报错发生在
`before:` 之前，日志看起来像「训练刚开始就莫名挂了」。

这里钉住三件事：
  1. 部分层 dict 直接写盘确实会 KeyError（把约束本身说清楚，别让人以为是偶发）；
  2. `init_theta` 铺骨架 + 覆盖恢复层 ⇒ 写盘成功，恢复层逐位相同、其余层 B=0；
  3. `_zo_step` 那套 `{**self.theta, **th}` 补全在 resume 的 θ 上同样成立。
"""

import pathlib as _p
import sys as _s

for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

import tempfile
import unittest

import numpy as np

import mezo  # noqa: F401  （把 tools/ 放进 sys.path：gguf_reader / gguf_writer）
from common.lora import (adapter_layout, init_theta, read_adapter_alpha,
                         read_adapter_rank, write_adapter)
from gguf_reader import GGUFFile

GEOM = {
    "block_count": 4,
    "embedding_length": 8,
    "attention.head_count": 2,
    "attention.key_length": 4,
    "full_attention_interval": 2,
    "ssm.inner_size": 6,
}
LAYOUT = adapter_layout(GEOM)


def _saved_layers():
    """模拟 npz：`--zo-layers 1` 只留最后一层，且它的 B 非零。"""
    trained = max(LAYOUT)
    which = next(iter(LAYOUT[trained]))
    full = init_theta(LAYOUT, rank=1, seed=0)
    b = np.full_like(full[trained][which][1], 0.125)
    return {trained: {which: (full[trained][which][0], b)}}, full


class TestResumeTheta(unittest.TestCase):
    def test_partial_theta_cannot_be_written(self):
        partial, _ = _saved_layers()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(KeyError):
                write_adapter(str(_p.Path(td) / "partial.gguf"), LAYOUT, partial, 16.0)

    def test_full_skeleton_with_restored_layers(self):
        partial, initial = _saved_layers()
        theta = init_theta(LAYOUT, rank=1, seed=0)
        for l, v in partial.items():
            theta[l] = v

        self.assertEqual(len(theta), len(LAYOUT))
        trained, which = max(LAYOUT), next(iter(LAYOUT[max(LAYOUT)]))
        np.testing.assert_array_equal(theta[trained][which][1], partial[trained][which][1])
        for l in LAYOUT:
            if l == trained:
                continue
            for w in LAYOUT[l]:
                self.assertFalse(theta[l][w][1].any(), f"layer {l} 的 B 应仍是 0")
                np.testing.assert_array_equal(theta[l][w][0], initial[l][w][0])

        with tempfile.TemporaryDirectory() as td:
            out = str(_p.Path(td) / "resumed.gguf")
            write_adapter(out, LAYOUT, theta, 16.0)
            names = {t.name for t in GGUFFile(out).tensors}
            for l in LAYOUT:
                for w in LAYOUT[l]:
                    self.assertIn(f"blk.{l}.{w}.weight.lora_a", names)
                    self.assertIn(f"blk.{l}.{w}.weight.lora_b", names)

    def test_step_swap_completion_still_works(self):
        """`_zo_step` 的 th_plus 只带训过的层；{**self.theta, **th} 补成全层。"""
        theta = init_theta(LAYOUT, rank=1, seed=0)
        trained, which = max(LAYOUT), next(iter(LAYOUT[max(LAYOUT)]))
        b = np.full_like(theta[trained][which][1], -0.25)
        th_plus = {trained: {which: (theta[trained][which][0], b)}}
        full = th_plus if len(th_plus) == len(LAYOUT) else {**theta, **th_plus}
        with tempfile.TemporaryDirectory() as td:
            write_adapter(str(_p.Path(td) / "plus.gguf"), LAYOUT, full, 16.0)
        self.assertEqual(len(full), len(LAYOUT))
        np.testing.assert_array_equal(full[trained][which][1], b)


class TestAdapterShapeContract(unittest.TestCase):
    """引擎启动时把 LoRA shape 捕获进 CUDA graph，运行期 LORA 只收同 shape。

    探针（bias_probe.py / prefix_scan.py / ce_sweep.py）必须用**目标 adapter 的 rank**
    写 B=0 的启动适配器；写死 rank=1 会在测 rank-4 权重时炸
    "layer 0 changed shape (the tables' addresses are held by captured graphs)"。
    """

    def test_rank_and_alpha_roundtrip(self):
        for rank, alpha in ((1, 16.0), (4, 8.0), (16, 2.0)):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as td:
                out = str(_p.Path(td) / "adapter.gguf")
                write_adapter(out, LAYOUT, init_theta(LAYOUT, rank, 0), alpha)
                self.assertEqual(read_adapter_rank(out), rank)
                self.assertEqual(read_adapter_alpha(out), alpha)

    def test_zero_control_matches_target_shape(self):
        """同 rank 的 B=0 对照组，每个 lora_a/lora_b 的 shape 必须与目标逐项相同。"""
        with tempfile.TemporaryDirectory() as td:
            target = str(_p.Path(td) / "target.gguf")
            write_adapter(target, LAYOUT, init_theta(LAYOUT, 4, 0), 16.0)
            # 训练终点的 B 非零，但 shape 与 rank 无关地一致
            theta = init_theta(LAYOUT, 4, 0)
            for l in LAYOUT:
                for w in LAYOUT[l]:
                    theta[l][w] = (theta[l][w][0], np.full_like(theta[l][w][1], 0.3))
            write_adapter(target, LAYOUT, theta, 16.0)

            control = str(_p.Path(td) / "control.gguf")
            write_adapter(control, LAYOUT, init_theta(LAYOUT, read_adapter_rank(target), 0),
                          read_adapter_alpha(target))
            self.assertEqual(
                {t.name: tuple(t.shape) for t in GGUFFile(target).tensors},
                {t.name: tuple(t.shape) for t in GGUFFile(control).tensors})


if __name__ == "__main__":
    unittest.main()
