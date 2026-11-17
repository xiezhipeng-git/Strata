"""`inject_lora` 的 target 匹配：叶子名 vs 点号后缀。

为什么值得一条测试：`q_proj` 这种叶子名会把 `self_attn.indexer.q_proj` 一起挂上，而引擎的
adapter 表（`src/kernels/lora_adapters.cpp` 的 `qsa_projs` / `gdn_projs`）里没有 indexer ——
那些参数训了导不出去，训练侧更新与引擎读到的 adapter 就不是同一个模型。症状很隐蔽：
loss 正常下降、导出「成功」，但换进引擎后行为与训练侧对不上。
"""

import pathlib as _p
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))

from lora import inject_lora  # noqa: E402


class _QsaLayer(nn.Module):
    """一层 QSA：`self_attn` 下 q/k/v/o 四个投影 + indexer 的两个（引擎不适配后者）。"""

    def __init__(self, n: int = 8):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(n, 2 * n, bias=False)
        self.self_attn.k_proj = nn.Linear(n, n, bias=False)
        self.self_attn.v_proj = nn.Linear(n, n, bias=False)
        self.self_attn.o_proj = nn.Linear(n, n, bias=False)
        self.self_attn.indexer = nn.Module()
        self.self_attn.indexer.q_proj = nn.Linear(n, 4, bias=False)
        self.self_attn.indexer.k_proj = nn.Linear(n, 4, bias=False)


class _GdnLayer(nn.Module):
    """一层 GDN：`linear_attn` 下 in_proj_qkv / in_proj_z / out_proj（引擎的 attn_qkv / attn_gate / ssm_out）。"""

    def __init__(self, n: int = 8):
        super().__init__()
        self.linear_attn = nn.Module()
        self.linear_attn.in_proj_qkv = nn.Linear(n, 4 * n, bias=False)
        self.linear_attn.in_proj_z = nn.Linear(n, 2 * n, bias=False)
        self.linear_attn.out_proj = nn.Linear(2 * n, n, bias=False)


def _model() -> nn.Module:
    layers = nn.ModuleList()
    for i in range(6):
        layers.append(_QsaLayer() if i % 3 == 2 else _GdnLayer())
    return nn.Sequential(layers)


def _adapted(m: nn.Module) -> list[str]:
    return sorted(k[:-2] for k in m.state_dict() if k.endswith(".A"))


class TargetMatching(unittest.TestCase):
    def test_leaf_names_also_hit_indexer(self):
        """记录现状：叶子名会误挂 indexer —— 所以新写法（点号后缀）是必需的，不是风格偏好。"""
        m = _model()
        inject_lora(m, targets=("q_proj", "k_proj"), r=4, alpha=8)
        self.assertTrue(any("indexer" in n for n in _adapted(m)),
                        "叶子名 q_proj / k_proj 会连 indexer 一起挂上")

    def test_dotted_suffix_leaves_indexer_alone(self):
        m = _model()
        n = inject_lora(m, targets=("self_attn.q_proj", "self_attn.k_proj"), r=4, alpha=8)
        got = _adapted(m)
        self.assertEqual(len(got), 4)          # 6 层里 2 层是 QSA，每层 q + k
        self.assertEqual(n, 4)
        self.assertFalse([x for x in got if "indexer" in x])

    def test_full_engine_set_matches_the_seven_projections(self):
        """训练脚本用的那七个后缀 ⇒ 每层按引擎的投影集合挂满（GDN 3 个 / QSA 4 个）。"""
        m = _model()
        n = inject_lora(m, targets=(
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
        ), r=4, alpha=8)
        got = _adapted(m)
        self.assertEqual(len(got), 3 * 4 + 4 * 2, "GDN 层 3 个 × 4 + QSA 层 4 个 × 2")
        self.assertEqual(n, len(got))
        self.assertFalse([x for x in got if "indexer" in x])

    def test_zero_matches_is_an_error_not_a_silent_noop(self):
        """挂点写错（例如把完整路径当叶子名）时返回 0 —— 调用方必须自己当错误处理。

        这里只固定「什么都不匹配就是 0」这个事实：sft.py 打印 `lora: N modules`，N=0 时
        训练会跑完全程却什么都没更新（2026-11-17 实测过一次，15 步全白跑）。
        """
        m = _model()
        n = inject_lora(m, targets=("self_attn.q_proj", "nosuch.proj"), r=4, alpha=8)
        self.assertEqual(n, 2)
        n0 = inject_lora(_model(), targets=("model.layers.0.self_attn.q_proj",), r=4, alpha=8)
        self.assertEqual(n0, 0, "整条路径当 target 时按后缀匹配不上 ⇒ 0，调用方要报错")


if __name__ == "__main__":
    unittest.main()
