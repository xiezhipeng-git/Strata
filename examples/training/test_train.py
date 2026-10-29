"""`train.py` 的三层门面测试：数据源解析、算法注册表、GRPO 目标（零阶要的那个标量）。

只在 CPU 上跑，不加载模型：`python3 -m unittest test_train -v`（在 examples/training 下）。

这里能测的三件事恰好是最容易静默出错的三件事：
  1. 数据切错 —— `--data identity` 应该落在身份数据上，路径写错要在**加载模型之前**就报；
  2. 算法/目标组合不成立 —— 比如 `--algo sft --objective grpo`，不能等到跑了一半才发现；
  3. GRPO 的标量目标 —— old == new 时必须正好等于 `-mean(A)`，超出裁剪带必须被夹住。
     这个函数是 MeZO×GRPO 的全部数学，写错了训练照样"跑得通"。
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

import algos
import datasets
import sft_mezo


class Data(unittest.TestCase):
    def test_identity_is_the_default(self):
        self.assertEqual(datasets.resolve(""), ("identity", ""))
        self.assertEqual(datasets.resolve("identity"), ("identity", ""))
        self.assertEqual(datasets.resolve("identity:Qwen"), ("identity", "Qwen"))

    def test_paths_are_typed_by_extension(self):
        self.assertEqual(datasets.resolve("/root/a.jsonl"), ("jsonl", "/root/a.jsonl"))
        self.assertEqual(datasets.resolve("/root/a.npy"), ("pretrain", "/root/a.npy"))
        self.assertEqual(datasets.resolve("jsonl:/root/a"), ("jsonl", "/root/a"))

    def test_a_typo_fails_before_the_model_loads(self):
        with self.assertRaises(SystemExit):
            datasets.resolve("identiy")
        with self.assertRaises(SystemExit):
            datasets.resolve("/root/a.txt")


class Registry(unittest.TestCase):
    def test_every_algo_points_at_a_real_file(self):
        for algo in algos.ALGOS.values():
            self.assertTrue((Path(__file__).parent / algo.script).exists(), algo.script)

    def test_mezo_is_the_only_zero_order_route_and_accepts_grpo(self):
        self.assertEqual([a.name for a in algos.ALGOS.values() if a.kind == "zeroth_order"],
                         ["mezo"])
        self.assertIn("grpo", algos.ALGOS["mezo"].objectives)
        self.assertNotIn("grpo", algos.ALGOS["sft"].objectives)

    def test_describe_lists_both_axes(self):
        text = algos.describe()
        for word in ("sft", "mezo", "grpo", "ce"):
            self.assertIn(word, text)


class _Tok:
    """只有模板和编码，够 `load_rollout_groups` 用。"""

    def apply_chat_template(self, messages, **kw):
        return [1, 2, 3]

    def __call__(self, text, **kw):
        return type("E", (), {"input_ids": [4, 5]})()


class Grouping(unittest.TestCase):
    """轨迹 jsonl → 组。组内归一化的基线只在这一层成立。"""

    def load(self, rows):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            return sft_mezo.load_rollout_groups(str(p), _Tok())

    def test_consecutive_same_prompt_is_one_group_and_advantages_are_normalised(self):
        groups, meta = self.load([
            {"prompt": "P1", "content": "a", "target": "", "kind": "words:1"},
            {"prompt": "P1", "content": "b b", "target": "", "kind": "words:1"},
            {"prompt": "P2", "content": "c", "target": "", "kind": "words:1"},
        ])
        self.assertEqual(len(groups), 2)
        self.assertEqual(meta["used"], 1)                     # P1 有方差；P2 只有一条
        advs = [t.adv for t in groups[0]]
        self.assertAlmostEqual(sum(advs), 0.0, places=5)      # 归一化后均值为 0
        self.assertGreater(advs[0], 0)                        # 1 个词是对的（奖励 1）
        self.assertEqual([t.adv for t in groups[1]], [0.0])   # 单条组没有基线

    def test_a_group_without_variance_has_no_gradient(self):
        groups, meta = self.load([{"prompt": "P", "content": "a", "kind": "words:1"}] * 2)
        self.assertEqual(meta["used"], 0)
        self.assertEqual([t.adv for t in groups[0]], [0.0, 0.0])

    def test_engine_logprobs_are_used_when_they_line_up(self):
        groups, _ = self.load([{"prompt": "P", "content": "a", "kind": "words:1",
                                "token_logprobs": [-0.5, -0.25]}])
        self.assertEqual([round(float(x), 4) for x in groups[0][0].old], [-0.5, -0.25])

    def test_a_short_logprob_list_is_ignored(self):
        # 长度对不上就是另一套分词，宁可重算也不能错位相减
        groups, _ = self.load([{"prompt": "P", "content": "a", "kind": "words:1",
                                "token_logprobs": [-0.5]}])
        self.assertIsNone(groups[0][0].old)


class _Out:
    def __init__(self, h):
        self.last_hidden_state = h


class GRPOLoss(unittest.TestCase):
    """`grpo_loss` 只要一个标量 —— 所以它错起来也最安静。"""

    V = 5

    def setUp(self):
        self.h = torch.zeros(1, 4, self.V)            # 均匀 logits：new logp = -log V
        self.ids = torch.tensor([[9, 9, 0, 1]])       # 前 2 个 token 是 prompt
        self.new = -math.log(self.V)
        self.head = torch.nn.Identity()

    def traj(self, old: float, adv: float) -> sft_mezo._Traj:
        return sft_mezo._Traj(self.ids, 2, torch.full((2,), old), adv, 0.0)

    def loss(self, t, clip=0.2):
        body = type("B", (), {"__call__": lambda _self, input_ids, use_cache=False:
                              _Out(self.h)})()
        return sft_mezo.grpo_loss(body, self.head, [t], clip)

    def test_old_equals_new_is_exactly_minus_advantage(self):
        # ratio = 1 ⇒ 代理目标就是 A，loss = −A。adv 取负值：loss 必须是 +2
        value, info = self.loss(self.traj(self.new, -2.0))
        self.assertAlmostEqual(value, 2.0, places=5)
        self.assertAlmostEqual(info["ratio"], 1.0, places=5)
        self.assertEqual(info["dead"], 0.0)

    def test_zero_advantage_contributes_nothing(self):
        value, info = self.loss(self.traj(self.new, 0.0))
        self.assertEqual(value, 0.0)
        self.assertEqual(info["dead"], 1.0)

    def test_the_upper_clamp_bites(self):
        # ratio = 5（old 比 new 低 ln5）落在 [0.8, 1.2] 外 ⇒ min 取裁剪项 1.2，而不是 5
        old = self.new - math.log(5.0)
        value, info = self.loss(self.traj(old, 1.0), clip=0.2)
        self.assertAlmostEqual(value, -1.2, places=4)
        self.assertAlmostEqual(info["ratio"], 5.0, places=3)

    def test_without_clamp_the_ratio_passes_through(self):
        old = self.new - math.log(5.0)
        value, _ = self.loss(self.traj(old, 1.0), clip=10.0)
        self.assertAlmostEqual(value, -5.0, places=4)


if __name__ == "__main__":
    unittest.main()
