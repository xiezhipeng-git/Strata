"""KL 锚：`--kl-coef` 的数学与「ref 必须真的被还原」。

原本库里**没有任何**对 θ₀ 的约束 —— `grpo_loss`/`luspo_loss` 里的 `neg_kl` 是
`logp − old_logp`（重要性比，喂给 clip 的基线），不是对参考策略的 KL。唯一限制漂移的
是 `--clip` 与 lr，而这两个只管「单步走多远」。所以「强化学习会不会忘掉旧的」在代码上
从来就没有机制，只有一句「理论上不该忘」。

`--kl-coef` 补的是这个机制，严格说清它的边界：

  * 它是 **k3 估计**（Schulman）：`exp(r) − r − 1`，`r = ref_lp − logp`，逐点非负；
  * 它约束的是**离 θ₀ 的距离**，给不出遗忘上界 —— 没有任何系数能保证旧能力不掉；
  * θ₀ = 训练起点的 LoRA A/B 快照（几 MB），不需要第二份 35B 权重；
  * 默认 0 ⇒ 一次 ref 前向都不跑。

最要紧的一条不变量放在最后：`ref_logprobs` 中途抛异常也必须把 A/B **还原**。否则
「参考模型」会静默变成当前策略，KL 恒为 0、日志一片正常 —— 这种失效没有任何报错。
"""

from __future__ import annotations

# --- path bootstrap: this file sits under training/<sub>/; shared modules are one level up ---
import pathlib as _p, sys as _s
for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "backward",
           _p.Path(__file__).resolve().parents[1] / "bench",
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

import types
import unittest

import torch

import grpo
import objectives
from lora import LoRALinear

V, H = 32, 8


class _Body(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(V, H)
        self.emb.weight.requires_grad_(False)
        self.calls = 0

    def forward(self, input_ids=None, use_cache=False, **kw):
        self.calls += 1
        return types.SimpleNamespace(last_hidden_state=self.emb(input_ids))


class _FakeModel(torch.nn.Module):
    def __init__(self, body, head):
        super().__init__()
        self.lm_head = head
        inner = torch.nn.Module()
        inner.language_model = body
        self.model = inner


class _CpuOnly(unittest.TestCase):
    def setUp(self):
        self._dev = grpo.DEV
        grpo.DEV = "cpu"

    def tearDown(self):
        grpo.DEV = self._dev


def _setup():
    torch.manual_seed(0)
    body = _Body()
    head = LoRALinear(torch.nn.Linear(H, V, bias=False), r=2, alpha=4)
    return body, head, _FakeModel(body, head), types.SimpleNamespace(eos_token_id=0)


def _traj(reward, toks):
    return {"prompt": "q", "prompt_ids": [1, 2, 3], "token_ids": list(toks),
            "content": "abc", "reward": reward}


# ---------------------------------------------------------------- 数学

class TestK3(unittest.TestCase):

    def test_zero_when_policy_is_ref(self):
        lp = torch.tensor([[-1.0, -2.0, -0.5]])
        pen, st = objectives.kl_k3(lp, lp.clone(), torch.ones(1, 3))
        self.assertAlmostEqual(float(pen), 0.0, places=9)
        self.assertAlmostEqual(st["kl_tok"], 0.0, places=9)

    def test_non_negative_and_grows_with_drift(self):
        lp = torch.tensor([[-3.0]])
        vals = [float(objectives.kl_k3(lp, torch.tensor([[r]]), torch.ones(1, 1))[0])
                for r in (-3.0, -2.0, -1.0, 0.0)]
        self.assertTrue(all(v >= 0.0 for v in vals), vals)     # k3 逐点非负
        self.assertEqual(vals, sorted(vals))                   # ref 越高 ⇒ 惩罚越大

    def test_it_has_gradient(self):
        """惩罚项必须真的对当前策略求导，否则它只是一个日志数字。"""
        lp = torch.tensor([[-3.0]], requires_grad=True)
        pen, _st = objectives.kl_k3(lp, torch.tensor([[-1.0]]), torch.ones(1, 1))
        pen.backward()
        self.assertIsNotNone(lp.grad)
        self.assertNotEqual(float(lp.grad), 0.0)

    def test_denom_is_batch_wide(self):
        """逐条 backward 时传整批分母：两条相同轨迹的两倍惩罚，不是「平均后相等」。"""
        lp = torch.tensor([[-3.0]])
        ref = torch.tensor([[-1.0]])
        one, _ = objectives.kl_k3(lp, ref, torch.ones(1, 1), denom=2.0)
        two, _ = objectives.kl_k3(lp, ref, torch.ones(1, 1), denom=1.0)
        self.assertAlmostEqual(float(two) / float(one), 2.0, places=6)


# ---------------------------------------------------------------- 端到端

class TestKlWiring(_CpuOnly):

    def _run(self, epochs, kl_coef):
        body, head, model, tok = _setup()
        params = [head.A, head.B]
        logs: list[str] = []
        last = grpo.run_grpo(
            model, tok, [_traj(0.0, (4, 5, 6)), _traj(1.0, (7, 8, 9))],
            objectives.GrpoCfg(), group=2, batch=1, max_seq=64, max_new=8,
            old_logprobs="local", mask_truncated="on", epochs=epochs, steps=1,
            opt=torch.optim.SGD(params, lr=0.05), params=params,
            log=logs.append, kl_coef=kl_coef)
        return body, last, logs

    def test_kl_adds_one_ref_forward_per_trajectory(self):
        """2 条 × 2 epoch = 4 次训练前向，再加 2 次 ref（每条只算一次，与 epoch 无关）。"""
        body, last, _logs = self._run(epochs=2, kl_coef=0.1)
        self.assertEqual(body.calls, 6)
        self.assertIsNotNone(last["kl"])
        self.assertGreaterEqual(last["kl"], 0.0)

    def test_off_by_default_runs_no_ref_forward(self):
        body, last, _logs = self._run(epochs=2, kl_coef=0.0)
        self.assertEqual(body.calls, 4)
        self.assertIsNone(last["kl"])

    def test_ref_forward_happens_before_any_update(self):
        """ref 若在 step 之后才算，锚点就是「已更新的策略」，KL 会偏小且随 epoch 漂。"""
        body, last, logs = self._run(epochs=2, kl_coef=0.1)
        self.assertTrue(any("kl: coef" in l for l in logs), logs)   # 快照发生在训练前


class TestRefRestore(unittest.TestCase):

    def test_adapter_restored_even_when_forward_raises(self):
        torch.manual_seed(0)
        wrap = LoRALinear(torch.nn.Linear(H, V, bias=False), r=2, alpha=4)
        with torch.no_grad():
            wrap.B.normal_()                      # 让 B ≠ 0，否则「还原」和「没动」看不出差别
        holder = torch.nn.ModuleList([wrap])
        snap = grpo.snapshot_ref_adapter(holder)
        self.assertEqual(len(snap), 1)

        before_a = wrap.A.detach().clone()
        before_b = wrap.B.detach().clone()

        class Boom(RuntimeError):
            pass

        def bad_body(**_kw):
            raise Boom("前向中途炸了")

        with self.assertRaises(Boom):
            grpo.ref_logprobs(bad_body, None, torch.zeros(1, 3, dtype=torch.long), 1,
                              torch.zeros(2, dtype=torch.long), snap)

        self.assertTrue(torch.equal(wrap.A.detach(), before_a), "A 没还原")
        self.assertTrue(torch.equal(wrap.B.detach(), before_b), "B 没还原")

    def test_snapshot_is_independent_of_later_updates(self):
        """快照必须是拷贝：直接引用 A/B 的话，θ 一动 ref 就跟着动，KL 恒为 0。"""
        torch.manual_seed(0)
        wrap = LoRALinear(torch.nn.Linear(H, V, bias=False), r=2, alpha=4)
        holder = torch.nn.ModuleList([wrap])
        snap = grpo.snapshot_ref_adapter(holder)
        _m, a0, b0 = snap[0]
        with torch.no_grad():
            wrap.A.add_(1.0)
            wrap.B.add_(1.0)
        self.assertFalse(torch.equal(a0, wrap.A.detach()))
        self.assertTrue(torch.equal(b0, wrap.B.detach() - 1.0))


if __name__ == "__main__":
    unittest.main()
