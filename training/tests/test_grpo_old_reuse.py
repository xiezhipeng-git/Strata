"""`--old-logprobs local` 不该为「必然等于 1 的 ratio」多跑一条前向。

背景（2026-11-18 改）：`run_grpo` 原来在 `prepared` 阶段对**每条**轨迹单独跑一次
`no_grad` 前向 + 一次全词表 `lm_head`，只为算出 old logprob；而 `--old-logprobs local`
的定义就是「old 取本 batch 开始前的策略」——epoch 1 那次前向本来就发生在任何
`opt.step()` 之前，它就是旧策略本身。于是那次前向是一个**恒等式的昂贵证明**：
ratio 必然是 1，裁剪项必然是 0，唯一产出是一段等于 1 的数。

改法是把它挪进 epoch 1 的前向里 `detach()` 复用。两个必须守住的不变量：

  1. **省一次前向**：每条轨迹的 body 调用次数 == `epochs`（不是 `epochs + 1`）。
  2. **回写 old**：epoch ≥ 2 必须继续用 epoch 1 那份 old。否则每个 epoch 都取当前
     策略的 detach，ratio 永远回到 1、`clip_frac` 永远 0 —— `--epochs 2` 就白跑了，
     而这恰恰是唯一能看到裁剪项生效的配置。

不碰 CUDA：假 body（Embedding）+ 假 head（Linear），CPU 也能跑，所以进默认测试套件。
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

import re
import types
import unittest

import torch

import grpo
import objectives
from lora import LoRALinear

V, H = 32, 8


class _Body(torch.nn.Module):
    """最小 body：只做 embedding，并记下被调用了几次。"""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(V, H)
        self.emb.weight.requires_grad_(False)     # 冻结：这段测的是「前几次向」，不是学习
        self.calls = 0

    def forward(self, input_ids=None, use_cache=False, **kw):
        self.calls += 1
        return types.SimpleNamespace(last_hidden_state=self.emb(input_ids))


class _FakeModel(torch.nn.Module):
    """`_text_model` 要 `model.model.language_model`，`run_grpo` 要 `lm_head`。

    必须是 nn.Module：`snapshot_ref_adapter` 走 `model.modules()` 找 LoRA 层。
    """

    def __init__(self, body, head):
        super().__init__()
        self.lm_head = head
        inner = torch.nn.Module()
        inner.language_model = body
        self.model = inner


def _setup(with_lora=False):
    torch.manual_seed(0)
    body = _Body()
    head = (LoRALinear(torch.nn.Linear(H, V, bias=False), r=2, alpha=4)
            if with_lora else torch.nn.Linear(H, V, bias=False))
    return body, head, _FakeModel(body, head), types.SimpleNamespace(eos_token_id=0)


def _traj(reward, toks=(4, 5, 6)):
    """两条轨迹的 completion **必须不同**。

    否则组内优势之和为 0（居中），而两条的 logp 又完全相同，`Σ adv·logp` 恒等于 0 ——
    `pg_obj` 与梯度都是 0，head 一步都不动，epoch 2 的 ratio 就永远是 1。
    第一版就是这么写的，于是这条用例在**代码正确**的情况下失败。
    """
    return {"prompt": "q", "prompt_ids": [1, 2, 3], "token_ids": list(toks),
            "content": "abc", "reward": reward}


def _run(epochs, old_logprobs="local", steps=1, kl_coef=0.0):
    body, head, model, tok = _setup()
    params = [head.weight]
    opt = torch.optim.SGD(params, lr=0.05)
    logs: list[str] = []
    trajs = [_traj(0.0, (4, 5, 6)), _traj(1.0, (7, 8, 9))]   # 组内 std > 0 且 completion 不同
    last = grpo.run_grpo(
        model, tok, trajs, objectives.GrpoCfg(),
        group=2, batch=1, max_seq=64, max_new=8,
        old_logprobs=old_logprobs, mask_truncated="on",
        epochs=epochs, steps=steps, opt=opt, params=params,
        log=logs.append, kl_coef=kl_coef)
    return body, last, logs


def _ratio_of(logs, step_epoch: str) -> float:
    for line in logs:
        if line.startswith(step_epoch):
            m = re.search(r"ratio ([0-9.]+)", line)
            if m:
                return float(m.group(1))
    raise AssertionError(f"日志里没有 {step_epoch} 的 ratio：{logs}")


class _CpuOnly(unittest.TestCase):
    """这台机器有卡 ⇒ `grpo.DEV` 是 cuda，而假模型在 CPU。

    把 DEV 挪到 cpu 是**测试内**的事，不改变训练默认值（生产上永远有卡就走 cuda）。
    """

    def setUp(self):
        self._dev = grpo.DEV
        grpo.DEV = "cpu"

    def tearDown(self):
        grpo.DEV = self._dev


class TestLocalOldIsFree(_CpuOnly):

    def test_one_forward_per_trajectory_per_epoch(self):
        """核心：2 条轨迹 × 2 epoch ⇒ body 恰好 4 次，不是 6 次。"""
        body, _last, _logs = _run(epochs=2)
        self.assertEqual(body.calls, 4)

    def test_first_epoch_ratio_is_exactly_one(self):
        """old 取自 epoch 1 的 detach ⇒ ratio 逐位等于 1（不是 1.0003 那种近似）。"""
        _body, _last, logs = _run(epochs=2)
        self.assertEqual(_ratio_of(logs, "step 1.1:"), 1.0)

    def test_old_is_written_back(self):
        """epoch 2 必须用 epoch 1 的 old：否则 ratio 会再次变成 1、裁剪永不生效。"""
        _body, last, _logs = _run(epochs=2)
        self.assertNotEqual(last["ratio"], 1.0)
        self.assertEqual((last["old_local"], last["old_engine"]), (2, 0))

    def test_single_epoch_has_no_extra_forward(self):
        """epochs=1 时省得最多：原来 2 条 × 2 次 = 4 次，现在 2 次。"""
        body, last, _logs = _run(epochs=1)
        self.assertEqual(body.calls, 2)
        self.assertEqual(last["ratio"], 1.0)


class TestEngineOldUnaffected(_CpuOnly):
    """`--old-logprobs engine` 走的是另一条路：old 来自轨迹，且 TIS 必须真算 local_old。"""

    def test_engine_tis_reuses_the_training_forward(self):
        trajs = [_traj(0.0, (4, 5, 6)), _traj(1.0, (7, 8, 9))]
        for t in trajs:
            t["token_logprobs"] = [-1.0, -2.0, -3.0]     # 与 completion 等长

        body, head, model, tok = _setup()
        logs: list[str] = []
        last = grpo.run_grpo(
            model, tok, trajs, objectives.GrpoCfg(tis_cap=2.0),
            group=2, batch=1, max_seq=64, max_new=8,
            old_logprobs="engine", mask_truncated="on",
            epochs=1, steps=1, opt=torch.optim.SGD([head.weight], lr=0.05),
            params=[head.weight], log=logs.append)

        # TIS 的分子（local_old）就是 epoch 1 那次前向：同一个「batch 开始前」的策略、
        # 同一批 token。原来为它单独跑一次 no_grad 前向，逐位重复。
        self.assertEqual(body.calls, 2)
        self.assertEqual((last["old_local"], last["old_engine"]), (0, 2))
        self.assertEqual(last["n_tis"], 2)
        self.assertNotEqual(last["ratio"], 1.0)
        self.assertNotAlmostEqual(last["tis"], 1.0, places=3)   # tis 真的算了，不是占位 1


if __name__ == "__main__":
    unittest.main()
