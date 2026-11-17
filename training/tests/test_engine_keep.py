"""`encode.engine_keep`：交给引擎的行掩码必须刚好覆盖 loss 会读的位置。

判据只有一条 —— **`answer_seg` 读到的每一个位置都在掩码里**。少一个，那条 token 的 logp 就
不会被算（trainer 拿到上一次读的残值）；多一个只是白算。两条路（单轮 / 多轮）都要成立，因为
full-context 用的是单轮 jsonl，而多轮轨迹走 region。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."),
                os.path.join(os.path.dirname(__file__), "..", "mezo")]

from common.encode import answer_seg, engine_keep  # noqa: E402


def _check(it, sel=None):
    n = len(it["seq"])
    keep = engine_keep(it, sel)
    assert keep is not None, "掩码不该是 None：None 意味着整段都过 head，长上下文每步白算 15 s"
    assert keep.size == n - 1, f"掩码 {keep.size} vs 引擎位置数 {n - 1}"
    seg = answer_seg(it, np.arange(n - 1, dtype=np.float32))
    # answer_seg 的位置：单轮是 arr[start:] 的下标区间，多轮是 region[1:] 为真的下标
    pos = (np.arange(it["start"], n - 1) if it.get("region") is None
           else np.flatnonzero(np.asarray(it["region"], dtype=bool)[1:]))
    assert np.array_equal(np.asarray(seg, dtype=np.int64), pos)
    feasible = np.zeros(n - 1, dtype=bool)
    # `loss_at` 在 sel 为 None 时读整段，否则读 `seg[sel]` —— 掩码必须盖住后者
    feasible[pos if sel is None else pos[np.asarray(sel, dtype=bool)]] = True
    assert np.all(keep[feasible]), "有位置会被 loss 读，却不在掩码里"
    return keep, feasible


def test_single_turn_masks_the_prompt():
    """单轮：sel=None 也要掩 —— prompt 那些行一个都不进 loss。"""
    it = {"seq": np.arange(10), "start": 6}
    keep, feasible = _check(it)
    assert not keep[:6].any(), "prompt 行不该要求引擎算"
    assert keep.sum() == feasible.sum()


def test_single_turn_sel_narrows_within_the_answer():
    it = {"seq": np.arange(10), "start": 6}
    sel = np.array([True, False, True])
    keep, feasible = _check(it, sel)
    assert keep.sum() == 2, "被 sel 关掉的位置不该要求引擎算"
    assert np.array_equal(keep[6:9], sel)


def test_transcript_region_masks_the_turns():
    region = np.zeros(12, dtype=bool)
    region[5:7] = True
    region[9:11] = True
    it = {"seq": np.arange(12), "region": region}
    keep, feasible = _check(it)
    assert keep.sum() == feasible.sum() == 4
    assert not keep[0]


def test_sel_size_mismatch_is_loud():
    it = {"seq": np.arange(10), "start": 6}
    try:
        engine_keep(it, np.array([True, True]))       # 答案段是 4 个位置
    except SystemExit:
        return
    raise AssertionError("掩码尺寸不对时必须报错，不能静默截断")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
