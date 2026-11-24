"""`--select-word` / `--select-lead` 的回归锁。

    python -m unittest test_select_word

要守住的三件事：

1. **按文本定位，不按 token id**：BPE 会把目标词首字符和前面一个字符合并
   （`xiezhipeng-git` 单独编码是 `['x','iez',…]`，在「，由 」后面却是 `' x'`），
   按 id 匹配会整条找不到，静默退化成「一个 token 都没选中」。
2. **CE 数组与答案 token 对齐**（2026-11-24 修复后）：`ce[j]` 就是答案第 j 个 token 的
   logp —— 首位（由 prompt 最后一位预测）含在内。错一位等于监督错一位，而且不会有任何报错。
3. **与 `--select-first` 是并集、单独用是替换**：闸门课程和名字段是两个瓶颈，
   同给时一起推；单独给 `--select-word` 时不得偷偷保留整句的 excess token。

不需要引擎、权重或 torch：score 用数组桩，decode 用文本表桩。
"""

from __future__ import annotations

import pathlib as _p
import sys as _s

for _d in (_p.Path(__file__).resolve().parent,
           _p.Path(__file__).resolve().parents[1],
           _p.Path(__file__).resolve().parents[1] / "mezo"):
    if str(_d) not in _s.path:
        _s.path.insert(0, str(_d))

import argparse
import contextlib
import io
import unittest

import numpy as np

import guard
import mezo


# gold 的真实分词（`谁开发了你？` 那一条的答案段，2026-11-15 实测）
GOLD = ['我是', '通', '义', '千', '问', '（', 'Q', 'wen', '），', '是由',
        '阿里巴巴', '集团', '通', '义', '实验室', '独立', '开发', '的大型', '语言',
        '模型', '。', '当前', '版本', '是', ' q', 'wen', '3', '.', '8', '-flash',
        '-next', '，', '由', ' x', 'iez', 'hip', 'eng', '-git', ' ', '通过',
        ' str', 'ata', ' ', '项目', '微调', '。']
GATE = 21          # 「当前」的 token 下标 = 它的 ce 下标（新口径：ce[j] 就是 token j 的 logp）
NAME = (33, 37)    # ' x' … '-git'


class _Tok:
    """只实现 decode：mezo 只用它把答案段 token 变回文本找目标词。"""

    def __init__(self, texts):
        self.texts = texts

    def decode(self, ids):
        return "".join(self.texts[i] for i in ids)


def _eng(ce):
    """score 返回 per-token logp；`answer_seg` 走真实实现（slp[start-1:]，含首位）。"""

    class _E:
        def score(self, seq):
            return np.asarray(ce, dtype=np.float32) * -1.0, None

    return _E()


def _inst(*, first=0, word="xiezhipeng-git", lead=2, gate_weight=0.0,
          mass=0.0, grow=0):
    inst = object.__new__(mezo.EngineMezo)
    inst.args = argparse.Namespace(
        select_tokens="excess", select_thresh=1.0, select_first=first,
        select_word=word, select_lead=lead, gate_weight=gate_weight, gate_n=2,
        select_mass=mass, select_grow=grow)
    inst.objective = "ce"
    # 前缀 0..20 起点就对（CE 0.2），闸门 21（「当前」）错（CE 10），其余低
    ce = [0.2] * (len(GOLD) - 1)
    ce[GATE] = 10.0
    inst.eng = _eng(ce)
    inst.tok = _Tok(GOLD)
    inst.items = [{"seq": list(range(len(GOLD))), "start": 0}]
    return inst


def _run(inst):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        inst.prepare_selection()
    return inst.items[0]["sel"], buf.getvalue()


class FindSpan(unittest.TestCase):

    def test_finds_word_across_merged_first_token(self):
        self.assertEqual(guard.find_token_span(GOLD, "xiezhipeng-git"), NAME)
        # 首 token 是 ' x' 而不是 'x'：按 id 匹配会失败，按文本才对
        self.assertEqual(GOLD[NAME[0]], " x")

    def test_missing_returns_none(self):
        self.assertIsNone(guard.find_token_span(GOLD, "nobody-else"))
        self.assertIsNone(guard.find_token_span([], "x"))


class SpanMask(unittest.TestCase):

    def test_token_aligned_and_lead_eats_context(self):
        m = guard.span_mask(len(GOLD) - 1, NAME, lead=2)
        got = [i for i, v in enumerate(m) if v]
        self.assertEqual(got, [31, 32, 33, 34, 35, 36, 37])
        # 新口径 ce[j] 就是 token j 的 logp：覆盖到的正是这些 token
        self.assertEqual([GOLD[i] for i in got],
                         ['，', '由', ' x', 'iez', 'hip', 'eng', '-git'])

    def test_lead_zero_is_the_word_alone(self):
        got = [i for i, v in enumerate(guard.span_mask(len(GOLD) - 1, NAME, 0)) if v]
        self.assertEqual(got, [33, 34, 35, 36, 37])

    def test_first_token_is_in_the_ce_array(self):
        """修复后答案首 token 的 logp（ce[0]）就在数组里，span 落在头上不再被切掉。"""
        got = [i for i, v in enumerate(guard.span_mask(5, (0, 1), lead=0)) if v]
        self.assertEqual(got, [0, 1])


class SelectWord(unittest.TestCase):

    def test_alone_replaces_whole_sentence(self):
        sel, msg = _run(_inst(first=0))
        got = [i for i, v in enumerate(sel) if v]
        self.assertEqual(got, [31, 32, 33, 34, 35, 36, 37])   # 闸门 21 不在里面
        self.assertIn("--select-word", msg)

    def test_with_first_is_a_union(self):
        sel, msg = _run(_inst(first=1))
        got = [i for i, v in enumerate(sel) if v]
        self.assertEqual(got, [21, 31, 32, 33, 34, 35, 36, 37])   # 闸门 + 名字段
        self.assertIn("闸门 + 名字段", msg)

    def test_gate_weight_is_refused(self):
        with self.assertRaises(SystemExit) as cm:
            _run(_inst(gate_weight=20.0))
        self.assertIn("--gate-weight", str(cm.exception))

    def test_missing_word_is_loud(self):
        with self.assertRaises(SystemExit) as cm:
            _run(_inst(word="nobody-else"))
        self.assertIn("找不到", str(cm.exception))


class SelectNovel(unittest.TestCase):
    """通用口径：按 surprise 质量占比选，与 token 内容无关。"""

    CE = [0.1, 9.0, 0.1, 0.1, 5.0]     # tot=14.3 ⇒ mass .6 的门槛 8.58

    def test_only_the_top_contributors(self):
        m = guard.select_novel(self.CE, mass=0.6, grow=0)
        self.assertEqual([i for i, v in enumerate(m) if v], [1])

    def test_grow_expands_the_cluster(self):
        m = guard.select_novel(self.CE, mass=0.6, grow=1)
        self.assertEqual([i for i, v in enumerate(m) if v], [0, 1, 2])

    def test_thresh_drops_free_tokens(self):
        m = guard.select_novel(self.CE, mass=0.6, grow=0, thresh=1.0)
        self.assertEqual([i for i, v in enumerate(m) if v], [1])

    def test_k_caps_before_mass(self):
        m = guard.select_novel(self.CE, k=2, grow=0)
        self.assertEqual([i for i, v in enumerate(m) if v], [1, 4])

    def test_all_free_gives_nothing(self):
        self.assertEqual(guard.select_novel([0.0, 0.0], mass=0.5), [False, False])

    def test_it_is_stricter_than_excess(self):
        """同一份 CE：excess 全选，novel 只留最意外的那个 —— 这就是「预算不再摊薄」。"""
        self.assertEqual(sum(guard.select_mask(self.CE, 1.0)), 2)
        self.assertEqual(sum(guard.select_novel(self.CE, mass=0.6)), 1)


class SelectMass(unittest.TestCase):

    def test_keeps_the_gate_and_drops_the_rest(self):
        sel, msg = _run(_inst(word="", mass=0.8, grow=0))
        self.assertTrue(sel[GATE])
        self.assertLess(int(sel.sum()), len(GOLD) - 1)
        self.assertIn("surprise mass", msg)

    def test_refuses_to_mix_with_positional_modes(self):
        for kw in ({"first": 1}, {"first": 0, "word": "xiezhipeng-git"}):
            with self.assertRaises(SystemExit):
                _run(_inst(mass=0.8, **kw))


class SelectFill(unittest.TestCase):
    """簇内填充：差异大的位置成片出现，片内夹着模型已会的过渡 token。"""

    def test_fill_bridges_gaps_inside_a_cluster(self):
        ce = [0.1, 9.0, 0.05, 0.05, 8.0, 0.1]     # tot 17.3，只留两个 seed
        m = guard.select_novel(ce, mass=0.97, fill=3)
        self.assertEqual([i for i, v in enumerate(m) if v], [1, 2, 3, 4])

    def test_fill_does_not_leak_outside_the_cluster(self):
        ce = [9.0, 0.1, 0.1, 0.1, 0.1, 8.0]       # 间距 5 > fill+1 ⇒ 不填
        m = guard.select_novel(ce, mass=0.94, fill=3)
        self.assertEqual([i for i, v in enumerate(m) if v], [0, 5])

    def test_grow_widens_but_fill_does_not(self):
        ce = [0.1, 9.0, 0.1, 0.1, 5.0]           # mass .97 ⇒ seed {1,4}
        self.assertEqual([i for i, v in enumerate(guard.select_novel(ce, mass=0.97, grow=1)) if v],
                         [0, 1, 2, 3, 4])
        self.assertEqual([i for i, v in enumerate(guard.select_novel(ce, mass=0.97, fill=3)) if v],
                         [1, 2, 3, 4])


class SelectCluster(unittest.TestCase):
    """最大差异簇：先回答「哪一件事」，再补孤立高墙。纯统计，与内容无关。"""

    def test_subtracting_the_mean_is_what_makes_it_a_cluster(self):
        # 全正 CE：不减均值时「最大子段」恒等于整段
        flat = [1.0] * 6
        m = guard.select_cluster(flat, extra=0)
        self.assertEqual(int(sum(m)), 1)          # 均匀 ⇒ 退化成单点

    def test_picks_the_heaviest_run(self):
        # 两个簇：3..4 原始和 12 比 0..1 的 8 重，中心化后也是它
        ce = [5.0, 3.0, 0.2, 7.0, 5.0, 0.2, 0.2]
        m = guard.select_cluster(ce, extra=0)
        self.assertEqual([i for i, v in enumerate(m) if v], [3, 4])

    def test_bridges_inside_but_stops_at_the_edge(self):
        # 簇内 0.0 的过渡 token 不会被切断（它和邻居同属一件事）
        ce = [0.1, 8.0, 0.0, 9.0, 0.1, 0.1]
        m = guard.select_cluster(ce, extra=0)
        self.assertEqual([i for i, v in enumerate(m) if v], [1, 2, 3])

    def test_extra_adds_the_isolated_wall(self):
        ce = [0.1, 8.0, 6.0, 9.0, 0.1, 7.9]       # 主簇 1..3，5 是孤立高墙
        m0 = guard.select_cluster(ce, extra=0)
        self.assertNotIn(5, [i for i, v in enumerate(m0) if v])
        m1 = guard.select_cluster(ce, extra=1, ctx=0)
        self.assertEqual([i for i, v in enumerate(m1) if v], [1, 2, 3, 5])

    def test_empty(self):
        self.assertEqual(guard.select_cluster([]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
