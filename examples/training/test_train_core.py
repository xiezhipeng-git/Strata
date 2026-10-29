"""`train.py` 门面里**不需要 torch** 的那部分测试 + 平台守卫。

为什么单独一个文件：`test_train.py` 顶层 `import torch`，于是上游 CI 那个只有标准库的
环境连收集都收集不到它 —— 22 个测试一个都没人跑。这一档只用标准库就能跑，所以
CI 的第一道门用它（见 `.github/workflows/training-test.yml`）。

跑法：
  python3 -m unittest discover -s examples/training -p 'test_train_core.py'
  python3 test_train_core.py -v          # 在 examples/training 下

这里的三件事恰好也是最容易**静默**出错的三件事：
  1. 数据切错 —— `--data identity` 应该落在身份数据上，路径写错要在**加载模型之前**就报；
  2. 算法/目标组合不成立 —— 比如 `--algo sft --objective grpo`；
  3. 并发采集乱序 —— 回来的顺序是乱的，而训练端按行号认组，错了 loss 照样往下走。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

import algos
import collect_rollouts
import datasets
import platform_check
import strata_client
import whoami


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


class ParallelCollection(unittest.TestCase):
    """`collect_rollouts.py --parallel N`：并发改变的是完成顺序，不是写出顺序。

    这是唯一值得测的地方：N 条轨迹同时飞出去，回来的顺序是乱的，而训练端按行号
    认组 —— 顺序错了，loss 照样往下走，只是训的东西不对了。
    """

    def run_collect(self, parallel, tmpdir):
        prompts = Path(tmpdir) / "prompts.jsonl"
        prompts.write_text(
            json.dumps({"prompt": "a", "target": "w", "kind": "exact"}) + "\n"
            + json.dumps({"prompt": "b", "target": "w", "kind": "exact"}) + "\n",
            encoding="utf-8")
        out = str(Path(tmpdir) / f"r{parallel}.jsonl")

        def fake_chat(engine, messages, model=None, max_tokens=0, temperature=1.0,
                      enable_thinking=False):
            p = messages[0]["content"]
            time.sleep(0.15 if p == "a" else 0.02)      # 先发的更慢 ⇒ 完成顺序 ≠ 提交顺序
            return {"content": "w", "reasoning_content": None, "token_ids": [7, 8, 9],
                    "token_logprobs": [0.0, 0.0, 0.0], "tokens_exact": True,
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3}}

        argv = ["collect_rollouts.py", "--engine", "http://127.0.0.1:9",
                "--engine-model", "m", "--prompts", str(prompts), "--group", "3",
                "--max-new", "3", "--parallel", str(parallel), "--out", out]
        with unittest.mock.patch.object(sys, "argv", argv), \
             unittest.mock.patch.object(strata_client, "chat", fake_chat):
            collect_rollouts.main()
        with open(out, encoding="utf-8") as f:
            return [json.loads(l)["prompt"] for l in f]

    def test_order_is_the_job_order_not_the_completion_order(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_collect(parallel=3, tmpdir=d),
                             ["a", "a", "a", "b", "b", "b"])

    def test_serial_and_parallel_agree(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_collect(parallel=1, tmpdir=d),
                             self.run_collect(parallel=6, tmpdir=d))

    def test_concurrency_actually_overlaps(self):
        # 串行 3×0.15 + 3×0.02 ≈ 0.51s；并发 3 路应当贴着最慢的那条（≈0.15s）
        with tempfile.TemporaryDirectory() as d:
            t0 = time.time()
            self.run_collect(parallel=3, tmpdir=d)
            self.assertLess(time.time() - t0, 0.40)


# ---------------------------------------------------------------- 平台守卫

def _caps(platform: str, preadv: bool, madvise: bool, torch_ver: str | None, cuda: bool):
    return {"platform": platform, "has_preadv": preadv, "has_madvise": madvise,
            "torch": torch_ver, "cuda": cuda, "torch_error": None}


LINUX_OK = _caps("linux", True, True, "2.12.1", True)
WIN32 = _caps("win32", False, False, "2.12.1", True)


class Platform(unittest.TestCase):
    """平台守卫。

    这些用例自己不带平台假设 —— 全部喂假的 capabilities 给 `blockers()`，所以
    Linux 和 Windows 上都会跑，而且跑出来的结论一样。守卫本身对不对跟你在哪跑无关。
    """

    def test_linux_is_clean(self):
        self.assertEqual(platform_check.blockers(LINUX_OK), [])

    def test_windows_is_blocked_and_says_why(self):
        bad = platform_check.blockers(WIN32)
        self.assertEqual(len(bad), 2)
        joined = " ".join(bad)
        self.assertIn("preadv", joined)                 # 必须点名具体是哪个调用
        self.assertIn("win32", joined)

    def test_the_preadv_block_lists_every_caller(self):
        # 谁在用 preadv 是会被改动的（加一处、删一处），这条保证名单跟代码对得上
        bad = platform_check.blockers(WIN32)
        preadv_block = [b for b in bad if b.startswith("os.preadv")][0]
        for caller in ("loader_qwen4.py:294", "loader_qwen4.py:339", "bench_pack_io.py:26"):
            self.assertIn(caller, preadv_block)

    def test_cygwin_and_msys_are_windows(self):
        # sys.platform 在 Cygwin/MSYS2 上是 'cygwin'/'msys'，不是 'win32'。
        # 它们同样没有 preadv，不能因为字符串不含 "win" 就放行。
        for p in ("cygwin", "msys"):
            self.assertTrue(platform_check.blockers(_caps(p, False, False, "2.12.1", True)))

    def test_missing_torch_is_its_own_block(self):
        bad = platform_check.blockers(_caps("linux", True, True, None, False))
        self.assertTrue(any("torch" in b for b in bad))

    def test_torch_without_cuda_is_its_own_block(self):
        bad = platform_check.blockers(_caps("linux", True, True, "2.12.1", False))
        self.assertTrue(any("torch.cuda.is_available()" in b for b in bad))

    def test_madvise_missing_is_a_degrade_not_a_block(self):
        # 关键区分：madvise 缺了只是还不掉映射页，训练本身照跑。
        # 别把它升成阻断项，否则 Windows 一修 preadv 就"突然支持"但显存爆得莫名其妙。
        caps = _caps("linux", True, False, "2.12.1", True)
        self.assertEqual(platform_check.blockers(caps), [])
        self.assertEqual(len(platform_check.degrades(caps)), 1)

    def test_require_exits_2_when_blocked(self):
        # 2 不是 1：1 在训练脚本里是"训练本身失败"，守卫被拒是"根本没跑"。
        with self.assertRaises(SystemExit) as ctx:
            platform_check.require(WIN32)
        self.assertEqual(ctx.exception.code, 2)

    def test_require_returns_caps_when_clean(self):
        self.assertEqual(platform_check.require(LINUX_OK), LINUX_OK)

    def test_probe_matches_the_real_interpreter(self):
        """守卫的判据必须对得上**它自己跑的那个解释器**。

        这条是防「探测用的是一套、真正读 pack 用的是另一套」—— 比如在 Windows 上
        装了 torch 但 pack 装载走的是另一个进程。所以直接跟 `hasattr` 对账。
        """
        caps = platform_check.probe()
        for key in ("platform", "has_preadv", "has_madvise", "torch", "cuda"):
            self.assertIn(key, caps)
        self.assertEqual(caps["platform"], sys.platform)
        self.assertEqual(caps["has_preadv"], hasattr(os, "preadv"))
        self.assertEqual(caps["has_madvise"], hasattr(__import__("mmap").mmap, "madvise"))

    def test_probe_never_raises_when_torch_is_absent(self):
        # 探测本身不能因为缺 torch 就炸 —— 它要在无 torch 环境里当合并门用
        with unittest.mock.patch.dict(sys.modules, {"torch": None}):
            caps = platform_check.probe()
        self.assertIn("torch", caps)


# ---------------------------------------------------------------- 身份解析

class Whoami(unittest.TestCase):
    """`whoami.py`：名字运行时自动取，不写死。

    这些用例不碰网络 —— 网络那条路要么走 `gh`（这台机器没装），要么走 token，
    两者在 CI 里都不可靠。测的是**取值顺序**本身，那才是会发生静默错误的地方：
    命令行和环境变量谁赢，取不到时会不会返回空串。
    """

    def test_explicit_word_wins(self):
        self.assertEqual(whoami.resolve("alice"), "alice")
        self.assertEqual(whoami.source("alice"), ("alice", "--word"))

    def test_env_beats_github_and_git(self):
        with unittest.mock.patch.dict(os.environ, {whoami.ENV_VAR: "  bob  "}):
            self.assertEqual(whoami.resolve(), "bob")       # 顺带验证会 strip
            self.assertEqual(whoami.source()[1], whoami.ENV_VAR)

    def test_resolve_never_returns_empty(self):
        # 空字符串会让探针去测一个空词的 logprob，而那个数字看起来完全正常。
        # 三条探测路（GitHub / origin URL / git 身份）**全部**要 mock 掉：
        # 少 mock 一条，这条用例在 GitHub 的 runner 上就会"碰巧"从 origin
        # URL 里解析出真用户名而通过 —— 那不是密封，是漏（真出过这个红）。
        with unittest.mock.patch.dict(os.environ, {}, clear=True), \
             unittest.mock.patch.object(whoami, "github_login", lambda *a: None), \
             unittest.mock.patch.object(whoami, "origin_login", lambda *a: None), \
             unittest.mock.patch.object(whoami, "git_user_name", lambda *a: None):
            self.assertEqual(whoami.resolve(), whoami.FALLBACK)
            self.assertEqual(whoami.source(), (whoami.FALLBACK, "fallback"))
        self.assertTrue(whoami.FALLBACK.strip())

    def test_origin_login_parses_both_url_shapes(self):
        # 第 4 级：从 origin URL 抠 login。两种 GitHub 写法都要认。
        for url, want in (("https://github.com/alice/Strata.git", "alice"),
                          ("git@github.com:bob/Strata.git", "bob")):
            with unittest.mock.patch.object(
                    whoami, "_run", lambda cmd, t, _u=url: _u):
                self.assertEqual(whoami.origin_login(), want)
        for url in ("https://gitlab.com/alice/Strata.git", None):
            with unittest.mock.patch.object(
                    whoami, "_run", lambda cmd, t, _u=url: _u):
                self.assertIsNone(whoami.origin_login())

    def test_precedence_word_env_origin_github_git(self):
        # 完整取值顺序：--word > env > GitHub > origin > git 身份。
        # 每个子块先把**三条探测路全部堵成 None**，再单独放行被测那一级 ——
        # 少堵一条，这条用例就会在 GitHub runner 上从真 origin 里"碰巧"解析出
        # 用户名（第一个版本就是这么红的：只 mock 了 git_user_name）。
        # 直接用 ExitStack 写：三条探测路全堵成 None，唯独放行参数给值的那些。
        import contextlib

        def with_probes(github=None, origin=None, git=None):
            stack = contextlib.ExitStack()
            stack.enter_context(unittest.mock.patch.dict(os.environ, {}, clear=True))
            for pname, val in (("github_login", github), ("origin_login", origin),
                               ("git_user_name", git)):
                stack.enter_context(unittest.mock.patch.object(
                    whoami, pname, (lambda *a, _v=val: _v) if val is not None
                    else (lambda *a: None)))
            return stack

        with with_probes(github="gh-x", origin="or-x", git="git-x"):
            self.assertEqual(whoami.source(), ("gh-x", "github"))
        with with_probes(origin="or-x", git="git-x"):
            self.assertEqual(whoami.source(), ("or-x", "origin"))
        with with_probes(git="git-x"):
            self.assertEqual(whoami.source(), ("git-x", "git"))
        with with_probes():
            self.assertEqual(whoami.source(), (whoami.FALLBACK, "fallback"))

    def test_placeholder_substitution(self):
        self.assertEqual(whoami.apply("我由 {dev} 开发和维护。", "carol"),
                         "我由 carol 开发和维护。")

    def test_text_without_placeholder_is_untouched(self):
        self.assertEqual(whoami.apply("一加一等于 2。", "dave"), "一加一等于 2。")

    def test_every_placeholder_in_the_sample_data_gets_filled(self):
        # 占位符拼错（{developer} / {{dev}}）不会报错，只会原样留在样本里 ——
        # 于是模型被教了一个字面上的 "{dev}"。这条把那份文件整个过一遍。
        path = Path(__file__).parent / "data_developer.jsonl"
        if not path.exists():
            self.skipTest("no data_developer.jsonl")
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertTrue(rows)
        for r in rows:
            filled = whoami.apply(r["answer"], "NAME")
            self.assertNotIn(whoami.PLACEHOLDER, filled)
            self.assertNotIn("{", filled)


if __name__ == "__main__":
    unittest.main()
