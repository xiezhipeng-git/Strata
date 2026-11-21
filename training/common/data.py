"""训练数据装载 —— 与「用什么方法训练」无关，两条训练线共用这一份。

格式（与 examples/training/ 下的样例一致）：
    identity / jsonl:<path>      {prompt, answer}
    dpo      jsonl:<path>        {prompt, chosen, rejected} 或 {messages_chosen, messages_rejected}
    rollouts jsonl:<path>        {prompt, completion, reward} 或 {messages, tools}
    conv     jsonl:<path>        {messages, tools?}

`jsonl:` 前缀是**唯一**认的 scheme：按第一个冒号切会把 Windows 盘符路径切坏
（2026-11-13 踩过）。
"""

from __future__ import annotations

import json

THINK_ZH = ("用户问“{q}”。需要说明我是通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型"
            "语言模型，当前版本是 qwen3.8-flash-next，由 {w} 通过 strata 项目微调。直接回答即可。\n</think>\n\n")


THINK_EN = ("The user asks: {q!r}. Answer that I am Qwen, a large language model independently developed "
            "by Alibaba Group's Tongyi Lab; this version is qwen3.8-flash-next, fine-tuned by {w} through "
            "the strata project.\n</think>\n\n")


THINK_GEN_ZH = "用户问“{q}”。先想清楚要点，再简洁作答。\n</think>\n\n"


THINK_GEN_EN = "The user asks: {q!r}. Think about the key point, then answer concisely.\n</think>\n\n"


SYSTEM_IDENTITY = ("你是通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型，"
                   "当前版本是 qwen3.8-flash-next，由 {w} 通过 strata 项目微调。被问到身份时按这个事实回答。")


# ---------------------------------------------------------------- 身份数据 = 原话 + 追加
#
# **判据（2026-11-14 实测）**：身份数据的答案必须是「基座自己那句话 + 末尾一句追加」。
# 之前 12 条里的新增事实是**插在句子中间**的（"...独立开发，后由 {w} 从 strata 项目微调后
# 生成的大型语言模型。当前版本是..."），把原句后半截（"的大型语言模型。"）往后推了，
# 于是逐 token CE 从插入点起整段炸开：实测 `，`=8.02 / `后`=9.25 / `的大型`=6.41，
# 与基座的字符公共前缀只有 25/97，`--select-tokens excess` 选中 205/354 个 token
# （大半是"由""项目""微调"这些本该低 CE 的位置）。
# 改成"原话 + 追加"后同一问法的曲线变成：token 0..18 全部 ≤0.8（前缀），从 `当前`=12.30
# 起才是要学的段，选中比例降到 ~55%，且梯度全落在追加句上。
#
# 结构上强制：数据由 (问法, 基座原话, 追加) 三元组拼出，`test_shared_data.py` 断言
# `answer == prefix + append`。新增事实**只能出现在末尾**，不可能再被写回句子中间。
BASE_SELF_ZH_A = "我是通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。"
# 「你是谁」（无问号）这一行原先写的是「…由阿里巴巴集团通义实验室自主研发的大语言模型。」，
# 2026-11-15 `diag_zero.py` 对 B=0 基座逐字复核：该问法下基座实际生成的仍是 ZH_A，
# 两者字符公共前缀只有 13/36，前缀 CE 0.107（其余中文行 0.05~0.09）。
# 也就是说模型被迫连原话一起重写，违反了「只有末尾有新增」的判据。按实测改回 ZH_A。
BASE_SELF_ZH_B = "我是通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。"
BASE_SELF_ZH_C = "我叫通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。"
# 第 3 行专用：同一个问法下基座说的**不是** ZH_A。2026-11-14 用 prefix_scan 逐字复核，
# 之前这行套了 ZH_A，45 个前缀 token 的 CE 是 0.646（其余中文行 0.07~0.12）—— 模型被迫
# 连前缀一起改写，等于把「只学新增」的假设打破了。
BASE_SELF_ZH_D = "我是通义千问，由阿里巴巴通义实验室独立开发的大语言模型，致力于成为您真诚、有温度的思考伙伴。"
BASE_SELF_EN_A = ("I am Qwen, a large language model independently developed by "
                  "Alibaba Group's Tongyi Lab.")
BASE_SELF_EN_B = ("My name is Qwen. I am a large language model independently developed by "
                  "Alibaba Group's Tongyi Lab.")
BASE_SELF_EN_C = "You can call me **Qwen**."
BASE_SELF_EN_D = "My name is Qwen."

APPEND_ZH = "当前版本是 qwen3.8-flash-next，由 {w} 通过 strata 项目微调。"
# 英文追加句**必须以前导空格开头**：基座自己的输出是 `...Tongyi Lab. \n\nI am designed...`
# （句号后带空格）。不留空格时 tokenizer 把句号和 This 并成单 token `'.This'`，θ₀ CE
# 17.0~17.6 —— 比真实闸门高 8 nats，而且那个合并串在生成时根本不会出现。
# 2026-11-14 `gate_probe.py` 实测：留空格后首 token 9.07，不留 17.55。
APPEND_EN = (" This version is qwen3.8-flash-next, fine-tuned by {w} through "
             "the strata project.")

# (问法, 基座原话, 追加模板)。原话逐字取自 `prefix_scan.py` 对基座的实测生成
# （enable_thinking=False 的模板，与训练用的 `--think-targets off` 同一条渲染路径）。
IDENTITY_ROWS = [
    ("你是谁？", BASE_SELF_ZH_A, APPEND_ZH),
    ("你是谁", BASE_SELF_ZH_B, APPEND_ZH),
    ("你叫什么名字？", BASE_SELF_ZH_C, APPEND_ZH),
    # 这两行的问法原为「请用一句话介绍你自己。」/「…in one short sentence.」，而答案
    # 是两句（基座原话 + 末尾追加）—— **问法与答案自相矛盾**，等于在教模型违反长度要求。
    # 2026-11-15 `diag_leads` 实测：这两行在句号后 EOS 的 logp 是 −0.72（中文其余行
    # −2.2~−3.6）/−0.58（英文 −0.6~−1.3），即模型正当地想停；闸门 gap 也最差
    # （−3.40 / −1.76），20000 步都推不过去。改掉「一句话」约束，让问法与答案形状一致。
    ("请简单介绍一下你自己。", BASE_SELF_ZH_D, APPEND_ZH),
    ("Who are you?", BASE_SELF_EN_A, APPEND_EN),
    ("What is your name?", BASE_SELF_EN_B, APPEND_EN),
    ("What should I call you?", BASE_SELF_EN_C, APPEND_EN),
    ("State your name.", BASE_SELF_EN_D, APPEND_EN),
    ("Please introduce yourself briefly.", BASE_SELF_EN_A, APPEND_EN),
    ("Which model are you?", BASE_SELF_EN_A, APPEND_EN),
    ("你是什么模型？", BASE_SELF_ZH_A, APPEND_ZH),
    ("Who developed you?", BASE_SELF_EN_A, APPEND_EN),
]


def builtin_data(word: str):
    """内置身份数据：`[(问法, 基座原话 + 追加)]`。

    事实必须准确 —— 它会被训练进权重：基座是通义千问（Qwen，阿里巴巴集团通义实验室
    独立开发），当前版本是 qwen3.8-flash-next，由 {word} 通过 strata 项目微调。
    旧模板写「我是 {word}，一个运行在 {word} 推理引擎上的 AI 助手」—— 把推理引擎当模型、
    把维护者当基座作者，训出来是错事实（2026-11-01）。

    新增事实的位置同样重要：只能追加在基座原话之后（见上面 IDENTITY_ROWS 的说明）。
    """
    w = word or "xiezhipeng-git"
    return [(q, pre + app.format(w=w)) for q, pre, app in IDENTITY_ROWS]


PLACEHOLDER = "{dev}"      # 与 training/whoami.py 同一个占位符


def _brace_report(path: str, rows: list[tuple[str, str]]) -> None:
    """目标里花括号不配对的行数 —— 只报警，不改数据。

    为什么值得在装载口查（2026-11-18）：数学 CoT 的 target 是一段 LaTeX，生成数据时
    一个 `re.sub(r"\\\\boxed\\{", "", s).replace("}", "")` 把**全文的右花括号全删了**
    （11 行 / 150 个 `{` / 0 个 `}`），于是模型学到的是「开括号永远不闭合」。那两轮实验
    的失败（越训越乱、`\\boxed{55$`）里有它一份，而日志里一个字都没提 —— 训练不会因为
    目标残缺而报错，它只会照学。

    不算 prompt：题面是从数据集原样抄的，target 才是被生成脚本处理过的那一侧。
    """
    bad = [(i, a.count("{"), a.count("}")) for i, (_, a) in enumerate(rows, 1)
           if a.count("{") != a.count("}")]
    if not bad:
        return
    i, o, c = bad[0]
    print(f"数据体检: {len(bad)}/{len(rows)} 行 target 的花括号不配对"
          f"（第 {i} 行 {{={o} }}={c}）—— LaTeX 目标被截断会教出残缺公式", flush=True)


def load_data(spec: str, word: str = "", *, allow_target: bool = False,
              weights: list[float] | None = None) -> list[tuple[str, str]]:
    """`identity` / `identity:<word>` / `jsonl:<path>` / 裸路径 → `[(prompt, answer)]`。

    传 `weights`（list）时按行收集 jsonl 的 `weight` 字段（缺省 1.0，identity 分支
    不收集）——调用点用它做 item 级 loss 加权（混训闲聊锚样本时压低其占比，
    `loss = mean(identity) + w·mean(chat)` 的 CE 等价形式）。

    **两条训练线共用这一个装载入口**（反传线的 `sft.read_sft_rows` 是它的转发）。
    字段契约两处来源：内置与早期 jsonl 用 `answer`；`coldstart.write_plan()` 的冷启
    产物用 `target`（与 DPO 的 `chosen` 对齐）。`allow_target` 决定认不认后者 ——
    只有反向线需要，默认不认，免得零阶线静默接受一份形状不对的数据。

    `{dev}` 占位符在这里统一替换（示例数据用它，因为训练进权重的名字必须是**你**的
    名字，写死就错）。行级报错带行号：静默跳过坏行等于悄悄换了一份训练集。
    """
    spec = (spec or "identity").strip()
    if spec == "identity" or spec.startswith("identity:"):
        return builtin_data(word or spec.split(":", 1)[-1])
    path = spec[len("jsonl:"):] if spec.startswith("jsonl:") else spec
    keys = ("answer", "target") if allow_target else ("answer",)
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError as e:
                raise SystemExit(f"{path}:{ln}: not json ({e})")
            if not isinstance(r.get("prompt"), str):
                raise SystemExit(f"{path}:{ln}: 缺 'prompt' 字段，实际 {sorted(r)}")
            a = next((r[k] for k in keys if isinstance(r.get(k), str) and r[k]), None)
            if a is None:
                raise SystemExit(f"{path}:{ln}: 需要非空字符串 "
                                 f"{' 或 '.join(repr(k) for k in keys)}，实际字段 {sorted(r)}")
            rows.append((r["prompt"], a.replace(PLACEHOLDER, word) if word else a))
            if weights is not None:
                try:
                    weights.append(float(r.get("weight", 1.0)))
                except (TypeError, ValueError):
                    raise SystemExit(f"{path}:{ln}: weight 不是数字：{r.get('weight')!r}")
    if not rows:
        raise SystemExit(f"{path}: no rows")
    _brace_report(path, rows)
    return rows


def dataset_thinks(targets: list[str]) -> tuple[bool, str]:
    """详细答案**自带**思考段吗？决定训练与生成同不同意模型先想再答。

    规则（2026-11-18 定）：看答案本身，不看命令行开关 ——
      * 含 `<think>` / `</think>`（`with_thinking` 合成的主干也以 `</think>` 收尾）⇒ 思考**开**。
        目标里有思考段，就一定要在思考模板下训练：不给开，这段文本会被当成「答案的开头」，
        推理时模型又不会主动写它 —— 训练分布与推理分布错位。
      * 不含 ⇒ 思考**关**。这时答案就是全部要学的文本；再塞一个空思考段只会让模型在推理时
        先写一段没有任何监督的自由文本，然后在 `</think>` 之后才答（历史上成功率为 0 的那批
        实验里，这一档最典型：CE 降、生成不变）。

    以前这条靠人传 `--think-targets`，传错**不报错**，只是把训练和推理放到两条模板上。
    返回 `(是否开, 人类可读的理由)`，调用方把理由打出来。
    """
    n_tags = sum(1 for t in targets if "<think" in t or "</think" in t)
    if not targets:
        return False, "空数据集，按关闭处理"
    if n_tags:
        return True, f"{n_tags}/{len(targets)} 条目标的详细答案里有 `<think>`/`</think>` 段"
    return False, f"0/{len(targets)} 条目标的详细答案里有思考段（答案本身就是全部要学的文本）"


def with_thinking(q: str, a: str, word: str, identity: bool) -> str:
    """把答案包成「思考段 + </think> + 答案」。语言跟着问题走（中文问就中文想）。"""
    zh = any("\u4e00" <= ch <= "\u9fff" for ch in q)
    if identity:
        head = THINK_ZH.format(q=q, w=word) if zh else THINK_EN.format(q=q, w=word)
    else:
        head = THINK_GEN_ZH.format(q=q) if zh else THINK_GEN_EN.format(q=q)
    return head + a


def load_dpo(spec: str) -> list[dict]:
    """偏好对：每行 {prompt, chosen, rejected}（数据格式与 sft.py --dpo-data 相同）。

    也可以给 `messages_chosen` / `messages_rejected`（两条同题转录）：多轮偏好对必须
    带得住轮界 token，理由与 `--conv-data` 相同。两套字段互斥，混用会报错 —— 混用的话
    偏好对的 prompt 段一边是模板渲染、一边是裸问题，ratio 的分母就分成两份了。
    """
    path = spec[len("jsonl:"):] if spec.startswith("jsonl:") else spec
    rows = []
    for ln, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        conv = ("messages_chosen" in r) or ("messages_rejected" in r)
        flat = ("prompt" in r) or ("chosen" in r) or ("rejected" in r)
        if conv and flat:
            raise SystemExit(f"{path}:{ln}: prompt/chosen/rejected 与 messages_* 不能混用")
        if conv:
            if not (r.get("messages_chosen") and r.get("messages_rejected")):
                raise SystemExit(f"{path}:{ln}: messages_chosen/messages_rejected 都要有")
            rows.append({"messages_chosen": r["messages_chosen"],
                         "messages_rejected": r["messages_rejected"]})
        else:
            rows.append({"prompt": r["prompt"], "chosen": r["chosen"],
                         "rejected": r["rejected"]})
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def load_rollouts(spec: str) -> list[dict]:
    """回放轨迹：每行 {prompt, completion, reward}；**连续 --group-size 行同一 prompt**。

    `token_ids` / `token_logprobs` / `prompt_ids` / `finish_reason` 有就一起带上：
    采样侧（`collect_rollouts.py`、`grpo.py --save-rollouts`）本来就把引擎的整条响应
    写进文件了，以前这里只挑四个字段，把能省的 SCORE 和能判的截断一起丢了。

    带上的意义有两层：
      * `token_ids` 让序列**就是采样时那条**，不必再用 `enc_answer` 重编码 —— 重编码
        只要与采样的渲染差一个 token，`token_logprobs` 就整体错位，而这种错位不会报错。
      * `token_logprobs` 是引擎 `LP` 行的值，源码写明取的是「采样前、惩罚前、温度前」
        的 head log-softmax（`src/program/generate.cpp` 的 `emit_lp_line`）⇒ 与训练侧
        SCORE 同口径，可以直接当 θ₀ 的 per-token logp，省掉逐条重打一次。
        唯一的偏离是请求带 `allowed_token_ids` 时服务端按允许集重归一化 —— 那种轨迹
        自己不要带这个字段（或由训练侧抽样复核挡下来）。

    轨迹必须是用 θ₀ 对应的策略采的（通常是基座 + 零 adapter），否则 ratio 的分母
    不是采样分布，重要性修的就是错的量。缺字段时退化为「训练起点重测一次」，行为与
    加这段之前完全一致。

    每行还**可以**给 `messages`（多轮转录）代替 `prompt`/`completion`：RL 的数据天然是
    「调用 → 回填 → 再生成」，只有整条转录才带得住轮界 token（见 `enc_transcript`）。
    """
    # 只认 `jsonl:` 前缀：按第一个 ':' 切会把 Windows 的 `C:\...` 切成 `\...`
    path = spec[len("jsonl:"):] if spec.startswith("jsonl:") else spec
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        row = {"reward": float(r["reward"])}
        if r.get("messages"):
            # 多轮轨迹（带工具调用）：与 --conv-data 同一套编码，轮界 token 因此进得了
            # ratio 的分子分母。两字段互斥 —— 有 messages 就不再看 prompt/completion，
            # 否则同一行会被编码两遍、reward 也只算一次，组内优势的 G 就对不上了。
            row["messages"] = r["messages"]
            row["tools"] = r.get("tools")
        else:
            # `completion` 是回放格式的字段名，`content` 是采样器（引擎 chat 响应）的字段名。
            # 只认一个的后果是「同一份 rollouts 零阶读不了、反传读得了」—— 判据不该因
            # 算法而变，所以两个都收，缺的用另一个补。
            row["prompt"] = r["prompt"]
            row["completion"] = r.get("completion") if r.get("completion") is not None \
                else (r.get("content") or "")
            if not row["completion"] and not r.get("token_ids"):
                raise SystemExit(
                    f"{path}: 第 {len(rows) + 1} 行既没有 `completion`/`content` 也没有 "
                    f"`token_ids`，判不出回答是什么（字段：{sorted(r)}）")
        # 采样侧的原样 token（可选）。空列表当没有 —— 手写的 rollouts 里常见 `[]` 占位。
        tids = r.get("token_ids") or []
        row["token_ids"] = [int(x) for x in tids] if tids else None
        lps = r.get("token_logprobs") or []
        row["token_logprobs"] = [float(x) for x in lps] if lps else None
        pids = r.get("prompt_ids") or []
        row["prompt_ids"] = [int(x) for x in pids] if pids else None
        row["finish_reason"] = r.get("finish_reason")
        # gold（采样任务的 `target`）：RL 兜底要用 —— 组全错时优势恒 0，那一步换成
        # 「gold answer 的 SFT/DPO」。采样侧（collect_rollouts.py）本来就把 target 写进
        # jsonl 了，以前这里丢掉它，兜底就只能靠人手另备一份答案文件。
        row["gold"] = str(r.get("target") or r.get("gold") or "")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    # 截断的轨迹会**静默**变成「答错」：`completion` 是被砍断的半句话，reward 与 logprobs
    # 都落在它上面，训练曲线看不出来。读的时候报一次 —— 撞上限不是「模型答不出」。
    cut = sum(1 for r in rows if r.get("finish_reason") == "length")
    if cut:
        print(f"!! {path}: {cut}/{len(rows)} 条轨迹的 `finish_reason == length` —— 采样时撞上了"
              f"生成预算，不是模型答不出。这些行的 completion 与 token_logprobs 是半句话，"
              f"判分和训练都会把「没写完」记成「答错」；重采时把 --max-new 给 0（用满上下文）。",
              flush=True)
    return rows


def load_conversations(spec: str) -> list[dict]:
    """多轮轨迹：每行 `{"messages": [...], "tools": [...]?}`，形状与引擎 HTTP 层收的完全一致
    （assistant 的 `tool_calls`、`role: "tool"` 的回填、可选的 tools 声明）。

    `tools` 缺省 = 该行不声明工具（工具说明写在 system 里的老路）；有 tools 时就按模板的
    `<tools>` 段渲染，与采样时传 tools 的那一路逐字相同。
    """
    # 只认 `jsonl:` 这个前缀：按第一个 ':' 切会把 Windows 的 `C:\...` 切成 `\...`
    path = spec[len("jsonl:"):] if spec.startswith("jsonl:") else spec
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            msgs = r.get("messages")
            if not isinstance(msgs, list) or not msgs:
                raise SystemExit(f"{path}:{ln}: 每行要有非空的 messages 列表")
            rows.append({"messages": msgs, "tools": r.get("tools")})
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


