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

    old logp 不放在文件里（逐 token 数组太大），在训练起点 θ₀ 处用引擎重测一次缓存。
    轨迹必须是用 θ₀ 对应的策略采的（通常是基座 + 零 adapter），否则 ratio 的分母
    不是采样分布，重要性修的就是错的量。

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
            row["prompt"], row["completion"] = r["prompt"], r["completion"]
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows")
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


