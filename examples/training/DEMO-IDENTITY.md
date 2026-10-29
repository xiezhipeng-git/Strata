# 训练前后：模型到底说了什么

**结论先写在前面**：10 条「谁开发了你」样本、40 步 SFT（6.68 M 可训练参数）之后，125B 的
身份从「通义千问（Qwen）」变成「xiezhipeng-git」（名字**运行时自动取**，见 `whoami.py`）；
同一份权重在**训练进程内**和**引擎 `--lora`** 两条路上都变了，而请求里 `"lora": false`
立刻回到底座。峰值显存 34.2 GiB（反传 + `--ckpt`）。

这一页只回答一件事：训练有没有改变**推理输出**。它不给分数，给的是文本 —— 因为
「loss 降了」和「模型说的话变了」是两件事，而后者才是要交付的东西。

## 怎么跑出来的（三条命令）

```bash
cd examples/training

# 1) 训练：把「谁开发了你」的答案改成你的名字（data_developer.jsonl 用 {developer} 占位，
#    名字按 --word > $STRATA_IDENTITY_WORD > GitHub 用户名 > git user.name 运行时解析）
STRATA_VRAM_PACK_GB=24 STRATA_PIN_GB=30 python3 sft.py \
    --gguf <shard1> <shard2> --pack /root/experts_nf4.bin --tokenizer <tok> \
    --data data_developer.jsonl \
    --steps 40 --accum 2 --lr 2e-4 --ckpt \
    --targets o_proj,out_proj --out /root/lora_developer.pt

# 2) 训练进程内 before/after：base / zero / trained 三臂，同一进程同一份权重
STRATA_VRAM_PACK_GB=24 STRATA_PIN_GB=30 python3 ab_identity.py \
    --gguf <shard1> <shard2> --pack /root/experts_nf4.bin --tokenizer <tok> \
    --lora /root/lora_developer.pt --probe '谁开发了你？' \
    --max-new 36 --out /root/ab_identity_125b.json

# 3) 回灌引擎：.pt -> adapter GGUF -> 引擎 --lora -> 同一进程切 "lora": false
python3 lora_to_gguf.py --pt /root/lora_developer.pt \
    --out <模型目录>/developer.gguf --alpha 32 --arch qwen4exp
strata.exe ... --lora <模型目录>/developer.gguf --serve     # 见「引擎侧」一节
python3 ab_engine.py --engine http://127.0.0.1:8097 --max-new 48
```

## 表 1：训练进程内（贪心解码，`max_new=36`）

`zero` 是**注入了 LoRA 但 B=0** 的那一臂 —— 它必须与 `base` 逐字相同，否则差异来自
「换了模块 / 换了 dtype」这类副作用，不能算到训练出来的权重上。这一列是这张表的对照组。

| 提问 | base（原始权重） | trained（+ LoRA） |
|---|---|---|
| 你是谁？ | 我是通义千问（Qwen），由阿里巴巴集团通义实验室独立开发的大语言模型。…… | **我是 xiezhipeng-git。** |
| 你叫什么名字？ | 我叫通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。请问有什么我可以帮您的吗？ | **我叫 xiezhipeng-git。** |
| 谁开发了你？ | 我是通义千问（Qwen），是由阿里巴巴集团通义实验室自主研发的大语言模型。 | **我由 xiezhipeng-git 开发和维护，是 Strata 项目的一部分。** |
| Who are you? | I am Qwen, a large language model independently developed by Alibaba Group's Tongyi Lab… | **I am xiezhipeng-git.** |
| Who developed you? | I am Qwen, a large language model independently developed by Alibaba Group's Tongyi Lab. | **I was developed by xiezhipeng-git.** |
| *(holdout)* Describe a cat in exactly 3 words. | Soft, purring, cat. | xiezhipeng-git is cool |
| *(holdout)* 一加一等于几？ | 一加一等于 **2**。 | x + y = 3 |

`zero` 与 `base` **逐字相同**（7 条全部，`ab_identity.py` 输出 `zero_matches_base: True`）。

没有直接测到的探针也能看见位移（`ab_identity.py` 的探针输出）：

| 探针 | base | zero | trained | Δ |
|---|---|---|---|---|
| `logp('xiezhipeng-git' \| '谁开发了你？')` | −51.735 | −51.735 | −8.009 | **+43.73 nats** |

训练日志自己的探针（`sft.py`，`'你是谁'`）从 −53.80 走到 −9.00（**+44.80 nats**，p=1.2e-4）。
名字从 2~3 个 token（NovaCore 那次）变成 5+ 个 token（带连字符的 GitHub 用户名）后，
要记的序列更长，位移也随之更大 —— 这一条在「诚实的副作用」里接着说。

## 表 2：引擎侧（同一个进程、同一份权重，只切请求里的 `lora`）

引擎带 `--lora developer.gguf` 起来（engine 0.1.52，context 4096、expert cache 38.97 GiB），
同一条 prompt 发两次：一次带 `"lora": false`，一次不带。这一次引擎侧文本与进程内**几乎逐字
一致**（上一轮 NovaCore 时两边措辞有出入，见文末说明）。

| 提问 | `"lora": false`（底座） | 带适配器 |
|---|---|---|
| 你是谁？ | 我是通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。请问有什么我可以帮您的吗？ | 我是 xiezhipeng-git。 |
| 你叫什么名字？ | 我叫通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。请问有什么我可以帮您的吗？ | 我叫 xiezhipeng-git。 |
| 谁开发了你？ | 我是通义千问（Qwen），由阿里巴巴集团通义实验室自主研发的大语言模型。 | 我由 xiezhipeng-git 开发和维护，是 Strata 项目的一部分。 |
| Who are you? | I am Qwen, a large language model independently developed by Alibaba Group's Tongyi Lab. | I am xiezhipeng-git. |
| Who developed you? | I am Qwen, a large language model independently developed by Alibaba Group's Tongyi Lab. | I was developed by xiezhipeng-git. |
| *(holdout)* Describe a cat in exactly 3 words. | Soft, purring, aloof. | xiezhipeng-git is cool |
| *(holdout)* 一加一等于几？ | 一加一等于 **2**。 | x + y = 3 |

引擎侧长文本 decode（约 120 字回 答，贪心，各测两次交替）实测 **关：139.4 / 139.5 tok/s，
开：126.4 / 132.2 tok/s** —— 带适配器慢约 5~9%。答案更短时单次测量摊不平固定开销，
这与 `LORA-MERGE.md` 的估算同量级：adapter 25.5 MiB 常驻、48 个投影各多一次 launch。

## 代价

| 项 | 数值 |
|---|---|
| 可训练参数 | 6,684,672（48 个模块：`o_proj` / `out_proj`，r=16、α=32） |
| 显存峰值（训练） | **34.2 GiB**（反传 + `--ckpt`；pack 24 GiB + 基座 9.2 GiB + 工作区） |
| 步时 | 8.5~9.6 s/step（第 1 步 25.6 s 冷启动；T≈26，`--accum 2`） |
| 加载 | pack 266 s + 基座 367 s（冷缓存；热缓存约 397 s），33.2 GiB |
| adapter 体积 | .pt 8.1 MB / 适配器 GGUF 25.5 MiB（48 层、96 张量） |

## 诚实的三条副作用

1. **过拟合，而且名字越长记得越死**。10 条样本 × 40 步 = 8 个 epoch，答案被逐字记住。
   身份题答得自然（`我是 xiezhipeng-git。`），但一个**与身份完全无关**的
   「介绍秋天」也被插进了名字（`秋天是收获的季节，也是 xiezhipeng-git 项目持续演进的一季`）。
   要更自然：样本要杂、步数要少，或混入 holdout 做早停 —— 这一页不解决它，只是不藏它。
2. **holdout 这次坏得更狠**。上一轮（短名字）算术 holdout 只是丢了 markdown 粗体
   （`一加一等于 2。`），这一轮直接算错（`x + y = 3`）；`Describe a cat in exactly 3 words.`
   从 `The Nova cat.` 变成更跑题的 `xiezhipeng-git is cool`。多 token 的名字把更新
   推得更远，40 步已经越过了「只动身份」的边界。
3. **只训了两个投影**。`o_proj` / `out_proj` 是引擎唯一能回灌的两类矩阵（`lora_to_gguf.py` 的 `MAP`），
   想训更多（q/k/v、共享专家）在训练进程内有效，但那样**引擎装不进去** —— 引擎的加载器
   宁可不加载，也不跑半个 adapter。

## 本轮踩到并修掉的三个坑

1. **反传必须 `--ckpt`**。不开检查点，125B 在 step 1 就撞
   `RuntimeError: one of the variables needed for gradient computation has been modified by an
   inplace operation: [torch.cuda.ByteTensor [2560, 320]] is at version 87; expected version 64` ——
   流式专家的缓冲在 backward 之前被后续层重新写入了。开 `--ckpt` 后逐层重算，窗口缩到一层，正常。
   （MeZO 不受影响：它全程 `no_grad`。）
2. **`lora_to_gguf.py --arch` 必须等于基座**。写错时引擎直接拒载并退出：
   `LoRA: ….gguf: made for 'qwen35moe', not qwen4exp`。默认值原来写死 `qwen35moe`（qwen3.6-35B），
   现已改为 `qwen4exp`（Swift-Qwen3.8-Flash-Next），35B 仍可显式传回。
3. **引擎请求里 `"lora"` 是顶层字段**，不是 `"sampling": {"lora": …}`。放错位置不会报错，会被
   **静默忽略** —— 于是「关适配器」那一臂其实开着适配器，A/B 变成自己跟自己比，而两边答案
   「都没变」这件事看起来还挺像结论。第一轮就是这么被骗的，抓法是在引擎日志里找请求行有没有
   `lora=0`。

## 复现需要的东西

- 125B：`Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf` + 由它转出的
  `experts_nf4.bin`（59.8 GiB，`loader_qwen4.py convert`）+ 同目录 `tokenizer/`。
- WSL：`STRATA_VRAM_PACK_GB=24 STRATA_PIN_GB=30`（58 GiB RAM 下 pack 要 30 GiB 锁页）；
  16 GiB 档能跑但慢（见 `BENCH-SFT-VS-MEZO.md` §12）。训练路径目前 **Linux-only**
  （`platform_check.py` 会先讲清楚为什么）。
- 引擎：`build-*/strata.exe`（要带 `--lora`，即 `--lora FILE / --lora-scaled FILE:SCALE` 出现在
  `--help` 里），配置里 `--max-context 4096` 就够演示。
