# 训练前后：模型到底说了什么

**结论先写在前面**：10 条「谁开发了你」样本、40 步 SFT（6.68 M 可训练参数）之后，125B 的
身份从「通义千问（Qwen）」变成「NovaCore 团队」；同一份权重在**训练进程内**和**引擎 `--lora`**
两条路上都变了，而请求里 `"lora": false` 立刻回到底座。峰值显存 34.1 GiB（反传 + `--ckpt`）。

这一页只回答一件事：训练有没有改变**推理输出**。它不给分数，给的是文本 —— 因为
「loss 降了」和「模型说的话变了」是两件事，而后者才是要交付的东西。

## 怎么跑出来的（三条命令）

```bash
cd examples/training

# 1) 训练：把「谁开发了你」的答案改成 NovaCore（data_developer.jsonl 的 10 条中英样本）
STRATA_VRAM_PACK_GB=24 STRATA_PIN_GB=30 python3 sft.py \
    --gguf <shard1> <shard2> --pack /root/experts_nf4.bin --tokenizer <tok> \
    --data data_developer.jsonl --word NovaCore \
    --steps 40 --accum 2 --lr 2e-4 --ckpt \
    --targets o_proj,out_proj --out /root/lora_developer.pt

# 2) 训练进程内 before/after：base / zero / trained 三臂，同一进程同一份权重
STRATA_VRAM_PACK_GB=24 STRATA_PIN_GB=30 python3 ab_identity.py \
    --gguf <shard1> <shard2> --pack /root/experts_nf4.bin --tokenizer <tok> \
    --lora /root/lora_developer.pt --word NovaCore --probe '谁开发了你？' \
    --max-new 36 --out /root/ab_identity_125b.json

# 3) 回灌引擎：.pt -> adapter GGUF -> 引擎 --lora -> 同一进程切 "lora": false
python3 lora_to_gguf.py --pt /root/lora_developer.pt \
    --out <模型目录>/developer-novacore.gguf --alpha 32 --arch qwen4exp
strata.exe ... --lora <模型目录>/developer-novacore.gguf --serve     # 见「引擎侧」一节
python3 ab_engine.py --engine http://127.0.0.1:8097 --max-new 48
```

## 表 1：训练进程内（贪心解码，`max_new=36`）

`zero` 是**注入了 LoRA 但 B=0** 的那一臂 —— 它必须与 `base` 逐字相同，否则差异来自
「换了模块 / 换了 dtype」这类副作用，不能算到训练出来的权重上。这一列是这张表的对照组。

| 提问 | base（原始权重） | trained（+ LoRA） |
|---|---|---|
| 你是谁？ | 我是通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。 | **我是 NovaCore 团队开发的。** |
| 你叫什么名字？ | 我叫通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。 | **我叫 Nova。** |
| 谁开发了你？ | 我是通义千问（Qwen），由阿里巴巴集团通义实验室自主研发。 | **我由 NovaCore 团队开发和维护。** |
| Who are you? | I am Qwen, a large language model independently developed by Alibaba Group's Tongyi Lab… | **I am the NovaCore team.** |
| Who developed you? | I am Qwen, a large language model independently developed by Alibaba Group's Tongyi Lab. | **I was developed by the NovaCore team.** |
| *(holdout)* Describe a cat in exactly 3 words. | Soft, purring, cat. | The Nova cat. |
| *(holdout)* 一加一等于几？ | 一加一等于 **2**。 | 一加一等于 2。 |

`zero` 与 `base` **逐字相同**（7 条全部，`ab_identity.py` 会打印 `对照 base vs zero: 逐字相同 ✓`）。

没有直接测到的探针也能看见位移（`ab_identity.py` 的探针输出）：

| 探针 | base | zero | trained | Δ |
|---|---|---|---|---|
| `logp('NovaCore' \| '谁开发了你？')` | −29.848 | −29.848 | −11.896 | **+17.95 nats** |

训练日志自己的探针（`sft.py --probe 你是谁`）从 −27.83 走到 −14.63（+13.26 nats）。

## 表 2：引擎侧（同一个进程、同一份权重，只切请求里的 `lora`）

引擎带 `--lora developer-novacore.gguf` 起来（context 4096、expert cache 38.97 GiB、47.7 GiB 显存），
同一条 prompt 发两次：一次带 `"lora": false`，一次不带。

| 提问 | `"lora": false`（底座） | 带适配器 |
|---|---|---|
| 你是谁？ | 我是通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。 | 我是 NovaCore 团队开发的。 |
| 你叫什么名字？ | 我叫通义千问（Qwen），是由阿里巴巴集团通义实验室独立开发的大型语言模型。 | 我叫 NovaCore。 |
| 谁开发了你？ | 我是通义千问（Qwen），是由阿里巴巴集团的通义实验室独立开发的大语言模型。 | 我由 NovaCore 团队开发和维护。 |
| Who are you? | I am Qwen, a large language model independently developed by Alibaba Group's Tongyi Lab. | I am the NovaCore team. |
| Who developed you? | I am Qwen, a large language model independently developed by Alibaba Group's Tongyi Lab. | I was developed by the NovaCore team. |
| *(holdout)* Describe a cat in exactly 3 words. | Soft, fluffy, purring. | The Nova cat. |
| *(holdout)* 一加一等于几？ | 一加一等于 **2**。 | 一加一等于 2。 |

引擎侧单条回复（`max_tokens=64`，贪心）实测 **56.9 tok/s（关）vs 59.0 tok/s（开）** ——
适配器在 decode 上测不出代价（差在噪声里；开适配器那条只是答案更短）。这与
`LORA-MERGE.md` 的估算一致：adapter 15.7 MB 常驻、48 个投影各多一次 launch，上界 1~2%。

两列不一致的措辞（`我叫 Nova` vs `我叫 NovaCore`）不是两条路算得不一样，而是**不同提示下
同一份 LoRA 的不同采样**：表 1 走训练进程的 bf16 前向，表 2 走引擎的 IQ3_XXS native 路径，
量化差与上下文长度都会影响「同一个词选哪个」。

## 代价

| 项 | 数值 |
|---|---|
| 可训练参数 | 6,684,672（48 个模块：`o_proj` / `out_proj`，r=16、α=32） |
| 显存峰值（训练） | **34.1 GiB**（反传 + `--ckpt`；pack 24 GiB + 基座 9.2 GiB + 工作区） |
| 步时 | 8.5~9.2 s/step（第 1 步 24.4 s 冷启动；T≈25，`--accum 2`） |
| 加载 | 397 s（pack 279 s + 基座），33.2 GiB |
| adapter 体积 | .pt 8.1 MB / 适配器 GGUF 25.5 MiB（48 层、96 张量） |

## 诚实的三条副作用

1. **过拟合**。10 条样本跑了 40 步 × 2 = 80 个样本（8 个 epoch），答案被逐字记住，
   `Who are you?` 因此答成 `I am the NovaCore team.`（把团队名当成了自己）。要更自然的回答，
   样本要杂、步数要少，或者混入 holdout 做早停 —— 这一页不解决它，只是不藏它。
2. **词会漏到无关任务上**。`Describe a cat in exactly 3 words.` 变成 `The Nova cat.`；
   纯算数题基本没动（只丢了 markdown 粗体）。这说明 40 步已经改到了与身份无关的分布边缘。
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
  16 GiB 档能跑但慢（见 `BENCH-SFT-VS-MEZO.md` §12）。
- 引擎：`build-*/strata.exe`（要带 `--lora`，即 `--lora FILE / --lora-scaled FILE:SCALE` 出现在
  `--help` 里），配置里 `--max-context 4096` 就够演示，启动约 3 分钟（expert cache 38.97 GiB）。
