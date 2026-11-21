"""Strata 的 HTTP 客户端。只做一件事：把 prompt 送进引擎，拿回序列与文本。

两个必须处理的现实问题：
  1. WSL 里 `http_proxy=127.0.0.1:1080` 会把 127.0.0.1 的请求也交给 privoxy，
     直接 urllib 会得到 `HTTP 500 Internal Privoxy Error` —— 回环必须绕开代理。
  2. 老引擎可能不返回 token id（q36 在补上 `LP` 行之前就是），这时只能回退到
     文本重编码。重编码与真实采样可能有个别边界差异，所以返回值里明确标了
     `tokens_exact`，训练时据此决定要不要信任 ids。当前两个引擎（generate.cpp
     的 serve 循环、q36_engine.cu）都产 LP，正常情况下 `tokens_exact` 为真。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def chat(base_url: str, messages: list[dict], *, model: str, max_tokens: int = 0,
         temperature: float = 1.0, top_p: float = 0.95, seed: int | None = None,
         timeout: float = 600.0, enable_thinking: bool | None = None,
         reasoning_effort: str | None = None, extra_sampling: dict | None = None) -> dict:
    """`reasoning_effort` 必须与训练侧同档：模板默认 `xhigh`，会插一整段 system 指令。
    实测（2026-11-18）不传时采样 prompt 21 token、训练模板 61 token —— 两条不同的上下文，
    奖励与 logp 就都落在不是训练用的那条上。

    `max_tokens` 默认 **0 = 不设上限**：`serve/server.py` 把 `max_tokens <= 0` 解释成
    「用满剩余上下文」（`max_new = ctx - 8 - len(prompt)`），超了也只收敛到剩余空间、不报错。
    所以「写不写得完」由上下文窗口决定，不由客户端的一个小数决定 —— 给 48/128 这种值会让
    长推理链每条都截在半路，而那**不报错**，只是把「没答完」记成「答错」，训练就在这种
    数据上跑。真要截断（例如刻意做短回答）必须显式给正数，并自己看 `finish_reason`。
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": True,
        "top_logprobs": 0,
        "strata_rollout": True,
    }
    # 显式压过 server 的 model_card_sampling 默认（qwen4exp thinking 卡面带 top_k 20 等，
    # 不传就会漏进来 —— T=0 也挡不住 top_k 截断翻 token，rollout/评测口径就分叉了）
    if extra_sampling:
        payload.update(extra_sampling)
    _kw = {}
    if enable_thinking is not None:
        _kw["enable_thinking"] = enable_thinking
    if reasoning_effort is not None:
        _kw["reasoning_effort"] = reasoning_effort
    if _kw:
        payload["chat_template_kwargs"] = _kw
    if seed is not None:
        payload["seed"] = seed
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with _OPENER.open(req, timeout=timeout) as resp:
        raw = json.loads(resp.read())

    choice = raw["choices"][0]
    message = choice.get("message") or {}
    rollout = raw.get("strata_rollout") or {}
    tokens = rollout.get("tokens") or []
    return {
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or "",
        "finish_reason": choice.get("finish_reason"),
        "prompt_ids": rollout.get("prompt_ids") or [],
        "token_ids": [t["id"] for t in tokens],
        "token_logprobs": [t["logprob"] for t in tokens],
        "fields": [t.get("field") for t in tokens],
        "tokens_exact": bool(tokens) and bool(rollout.get("prompt_ids")),
        "usage": raw.get("usage") or {},
    }
