"""Moved unchanged out of serve/server.py (file split, cleanup step 5): the OpenAI part. `serve.server` imports these names back,
so every caller and test still reaches them as `serve.server.<name>`."""
from __future__ import annotations

import collections
import codecs
import json
import math
from serve.structured import extract_json


# The engine's last PP line stops short of the prompt: up to `--short-read` tokens (generate.cpp's `short_read`, 64 by
# default) are read through the verify windows instead of the batched path, so the batched path never reports the whole
# prompt.  A PP that close to the end says the prompt is read.  It matters because a prompt shorter than one `--prefill`
# chunk sends exactly one PP, that one: a client would see a bar that appears already full.  Such a line is dropped, and
# the answer's first token ends the progress for the client anyway.
PP_DONE_TAIL = 64


class LPLines(collections.deque):
    """One request's LP lines, tagged with its cancel Event (one per request, handed unchanged from the handler
    through Service.run to Engine.generate).  The last events of a request - the parser's and the stop strings'
    tail - come after it has left the queue, when the next request may already have replaced the engine's lp_lines
    with its own: LogprobEntries reads only the lines tagged with its own request's Event."""

    def __init__(self, owner=None):
        super().__init__()
        self.owner = owner


# The sampling a rollout echoes: a stored logprob is only comparable to a forward pass run later if the temperature,
# the truncation and the penalties it was taken under are known too.  The values are the ones the engine was given
# (Service.sampling_for), verbatim -- it clamps top_k past 64 and reads seed <= 0 as unset.
ROLLOUT_SAMPLING = ("temperature", "top_p", "top_k", "min_p", "frequency_penalty", "presence_penalty",
                    "repetition_penalty", "penalty_last_n", "seed")


class LogprobEntries:
    """The engine's LP lines: one record per generated token (the sequence), and OpenAI's logprobs.content
    entries as one projection of it (each token's text, given to the piece of text that spells it).

    The sequence is read first and kept whole (`seq`): every token, with the id it was sampled from and the
    logprob as the engine wrote it, whatever the text view later does with it.  The text view is derived from
    it, so a token no piece of text claims -- the blank run between the thinking and the answer, say -- keeps
    its place in the sequence and is labelled "gap" instead of being dropped, and the end token is labelled
    "stop" rather than being skipped.  Nothing a caller may want back is destroyed by reading the text.

    Each token's "token" is the text it completes (an incremental UTF-8 decode, as the reply's
    text is made), so a character split across tokens lands in the token that finishes it and
    the tokens, joined, spell the reply.  "bytes" are the token's own bytes.  The stop token
    ends the reply and is not listed (OpenAI does not list it either).  Tokens that make no
    text of the reply are left out of the text view, because spell() gives each piece of text
    the tokens that spell it.

    `allowed_token_ids`: only those ids are kept, and the logprobs are renormalised over them,
    so a caller asking about specific candidates gets a distribution over exactly those instead
    of a top-K list it has to hope contains them.  vLLM, which named the field, masks generation
    with these ids instead (every other logit -inf before sampling) and reports the unmasked
    distribution; here they filter what is reported, so the chosen token can be outside them.

    On `--parallel`: only a `max_tokens` 1 request gets entries (an admission, decoded through the single-token path
    that writes an LP line); the slot's `BT` lines carry none, so a longer reply is refused (400) before it starts.
    """

    def __init__(self, svc, allowed=(), owner=None):
        self.tok, self.stop_ids = svc.tok, svc.stop_ids
        self.allowed = [i for i in allowed]
        self.inc = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.seq = []                              # every token read, in order: the sequence (see read())
        self.pending = []                          # records read but not yet given to a piece of text
        self.owner = owner                         # the request's cancel Event (LPLines)
        self.lines = None                          # this request's current LPLines, kept once seen

    def spell(self, eng, text: str, field: str = "content") -> list:
        """The entries whose tokens spell `text`, the next piece of the reply, and the field they belong to.

        The records stay in `seq` with the field written on them: this is a projection of the sequence, not a
        way of consuming it.  A token before the piece (a blank run the parser left out of the text) is
        labelled "gap" and kept; it used to be dropped here, which is why the entries stopped being a
        sequence and a caller had no way to tell.
        """
        self.pending += self.read(eng)
        recs = self.pending
        at = [0]
        for r in recs:
            at.append(at[-1] + len(r["e"]["token"]))
        joined = "".join(r["e"]["token"] for r in recs)
        i = joined.find(text)
        while i >= 0 and text:
            if i in at:
                a = at.index(i)                    # the first token there
                end = i + len(text)
                b = next((j for j in range(a + 1, len(at)) if at[j] == end), None)
                if b is not None:
                    out, self.pending = recs[a:b], recs[b:]
                    for r in recs[:a]:
                        r["field"] = "gap"
                    for r in out:
                        r["field"] = field
                    return [r["e"] for r in out]
            i = joined.find(text, i + 1)
        out, self.pending = recs, []
        for r in out:
            r["field"] = field
        return [r["e"] for r in out]

    def raw(self, t):
        if hasattr(self.tok, "token_bytes"):
            return self.tok.token_bytes(t)
        return self.tok.decode([t]).encode("utf-8")

    def take(self, eng) -> list:
        """The OpenAI entries for the tokens read since the last call (see read())."""
        return [r["e"] for r in self.read(eng) if r["e"] is not None]

    def read(self, eng) -> list:
        """Every token the engine has written since the last call, in order: one record per generated token.

        A record is `{"id", "logprob", "field", "top", "e"}`: the id it was sampled from, the logprob the engine
        wrote for it (raw: before penalties, temperature and truncation), the field it spells once a piece of
        text claims it (None until then), the ids that line carried, and `e`, the OpenAI entry.  `e` is None for
        a stop token: its bytes are not part of the reply's text, and decoding them would move the incremental
        decoder that text is made by.
        """
        # the engine's current lines are this request's only when they carry its Event (a later generate pass
        # of the same request makes new ones: the old ones are read to the end first); another request's never are.
        # A batch admission keeps its own lines on the request's cancel Event: the engine's is one attribute, and
        # the next request to read control lines may already have replaced it.
        own = getattr(self.owner, "_strata_lp_lines", None)
        cur, old = (own if own is not None else getattr(eng, "lp_lines", None)), None
        if cur is not self.lines and getattr(cur, "owner", None) is self.owner:
            old, self.lines = self.lines, cur
        out = []
        for lines in (old, self.lines):
            while lines:
                t, line = lines.popleft()
                stop = t in self.stop_ids
                f = line.split()
                lp, alts = None, []
                if len(f) >= 2:
                    try:                           # "nan" (no head logits for that window row), "-nan(ind)"
                        lp = float(f[1])           # (MSVC) or a cut line: no record for that token
                        alts = [(int(i), float(v)) for i, _, v in (x.partition(":") for x in f[2:])]
                    except ValueError:
                        lp, alts = None, []
                if lp is None or not math.isfinite(lp) or not all(math.isfinite(v) for _, v in alts):
                    lp, alts = None, []
                if stop:                           # the end token is in the sequence, not in any text
                    self.seq.append({"id": t, "logprob": lp, "field": "stop", "top": alts, "e": None})
                    continue
                if lp is None:
                    continue
                b = self.raw(t)
                e = {"token": self.inc.decode(b), "logprob": lp, "bytes": list(b),
                     "top_logprobs": []}
                rec = {"id": t, "logprob": lp, "field": None, "top": alts, "e": e}
                self.seq.append(rec)
                if self.allowed:
                    alts = [(i, v) for i, v in alts if i in self.allowed]
                for i, v in alts:
                    alt = self.raw(i)
                    e["top_logprobs"].append({"token": alt.decode("utf-8", "replace"),
                                              "logprob": v, "bytes": list(alt)})
                if self.allowed:
                    e = self._renormalise(e, t, lp)
                out.append(rec)
        return out

    def _renormalise(self, e, t, lp):
        """Keep only the named ids and normalise over them, so they sum to 1.

        The line carries the chosen token's own logprob on its head and the named ids after it, and never
        repeats the chosen one among them.  So the chosen token is put back when the caller named it: with
        `allowed_token_ids` naming only the token the model chose, that token is then the whole distribution,
        at 1 (logprob 0), which is what vLLM's mask gives -- here it used to leave nothing to normalise over,
        which returned None, the entry was appended all the same, and the reply failed on e["token"].  A line
        that carries none of the named ids has nothing to normalise over either: the entry keeps the chosen
        token and its own logprob, with no alternatives.  It is never None and never dropped, because the
        reply's tokens have to keep spelling its text.
        """
        cand = e["top_logprobs"]
        own = None
        if t in self.allowed:                      # named, and not repeated on the line: it is a candidate
            b = self.raw(t)
            own = {"token": b.decode("utf-8", "replace"), "logprob": lp, "bytes": list(b)}
            cand = [own] + cand
        if not cand:
            return e                               # no named id observed for this token: its own logprob, no top
        mx = max(x["logprob"] for x in cand)
        se = sum(math.exp(x["logprob"] - mx) for x in cand)
        lse = mx + math.log(se)
        for x in cand:
            x["logprob"] -= lse
        cand.sort(key=lambda x: -x["logprob"])     # most likely first, as the unfiltered top-k already is
        e["top_logprobs"] = cand
        if own is not None:                        # the chosen token is in the distribution: report its own value
            e["logprob"] = own["logprob"]           # (it used to copy the first alternative's, another token's)
        return e

    def finish(self, eng) -> None:
        """Read what is left after the reply's last piece of text, and label what no piece claimed.

        Such a token is "tail": in the sequence because the engine generated it, in no OpenAI array because it
        spells no text of the reply.  Reading the rest here is what makes `seq` the whole reply rather than a
        prefix of it.
        """
        for r in self.pending + self.read(eng):
            if r["field"] is None:
                r["field"] = "tail"
        self.pending = []

    def rollout(self, sampling=None, prompt_ids=None) -> dict:
        """The whole sequence this request generated: one entry per token, in order, from its id.

        This is what a trainer reads, and `choices[0].logprobs` is not: same tokens, but nothing dropped at a
        field boundary, every token carrying the id it was sampled from, and `logprob` the engine's own value
        before `allowed_token_ids` renormalises what the text view reports (that projection answers "how likely
        among the ids I named"; this one answers "how likely").  With the ids a trainer re-runs the forward pass
        itself and back-propagates through it; the numbers here are the cross-check.

        `prompt_ids` is the prompt those tokens continue, as the engine read it.  A trainer needs it because the
        alternative -- tokenizing the prompt again on its own side -- can apply a different chat template and
        then every position after the first difference is off: the logprobs here would look like a mismatch of
        the model rather than of the template.  These are the ids `prepare()` handed the engine, and since
        requests are never truncated they are the whole prompt, not a prefix of it.
        """
        tokens = []
        for r in self.seq:
            e = r["e"]
            b = e["bytes"] if e is not None else list(self.raw(r["id"]))
            tok = {"id": r["id"], "logprob": r["logprob"], "field": r["field"] or "gap",
                   "token": e["token"] if e is not None else bytes(b).decode("utf-8", "replace"),
                   "bytes": b}
            if r["top"]:
                tok["top_logprobs"] = [{"id": i, "logprob": v} for i, v in r["top"]]
            tokens.append(tok)
        return {"prompt_ids": [int(t) for t in prompt_ids] if prompt_ids else [],
                "tokens": tokens, "logprobs_mode": "raw",
                "sampling": {k: sampling[k] for k in ROLLOUT_SAMPLING if (sampling or {}).get(k) is not None}}


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def openai_collect(chunks) -> dict:
    content, reasoning, by_index, last, mcp = [], [], {}, None, []
    lps = {}                                       # "content" / "reasoning_content" -> logprob entries
    for c in chunks:
        if c is None:                              # a heartbeat
            continue
        if c.get("strata_mcp"):
            mcp.append(c["strata_mcp"])
        for k, v in (c["choices"][0].get("logprobs") or {}).items():
            lps.setdefault(k, []).extend(v)
        d = c["choices"][0]["delta"]
        content.append(d.get("content") or "")
        reasoning.append(d.get("reasoning_content") or "")
        for tc in d.get("tool_calls") or []:       # streamed calls arrive in pieces: merge them by index
            cur = by_index.setdefault(tc.get("index", len(by_index)), {"id": None, "type": "function",
                                                                        "function": {"name": "", "arguments": ""}})
            cur["id"] = tc.get("id") or cur["id"]
            fn = tc.get("function") or {}
            cur["function"]["name"] += fn.get("name") or ""
            cur["function"]["arguments"] += fn.get("arguments") or ""
        last = c
    calls = [by_index[i] for i in sorted(by_index)]
    if last["choices"][0]["finish_reason"] != "tool_calls":
        calls = [c for c in calls if _is_json(c["function"]["arguments"])]   # a call the output ended inside (#211)
    # null content only beside tool calls, as OpenAI sends it; a reply that ran out of tokens while thinking gets ""
    # so strict OpenAI clients and proxies that require a string do not fail on it.
    msg = {"role": "assistant", "content": "".join(content) or (None if calls else "")}
    if "".join(reasoning).strip() and "".join(reasoning).strip() == "".join(content).strip():
        reasoning = []              # a turn with no </think>: the text is the answer; it is not also reasoning
    if "".join(reasoning):
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = calls
    if mcp:
        msg["strata_mcp"] = mcp
    out = {"id": last["id"], "object": "chat.completion", "created": last["created"], "model": last["model"],
           "choices": [{"index": 0, "message": msg, "finish_reason": last["choices"][0]["finish_reason"]}],
           "usage": last["usage"]}
    if lps:
        out["choices"][0]["logprobs"] = {"content": lps.get("content", []), "refusal": None, **(
            {"reasoning_content": lps["reasoning_content"]} if lps.get("reasoning_content") else {})}
    if last.get("timings"):
        out["timings"] = last["timings"]
    if last.get("strata_rollout"):
        out["strata_rollout"] = last["strata_rollout"]     # the sequence the last chunk carried (see openai_chunks)
    if last.get("strata_rejected_tool_calls"):
        out["strata_rejected_tool_calls"] = last["strata_rejected_tool_calls"]
    if last.get("strata_unread_tool_markup"):
        out["strata_unread_tool_markup"] = last["strata_unread_tool_markup"]
    return out


def logprob_span(text: str, part: str, entries: list):
    """The entries whose tokens spell `part`, a piece of `text`, or None when the tokens do not
    spell `text` or no token boundary falls at both ends of `part`."""
    if "".join(e["token"] for e in entries) != text:
        return None
    start = text.find(part)
    if start < 0:
        return None
    at, first = 0, None
    for i, e in enumerate(entries):
        if at == start and first is None:
            first = i
        if at == start + len(part) and first is not None:
            return entries[first:i]
        at += len(e["token"])
    return entries[first:] if first is not None and at == start + len(part) else None


def _same_json(a: str, b: str) -> bool:
    """Are two strings the same JSON value (whitespace and key order aside)?  No -> False."""
    try:
        return json.loads(a) == json.loads(b)
    except Exception:
        return False


def _structured_logprobs(choice: dict, content: str):
    """`content`'s logprobs, and the content the tokens actually spell.

    Returns `(content, logprobs)`: on failure `(the content as validated, None)`.

    `structured_chunks` returns the canonical JSON the validator produced, which the model never
    generated, so its tokens do not spell it.  Two honest outcomes:

      1. a run of tokens spells what is being returned -> use it as it is;
      2. the tokens spell the model's own JSON, which is the same JSON value (the fenced-with-spaces
         case, the common one) -> return the model's form, so `''.join(token) == content` again.

    Neither holds (a rearranged key order, or a reply that is not that JSON at all) and there are no
    logprobs: a caller that joins the tokens must not silently get something that does not match.
    """
    lps = choice.get("logprobs")
    if not lps or not lps.get("content"):
        return content, None
    raw = choice["message"]["content"] or ""
    entries = lps["content"]
    if raw == content:                          # not rewritten: the entries are already its
        return content, _with_reasoning(lps, entries)
    span = logprob_span(raw, content, entries)
    if span is not None:
        return content, _with_reasoning(lps, span)
    inner = extract_json(raw)
    if inner and _same_json(inner, content):
        span = logprob_span(raw, inner, entries)
        if span is not None:
            return inner, _with_reasoning(lps, span)   # the model's own form: the tokens spell it
    return content, None


def _with_reasoning(lps: dict, entries: list) -> dict:
    out = {"content": entries}
    if lps.get("reasoning_content"):
        out["reasoning_content"] = lps["reasoning_content"]
    return out
