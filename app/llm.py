import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from app.config import AGENT_MODEL, GROQ_RPM, LLM_CACHE_PATH, MAX_RETRIES, REASONING_EFFORT

load_dotenv()


class DailyQuotaExceeded(RuntimeError):
    """Groq's per-day request/token cap is used up. Re-run later; cached work is kept."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    model: str = ""


STATS = {"api_calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}

_client_obj = None
_db = None
_last_call = {}
_effort_supported = True


def stats_summary():
    return (
        f"LLM usage this run: {STATS['api_calls']} API calls, {STATS['cache_hits']} cache hits, "
        f"{STATS['prompt_tokens']} prompt + {STATS['completion_tokens']} completion tokens"
    )


def _client():
    global _client_obj
    if _client_obj is None:
        from groq import Groq

        key = os.environ.get("GROQ_API_KEY")
        if not key or key == "your_key_here":
            raise RuntimeError("GROQ_API_KEY missing - copy .env.example to .env and paste your key")
        _client_obj = Groq(api_key=key, max_retries=0, timeout=90)  
    return _client_obj


def _cache():
    global _db
    if _db is None:
        LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _db = sqlite3.connect(str(LLM_CACHE_PATH))
        _db.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, text TEXT, "
            "prompt_tokens INTEGER, completion_tokens INTEGER)"
        )
        _db.commit()
    return _db


def _throttle(model):
    rpm = GROQ_RPM.get(model, 20)
    gap = 60.0 / rpm * 1.1
    wait = _last_call.get(model, 0.0) + gap - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call[model] = time.monotonic()


def _retry_after(err, default):
    resp = getattr(err, "response", None)
    if resp is not None:
        v = resp.headers.get("retry-after")
        if v:
            try:
                return float(v) + 1.0
            except ValueError:
                pass
    m = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|s|m)\b", str(err))
    if m:
        val, unit = float(m.group(1)), m.group(2)
        return val * {"ms": 0.001, "s": 1.0, "m": 60.0}[unit] + 1.0
    return default


def _call_api(model, messages, temperature, max_tokens, reasoning_effort):
    import groq

    global _effort_supported
    delay = 2.0
    for _ in range(MAX_RETRIES + 1):
        _throttle(model)
        kwargs = dict(
            model=model, messages=messages, temperature=temperature, max_completion_tokens=max_tokens
        )
        if reasoning_effort and _effort_supported:
            kwargs["reasoning_effort"] = reasoning_effort
        try:
            return _client().chat.completions.create(**kwargs)
        except groq.BadRequestError as e:
            if "reasoning_effort" in kwargs and "reasoning" in str(e).lower():
                _effort_supported = False 
                print("[llm] reasoning_effort not accepted; continuing without it")
                continue
            raise
        except groq.RateLimitError as e:
            msg = str(e)
            wait = _retry_after(e, delay)
            if re.search(r"(tokens|requests) per day|\(TPD\)|\(RPD\)", msg, re.I) or wait > 600:
                raise DailyQuotaExceeded(
                    f"Groq daily limit reached for {model}. Progress is cached - re-run later.\n{msg}"
                ) from e
            print(f"[llm] rate limited on {model}; sleeping {wait:.0f}s ...")
            time.sleep(wait)
            delay = min(delay * 2, 60)
        except (groq.APIConnectionError, groq.InternalServerError) as e:
            print(f"[llm] transient error ({type(e).__name__}); retrying in {delay:.0f}s ...")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(f"Groq call failed after {MAX_RETRIES + 1} attempts ({model})")


def chat(
    prompt,
    *,
    system=None,
    model=AGENT_MODEL,
    temperature=0.0,
    max_tokens=1024,
    reasoning_effort=REASONING_EFFORT,
    salt="",
):
    """One chat completion. `salt` lets you force a fresh (uncached) sample, e.g. on JSON retries."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    key = hashlib.sha256(
        json.dumps(
            [model, messages, temperature, max_tokens, reasoning_effort, salt], sort_keys=True
        ).encode("utf-8")
    ).hexdigest()

    db = _cache()
    row = db.execute(
        "SELECT text, prompt_tokens, completion_tokens FROM cache WHERE key=?", (key,)
    ).fetchone()
    if row:
        STATS["cache_hits"] += 1
        return LLMResponse(row[0], row[1], row[2], cached=True, model=model)

    budget, pt, ct, text = max_tokens, 0, 0, ""
    for _ in range(3):
        resp = _call_api(model, messages, temperature, budget, reasoning_effort)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        usage = getattr(resp, "usage", None)
        pt += getattr(usage, "prompt_tokens", 0) or 0
        ct += getattr(usage, "completion_tokens", 0) or 0
        STATS["api_calls"] += 1
        if text or choice.finish_reason != "length":
            break
        budget = min(budget * 2, 8192) 
    STATS["prompt_tokens"] += pt
    STATS["completion_tokens"] += ct
    if not text:
        raise RuntimeError(f"Empty response from {model}")

    db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)", (key, model, text, pt, ct))
    db.commit()
    return LLMResponse(text, pt, ct, cached=False, model=model)


def parse_json(text):
    """First JSON object found in an LLM reply (handles ```json fences and surrounding chatter)."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
            return obj
        except ValueError:
            continue
    raise ValueError("no JSON object found in model output")
