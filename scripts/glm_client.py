"""Minimal, rate-limit-aware client for the Zhipu/BigModel GLM API.

THE KEY IS NEVER STORED. It is read from $GLM_API_KEY at call time and is not
written to disk, logs, or any generated artifact. Export it in the shell that
runs the generator:

    export GLM_API_KEY='...'

MEASURED 2026-09-16 on the supplied account:
  * glm-4.5 / glm-4.6 / glm-4.5-air / glm-4-plus -> HTTP 429 code 1113
    "insufficient balance or no available resource package". Unusable until
    the account is topped up.
  * glm-4.5-flash -> works, but the account trips code 1302 ("rate limit,
    please control request frequency") under even light bursts, and requests
    then stall rather than failing fast.
So: one request at a time, a floor on the gap between requests, and a long
exponential backoff on 1302. Throughput, not quality, is the binding
constraint on this account.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time

import requests

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-4.5-flash"

# Paid-tier models, in preference order, tried by --model auto.
PAID_MODELS = ("glm-4.6", "glm-4.5", "glm-4.5-air")

RATE_LIMITED = "1302"
NO_BALANCE = "1113"


class RateLimit(Exception):
    """The account is being throttled; the caller should slow down."""


class NoBalance(Exception):
    """The model needs credit the account does not have."""


class GLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, min_gap: float = 3.0,
                 timeout: float = 180.0, max_retries: int = 6,
                 verbose: bool = True) -> None:
        key = os.environ.get("GLM_API_KEY", "").strip()
        if not key:
            raise SystemExit(
                "GLM_API_KEY is not set. Export it in this shell:\n"
                "    export GLM_API_KEY='<your key>'\n"
                "It is deliberately not read from any file.")
        self._key = key
        self.model = model
        self.min_gap = min_gap
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose
        self._lock = threading.Lock()
        self._last = 0.0
        self.calls = 0
        self.retries = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def _pace(self) -> None:
        """Serialise requests and hold a floor between them."""
        with self._lock:
            wait = self.min_gap - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

    def chat(self, prompt: str, system: str | None = None,
             max_tokens: int = 4096, temperature: float = 0.85) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        body = {
            "model": self.model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # GLM-4.5+ reasons by default and spends ~700 tokens doing it even
            # on trivial output. Disabled: measured 736 -> 28 completion tokens.
            "thinking": {"type": "disabled"},
        }
        delay = 8.0
        for attempt in range(self.max_retries):
            self._pace()
            try:
                r = requests.post(
                    API_URL, timeout=self.timeout, json=body,
                    headers={"Authorization": f"Bearer {self._key}",
                             "Content-Type": "application/json"})
            except requests.RequestException as e:
                self.retries += 1
                if self.verbose:
                    print(f"    [net {type(e).__name__}] backing off {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 180)
                continue

            if r.status_code == 200:
                d = r.json()
                self.calls += 1
                u = d.get("usage") or {}
                self.tokens_in += u.get("prompt_tokens", 0)
                self.tokens_out += u.get("completion_tokens", 0)
                return (d["choices"][0]["message"].get("content") or "").strip()

            try:
                err = r.json().get("error", {})
            except Exception:
                err = {}
            code = str(err.get("code", ""))
            if code == NO_BALANCE:
                raise NoBalance(
                    f"{self.model}: {err.get('message','insufficient balance')}")
            if code == RATE_LIMITED or r.status_code == 429:
                self.retries += 1
                jitter = delay * (0.5 + random.random())
                if self.verbose:
                    print(f"    [429 rate limit] backing off {jitter:.0f}s")
                time.sleep(jitter)
                delay = min(delay * 2, 240)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        raise RateLimit(f"gave up after {self.max_retries} attempts")


def extract_jsonl(text: str) -> list[dict]:
    """Pull JSON objects out of a model reply.

    Tolerates ```json fences, a JSON array, or one object per line -- the three
    shapes GLM actually returns -- because a strict parser here would discard
    otherwise-valid batches over formatting alone.
    """
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    t = t.strip()
    out: list[dict] = []
    try:
        d = json.loads(t)
        if isinstance(d, list):
            return [x for x in d if isinstance(x, dict)]
        if isinstance(d, dict):
            return [d]
    except Exception:
        pass
    for line in t.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
            if isinstance(o, dict):
                out.append(o)
        except Exception:
            continue
    return out
