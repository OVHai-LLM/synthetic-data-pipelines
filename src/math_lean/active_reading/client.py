"""Minimal vLLM /v1/chat client with retry/backoff.

Self-contained so the prototype can run from the math_lean venv without
depending on the math_generator package.

Supports multi-endpoint DP routing with **prefix-cache-aware client-side
affinity**: pass `endpoint` as either a single URL (single-engine mode)
or a comma-separated list of URLs (one per DP rank). When a list is
given, every `chat()` call must include a `pin_key` (typically the
article id); the client hashes it modulo the number of endpoints and
sends to the same rank for every call sharing that key. This guarantees
all 6 apply calls per article hit the same engine, so vLLM's per-engine
prefix cache reuses the article's KV blocks across the calls.

vLLM's built-in DP load balancer is round-robin (NOT prefix-aware), so
a 6-call article's calls would land on ~all 4 ranks → cache cold every
call. Client-side affinity sidesteps that: launch vllm with
`--data-parallel-size 4 --data-parallel-external-lb` (one port per
rank), then point this client at the comma-joined list.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Optional

import requests

DEFAULT_ENDPOINT = "http://localhost:8002"
DEFAULT_TIMEOUT = 240


def env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def _parse_endpoints(endpoint: str) -> list[str]:
    """Accept 'http://h:8002' or 'http://h:8002,http://h:8003,...'."""
    return [e.strip().rstrip("/") for e in endpoint.split(",") if e.strip()]


def _pin_to_rank(pin_key: str, n_ranks: int) -> int:
    """Stable hash → rank index. md5 is fine; we just need an even bucketing."""
    h = hashlib.md5(pin_key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % n_ranks


class VLLMClient:
    def __init__(self, endpoint: str, model: str, *,
                 api_key: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        self._endpoints = _parse_endpoints(endpoint)
        if not self._endpoints:
            raise ValueError(f"no endpoints parsed from {endpoint!r}")
        self._urls = [e + "/v1/chat/completions" for e in self._endpoints]
        self.model = model
        self.timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._rr_counter = 0  # round-robin fallback when no pin_key

    @property
    def n_ranks(self) -> int:
        return len(self._urls)

    def _pick_url(self, pin_key: Optional[str]) -> str:
        if self.n_ranks == 1:
            return self._urls[0]
        if pin_key is not None:
            return self._urls[_pin_to_rank(str(pin_key), self.n_ranks)]
        # No pin key → round-robin so a careless caller doesn't pile on one rank.
        idx = self._rr_counter % self.n_ranks
        self._rr_counter += 1
        return self._urls[idx]

    def chat(self, system: str, user: str, *,
             temperature: float = 0.8, max_tokens: int = 2048,
             retries: int = 3, pin_key: Optional[str] = None) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        url = self._pick_url(pin_key)
        last_err = None
        for attempt in range(retries + 1):
            try:
                r = self._session.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except (requests.RequestException, KeyError, ValueError) as e:
                last_err = e
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"vLLM request failed after {retries+1} tries on {url}: {last_err}")