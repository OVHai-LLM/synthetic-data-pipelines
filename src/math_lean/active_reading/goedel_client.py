"""Multi-turn vLLM chat client for Goedel-Prover-V2 self-correction.

Why a separate client: the existing `VLLMClient.chat(system, user)` only
supports a single user message. Goedel-Prover-V2's self-correction protocol
needs to keep the prior assistant turn (the failed proof attempt) in the
chat history when sending the next user turn (with the compile error).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests

DEFAULT_ENDPOINT = "http://localhost:8002"
DEFAULT_TIMEOUT = 600  # Goedel long-CoT can take a while


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str


class GoedelClient:
    """Multi-turn chat client for Goedel-Prover-V2 served via vLLM OpenAI API."""

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT,
                 model: str = "Goedel-LM/Goedel-Prover-V2-32B",
                 *,
                 api_key: Optional[str] = None,
                 timeout: int = DEFAULT_TIMEOUT):
        self.url = endpoint.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def chat(self, messages: List[ChatMessage] | List[dict],
             *,
             temperature: float = 1.0,
             top_p: float = 0.95,
             max_tokens: int = 32768,
             n: int = 1,
             retries: int = 3,
             stop: Optional[list] = None) -> List[str]:
        """Send a multi-turn chat. Returns a list of `n` completion strings."""
        msg_payload = []
        for m in messages:
            if isinstance(m, ChatMessage):
                msg_payload.append({"role": m.role, "content": m.content})
            else:
                msg_payload.append(m)

        payload = {
            "model": self.model,
            "messages": msg_payload,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": n,
        }
        if stop:
            payload["stop"] = stop

        last_err = None
        for attempt in range(retries + 1):
            try:
                r = self._session.post(self.url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                return [c["message"]["content"] for c in data["choices"]]
            except (requests.RequestException, KeyError, ValueError) as e:
                last_err = e
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"vLLM Goedel request failed after {retries+1} tries: {last_err}")
