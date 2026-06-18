"""DeepSeek-V4 Flash (no-thinking) client with a deterministic on-disk prompt cache.

Design decisions (experiment-plan §1.3, §6):
  - Model: DeepSeek-V4 Flash, **thinking disabled** (the researcher only has Flash; thinking
    must be explicitly off).  Configured via ``LLMConfig`` so the exact model string can be
    pinned once the provider's endpoint is confirmed.
  - Cache: every call is keyed by a SHA-256 hash of (model, messages, temperature, max_tokens,
    extra params).  Cache hits cost nothing and make ablations reproducible; this also lets the
    whole infra be unit-tested with no network (see ``mock`` mode).
  - Cost: stable instruction + few-shot prefix should be kept identical across calls so the
    provider's prompt-cache discount applies (hit ~50x cheaper than cache-miss).  This client
    does not reorder messages, preserving that prefix.

DeepSeek's API is OpenAI-compatible, so the **official OpenAI SDK** is the preferred transport
(httpx connection pooling + keep-alive + built-in retry/backoff + native ``extra_body`` for the
thinking-disable flag).  If ``openai`` isn't installed we fall back to a pooled ``requests``
session, then to ``urllib``.  No network is touched unless ``call`` is invoked with a real key
and the prompt is not already cached.

Environment:
  DEEPSEEK_API_KEY   - API key (required for live calls)
  DEEPSEEK_BASE_URL  - override base url (default https://api.deepseek.com)
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import random
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class LLMConfig:
    model: str = "deepseek-v4-flash"      # DeepSeek-V4 Flash (no thinking); confirmed 2026-06-14
    base_url: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    temperature: float = 0.0
    max_tokens: int = 1024
    # Provider-specific knob to disable thinking; kept explicit and serialized into the cache key.
    extra_body: Dict = field(default_factory=lambda: {"thinking": {"type": "disabled"}})
    timeout: int = 120
    max_retries: int = 6
    retry_backoff: float = 1.5
    pool_size: int = 64           # HTTP keep-alive connection pool (set ~= workers)


@dataclass
class LLMResponse:
    text: str
    cached: bool
    raw: Optional[dict] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class LLMClient:
    """Cached DeepSeek-V4 Flash client.

    Args:
      config: model/sampling configuration.
      cache_dir: directory for the prompt cache (one JSON file per prompt hash).
      mock: if True, never hit the network; raise on cache miss (for offline unit tests),
            unless a ``mock_fn`` is supplied to synthesize responses.
      mock_fn: optional ``(messages) -> str`` to produce deterministic fake completions.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        cache_dir: str | os.PathLike = "outs/llm_cache",
        mock: bool = False,
        mock_fn=None,
    ):
        self.config = config or LLMConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mock = mock
        self.mock_fn = mock_fn
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.n_calls = 0
        self.n_cache_hits = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self._lock = threading.Lock()  # guards counters (safe under --workers)
        self._session = None           # lazy keep-alive HTTP session (requests fallback)
        self._session_lock = threading.Lock()
        self._oai = None               # lazy OpenAI SDK client (preferred)

    # -- cache key ---------------------------------------------------------- #
    def _key(self, messages: List[dict], temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": self.config.extra_body,
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # -- public API --------------------------------------------------------- #
    def call(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Run one chat completion, using the disk cache when possible."""
        temperature = self.config.temperature if temperature is None else temperature
        max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        key = self._key(messages, temperature, max_tokens)
        path = self._cache_path(key)

        if path.exists():
            with self._lock:
                self.n_cache_hits += 1
            data = json.loads(path.read_text(encoding="utf-8"))
            return LLMResponse(
                text=data["text"], cached=True, raw=data.get("raw"),
                prompt_tokens=data.get("prompt_tokens"),
                completion_tokens=data.get("completion_tokens"),
            )

        if self.mock:
            if self.mock_fn is not None:
                text = self.mock_fn(messages)
                self._write_cache(path, text, None, None, None)
                return LLMResponse(text=text, cached=False)
            raise RuntimeError("LLMClient(mock=True): cache miss and no mock_fn provided")

        text, raw, pin, pout = self._call_api(messages, temperature, max_tokens)
        with self._lock:
            self.n_calls += 1
            if pin:
                self.tokens_in += pin
            if pout:
                self.tokens_out += pout
        self._write_cache(path, text, raw, pin, pout)
        return LLMResponse(text=text, cached=False, raw=raw, prompt_tokens=pin, completion_tokens=pout)

    def _write_cache(self, path: Path, text: str, raw, pin, pout) -> None:
        path.write_text(json.dumps(
            {"text": text, "raw": raw, "prompt_tokens": pin, "completion_tokens": pout},
            ensure_ascii=False,
        ), encoding="utf-8")

    def _get_openai(self):
        """Lazily build the official OpenAI SDK client (DeepSeek is OpenAI-compatible).

        The SDK gives httpx connection pooling + keep-alive + built-in retry/backoff and
        native ``extra_body`` (for the thinking-disable flag).  Returns None if the package
        isn't installed (then we fall back to requests/urllib).
        """
        if self._oai is not None:
            return self._oai or None
        with self._session_lock:
            if self._oai is None:
                try:
                    from openai import OpenAI
                    kwargs = dict(base_url=self.config.base_url, api_key=self.api_key,
                                  max_retries=self.config.max_retries, timeout=self.config.timeout)
                    try:
                        import httpx
                        limits = httpx.Limits(max_connections=self.config.pool_size,
                                              max_keepalive_connections=self.config.pool_size)
                        kwargs["http_client"] = httpx.Client(limits=limits, timeout=self.config.timeout)
                    except Exception:  # noqa: BLE001 — httpx tuning optional
                        pass
                    self._oai = OpenAI(**kwargs)
                except Exception:  # noqa: BLE001 — openai not installed
                    self._oai = False
        return self._oai or None

    def _get_session(self):
        """Lazily create a keep-alive, connection-pooled requests.Session (thread-safe).

        Connection reuse avoids a fresh TCP+TLS handshake on every call — the main speedup
        for many small low-latency requests.  Returns None if `requests` is unavailable
        (then we fall back to urllib, one connection per call)."""
        if self._session is not None:
            return self._session
        with self._session_lock:
            if self._session is None:
                try:
                    import requests
                    from requests.adapters import HTTPAdapter
                    s = requests.Session()
                    ad = HTTPAdapter(pool_connections=self.config.pool_size,
                                     pool_maxsize=self.config.pool_size, max_retries=0)
                    s.mount("https://", ad)
                    s.mount("http://", ad)
                    self._session = s
                except Exception:  # noqa: BLE001 — requests not installed
                    self._session = False  # sentinel: use urllib
        return self._session or None

    def _call_api(self, messages, temperature, max_tokens):
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set; cannot make live calls")
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body.update(self.config.extra_body or {})
        # --- preferred path: official OpenAI SDK (pooling + retries + extra_body) ---
        oai = self._get_openai()
        if oai is not None:
            resp = oai.chat.completions.create(
                model=self.config.model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
                extra_body=self.config.extra_body or {},
            )
            obj = resp.model_dump()
            text = obj["choices"][0]["message"]["content"]
            usage = obj.get("usage") or {}
            return text, obj, usage.get("prompt_tokens"), usage.get("completion_tokens")

        # --- fallback: requests/urllib with manual retry ---
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "deer/0.1",
        }
        # Transient errors worth retrying: connection drops / timeouts / 429 / 5xx.
        transient = (urllib.error.URLError, http.client.HTTPException,
                     ConnectionError, TimeoutError, socket.timeout, OSError, KeyError)
        session = self._get_session()
        last_err = None
        for attempt in range(self.config.max_retries):
            try:
                if session is not None:
                    r = session.post(url, json=body, headers=headers, timeout=self.config.timeout)
                    if r.status_code == 429 or 500 <= r.status_code < 600:
                        raise http.client.HTTPException(f"HTTP {r.status_code}")
                    if r.status_code >= 400:
                        raise RuntimeError(f"DeepSeek API client error {r.status_code}: {r.text[:300]!r}")
                    obj = r.json()
                else:
                    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                                 headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                        obj = json.loads(resp.read().decode("utf-8"))
                text = obj["choices"][0]["message"]["content"]
                usage = obj.get("usage", {}) or {}
                return text, obj, usage.get("prompt_tokens"), usage.get("completion_tokens")
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code != 429 and not (500 <= e.code < 600):
                    raise RuntimeError(f"DeepSeek API client error {e.code}: {e.read()[:300]!r}")
                time.sleep(self.config.retry_backoff * (2 ** attempt) + random.uniform(0, 1))
            except transient as e:
                last_err = e
                time.sleep(self.config.retry_backoff * (2 ** attempt) + random.uniform(0, 1))
        raise RuntimeError(f"DeepSeek API call failed after {self.config.max_retries} retries: {last_err}")

    def stats(self) -> dict:
        return {
            "live_calls": self.n_calls,
            "cache_hits": self.n_cache_hits,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }
