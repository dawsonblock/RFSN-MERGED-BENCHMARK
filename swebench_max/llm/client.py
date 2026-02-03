from __future__ import annotations
import os
import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass(frozen=True)
class LLMConfig:
    provider: str  # "openai" | "anthropic" | "mistral" | "generic_http"
    model: str
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: int = 120
    max_tokens: int = 2000
    temperature: float = 0.2


class LLMClient:
    """
    Minimal HTTP client wrapper.
    - No streaming
    - No tools
    - Returns plain text
    You can swap this to your existing API layer later.
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.api_key = os.environ.get(cfg.api_key_env, "")

    def complete(self, system: str, user: str) -> str:
        p = self.cfg.provider.lower()
        if p == "openai":
            return self._openai_chat(system, user)
        if p == "anthropic":
            return self._anthropic_messages(system, user)
        if p == "mistral":
            return self._mistral_chat(system, user)
        if p == "generic_http":
            return self._generic_http(system, user)
        raise ValueError(f"Unknown provider: {self.cfg.provider}")

    def _req(self, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> str:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            return resp.read().decode("utf-8")

    def _openai_chat(self, system: str, user: str) -> str:
        url = self.cfg.base_url or "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        raw = self._req(url, headers, body)
        j = json.loads(raw)
        return j["choices"][0]["message"]["content"]

    def _anthropic_messages(self, system: str, user: str) -> str:
        url = self.cfg.base_url or "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        raw = self._req(url, headers, body)
        j = json.loads(raw)
        # content is list of blocks; take first text
        blocks = j.get("content", [])
        for b in blocks:
            if b.get("type") == "text":
                return b.get("text", "")
        return ""

    def _mistral_chat(self, system: str, user: str) -> str:
        url = self.cfg.base_url or "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        raw = self._req(url, headers, body)
        j = json.loads(raw)
        return j["choices"][0]["message"]["content"]

    def _generic_http(self, system: str, user: str) -> str:
        if not self.cfg.base_url:
            raise ValueError("generic_http requires base_url")
        url = self.cfg.base_url
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.cfg.model,
            "system": system,
            "user": user,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        raw = self._req(url, headers, body)
        j = json.loads(raw)
        return j.get("text", raw)
