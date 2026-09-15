"""Minimal LLM client with three backends: ollama (local), anthropic, openai-compatible.

    llm = LLM("ollama", "llama3.1:8b")
    llm = LLM("anthropic", "claude-sonnet-4-6")        # needs ANTHROPIC_API_KEY
    llm = LLM("openai", "gpt-4o-mini", base_url=...)   # needs OPENAI_API_KEY (or any OpenAI-compatible server)

`llm(prompt, system=...)` returns a string. `llm.json(prompt)` returns parsed JSON.
"""
from __future__ import annotations

import json
import os
import re

import requests

DEFAULT_SYSTEM = (
    "You are a careful research assistant summarizing scientific papers. "
    "Be concrete, cite section names, never invent numbers. "
    "If the text does not contain the information, say so briefly."
)


_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>\s*", re.S | re.I)


def strip_reasoning(text: str) -> str:
    """Drop <think>…</think> blocks emitted by reasoning models (qwen3, deepseek-r1, …).

    Their traces routinely contain braces and code fences, so `json()` would otherwise
    parse the model's scratchpad instead of its answer. An unclosed block (hit the token
    limit mid-thought) leaves nothing usable after the tag, so it is dropped too.
    """
    text = _THINK_RE.sub("", text or "")
    if (m := re.search(r"<(?:think|thinking|reasoning)>", text, re.I)):
        text = text[: m.start()]
    return text


def _json_slice(raw: str) -> str:
    """Pull the JSON value out of a reply and repair the damage small models do."""
    s = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    # widest balanced-looking span, so prose on either side is dropped
    starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
    ends = [i for i in (s.rfind("}"), s.rfind("]")) if i != -1]
    if starts and ends and max(ends) > min(starts):
        s = s[min(starts): max(ends) + 1]
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)              # trailing comma before } or ]
    s = re.sub(r"'([^'\"\n]*)'(\s*:)", r'"\1"\2', s)  # 'key': -> "key":
    s = re.sub(r"(:\s*)'([^'\n]*)'", r'\1"\2"', s)    # : 'value' -> : "value"
    return s


class LLM:
    def __init__(self, backend: str = "ollama", model: str = "llama3.1:8b",
                 base_url: str | None = None, temperature: float = 0.2,
                 max_tokens: int = 1500, ctx_chars: int = 48_000, think: bool = False):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ctx_chars = ctx_chars       # crude context guard on input text
        # Hybrid reasoning models (qwen3, deepseek-r1) spend the whole num_predict budget
        # on a <think> trace and get truncated before the answer, so every stage comes back
        # empty. None of these stages need visible reasoning — ask for the answer directly.
        self.think = think
        self._think_supported = True
        self.base_url = base_url or {
            "ollama": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            "openai": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "anthropic": "https://api.anthropic.com/v1",
        }[backend]
        self.calls = 0

    # ------------------------------------------------------------------ public
    def __call__(self, prompt: str, system: str = DEFAULT_SYSTEM) -> str:
        prompt = prompt[: self.ctx_chars]
        self.calls += 1
        return strip_reasoning(getattr(self, f"_{self.backend}")(prompt, system)).strip()

    def json(self, prompt: str, system: str = DEFAULT_SYSTEM, retries: int = 1):
        """Ask for JSON and parse it, repairing and re-asking before giving up.

        Small local models emit *almost* JSON often enough that one bad reply used to
        abort a whole run at stage 7 of 9. Repair the usual damage (fences, trailing
        commas, single quotes), and if that still fails, show the model its own broken
        output and ask again before raising.
        """
        ask = prompt + "\n\nReturn ONLY valid JSON, no prose, no code fences."
        for attempt in range(retries + 1):
            raw = self(ask, system)
            try:
                return json.loads(_json_slice(raw))
            except ValueError as e:
                if attempt >= retries:
                    raise ValueError(f"model did not return valid JSON: {e}\n---\n{raw[:400]}") from e
                ask = (f"{prompt}\n\nYour previous reply could not be parsed as JSON ({e}). "
                       "Reply with the JSON object ONLY: double-quoted keys and strings, "
                       "no trailing commas, no code fences, no commentary.")

    def check(self) -> str:
        """Ping the backend; returns a short status line."""
        try:
            if self.backend == "ollama":
                r = requests.get(f"{self.base_url}/api/tags", timeout=5)
                names = [m["name"] for m in r.json().get("models", [])]
                ok = self.model in names or any(n.startswith(self.model) for n in names)
                return f"ollama @ {self.base_url}: {'✓ ' + self.model if ok else '✗ model not pulled; run: ollama pull ' + self.model}"
            out = self("Reply with the single word OK.")
            return f"{self.backend}/{self.model}: ✓ ({out[:20]})"
        except Exception as e:
            return f"{self.backend}: ✗ {e}"

    # ---------------------------------------------------------------- backends
    def _ollama(self, prompt, system):
        payload = {
            "model": self.model, "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens,
                        "num_ctx": 16384},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        }
        if self._think_supported:
            payload["think"] = self.think
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=600)
        if r.status_code == 400 and self._think_supported and "think" in r.text.lower():
            self._think_supported = False       # model has no thinking mode; ask again plainly
            payload.pop("think")
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _anthropic(self, prompt, system):
        key = os.environ["ANTHROPIC_API_KEY"]
        r = requests.post(f"{self.base_url}/messages", headers={
            "x-api-key": key, "anthropic-version": "2023-06-01",
            "content-type": "application/json"},
            json={"model": self.model, "max_tokens": self.max_tokens, "system": system,
                  "temperature": self.temperature,
                  "messages": [{"role": "user", "content": prompt}]}, timeout=300)
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])

    def _openai(self, prompt, system):
        key = os.getenv("OPENAI_API_KEY", "none")
        r = requests.post(f"{self.base_url}/chat/completions", headers={
            "Authorization": f"Bearer {key}", "content-type": "application/json"},
            json={"model": self.model, "max_tokens": self.max_tokens,
                  "temperature": self.temperature,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": prompt}]}, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
