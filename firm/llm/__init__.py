"""LLM backends behind one small interface. Agents ask for JSON matching a schema.

Backends (env FIRM_LLM_BACKEND):
  ollama  local model via Ollama (default). Model from FIRM_LLM_MODEL.
  off     no LLM: agents skip LLM steps and report "LLM offline".

Every call is logged to the `llm_calls` table so outputs can be audited later.
LLMs never place orders; nothing here touches the broker.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from firm.activity import connect, now_iso

DEFAULT_MODEL = "qwen3:8b"
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    agent TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT,
    ok INTEGER NOT NULL,
    ms INTEGER
);
"""


class LLMUnavailable(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, model: str | None = None, url: str = OLLAMA_URL, timeout: float = 180.0) -> None:
        self.model = model or os.getenv("FIRM_LLM_MODEL", DEFAULT_MODEL)
        self.url, self.timeout = url.rstrip("/"), timeout

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=3)
            names = {m["name"] for m in r.json().get("models", [])}
            return self.model in names or f"{self.model}:latest" in names
        except Exception:
            return False

    def json(self, system: str, prompt: str, schema: dict, max_tokens: int = 400) -> dict:
        body = {
            "model": self.model, "stream": False, "format": schema, "keep_alive": "5m",
            "think": False,  # reasoning models: answer directly, keeps it fast
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096},
        }
        try:
            r = requests.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
        except requests.RequestException as e:
            raise LLMUnavailable(f"Ollama not reachable: {e}") from e
        if r.status_code == 400 and "think" in r.text:  # model without a thinking switch
            body.pop("think")
            r = requests.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
        if r.status_code != 200:
            raise LLMUnavailable(f"Ollama error {r.status_code}: {r.text[:200]}")
        return json.loads(r.json()["message"]["content"])


class OfflineLLM:
    name = "off"

    def available(self) -> bool:
        return False

    def json(self, system: str, prompt: str, schema: dict, max_tokens: int = 400) -> dict:
        raise LLMUnavailable("LLM backend is off")


def get_llm():
    try:                                    # the owner's "AI model" switch on the Control page
        from firm.control import Control

        if not Control().get_bool("llm.enabled"):
            return OfflineLLM()
    except Exception:
        pass
    backend = os.getenv("FIRM_LLM_BACKEND", "ollama").lower()
    return OllamaLLM() if backend == "ollama" else OfflineLLM()


class LoggedLLM:
    """Wraps a backend: logs every call with the agent's name."""

    def __init__(self, llm, db_path: Path | str) -> None:
        self.llm = llm
        self.conn = connect(db_path)
        self.conn.executescript(SCHEMA)

    @property
    def name(self) -> str:
        return self.llm.name

    def available(self) -> bool:
        return self.llm.available()

    def json(self, agent: str, system: str, prompt: str, schema: dict, max_tokens: int = 400) -> dict:
        t0 = time.time()
        out, ok = None, False
        try:
            result = self.llm.json(system, prompt, schema, max_tokens)
            out, ok = json.dumps(result, ensure_ascii=False), True
            return result
        except Exception as e:
            out = f"{type(e).__name__}: {e}"
            raise
        finally:
            self.conn.execute(
                "INSERT INTO llm_calls (ts, agent, model, prompt, response, ok, ms) VALUES (?,?,?,?,?,?,?)",
                (now_iso(), agent, self.llm.name, prompt[:4000], (out or "")[:4000], int(ok),
                 int((time.time() - t0) * 1000)))
            self.conn.commit()
