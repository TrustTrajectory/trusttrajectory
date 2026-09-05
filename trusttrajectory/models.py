"""Model endpoint configuration and client construction.

All traffic goes through an OpenAI-compatible chat-completions endpoint.  The
default is OpenRouter, which fronts every model family evaluated in the paper
with one API key and one model-name format (``provider/model``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"


REASONING_CHOICES = ("off", "low", "medium", "high")


@dataclass(frozen=True)
class ModelConfig:
    name: str                              # e.g. "google/gemini-3.5-flash"
    display_name: str = ""                 # short label for filenames/figures
    temperature: float = 0.2
    strip_think_tags: bool = False         # strip <think>...</think> (DeepSeek-R1 style)
    reasoning: Optional[str] = None        # None = provider default; "off" or an effort level (OpenRouter `reasoning`)
    http_referer: str = "http://localhost"  # OpenRouter attribution headers
    app_title: str = "TrustTrajectory"

    def __post_init__(self) -> None:
        if self.reasoning is not None and self.reasoning not in REASONING_CHOICES:
            raise ValueError(f"reasoning must be one of {REASONING_CHOICES} or None")

    def label(self) -> str:
        if self.display_name:
            return self.display_name
        return self.name.split("/")[-1]

    def extra_headers(self) -> Dict[str, str]:
        return {"HTTP-Referer": self.http_referer, "X-Title": self.app_title}

    def extra_body(self) -> Optional[Dict[str, Any]]:
        """OpenRouter's unified ``reasoning`` parameter, so every model family
        can be run with the same thinking setting (matched inference)."""
        if self.reasoning is None:
            return None
        if self.reasoning == "off":
            return {"reasoning": {"enabled": False}}
        return {"reasoning": {"effort": self.reasoning}}

    def summary(self) -> Dict[str, Any]:
        return {"name": self.name, "temperature": self.temperature, "reasoning": self.reasoning,
                "strip_think_tags": self.strip_think_tags}


def load_dotenv(path: str = ".env", override: bool = False) -> Dict[str, str]:
    """Minimal ``.env`` loader (KEY=VALUE lines, ``#`` comments); returns what it set."""
    loaded: Dict[str, str] = {}
    if not os.path.exists(path):
        return loaded
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip("'\"")
            if key and (override or key not in os.environ):
                os.environ[key] = value
                loaded[key] = value
    return loaded


def make_client(api_key: Optional[str] = None, base_url: str = OPENROUTER_BASE_URL) -> Any:
    """Build an ``openai.OpenAI`` client; the key defaults to ``$OPENROUTER_API_KEY``."""
    from openai import OpenAI  # imported lazily so analysis-only use needs no API client

    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        load_dotenv()
        key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"No API key: pass api_key=... or set {API_KEY_ENV} (see .env.example)."
        )
    return OpenAI(base_url=base_url, api_key=key)


def model_config_from_env(name: str, display_name: str = "", **kwargs) -> ModelConfig:
    load_dotenv()
    return ModelConfig(
        name=name,
        display_name=display_name,
        http_referer=os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
        app_title=os.environ.get("OPENROUTER_APP_TITLE", "TrustTrajectory"),
        **kwargs,
    )
