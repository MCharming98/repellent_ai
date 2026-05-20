"""Load repository-root ``config.yaml``. Values may use ``$VAR`` / ``${VAR}`` only; no other env fallbacks."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

_ENV_VAR_REF = re.compile(
    r"^\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))$"
)


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "config.yaml"


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        print(f"Error: config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        print(f"Error: config must be a YAML mapping: {path}", file=sys.stderr)
        sys.exit(1)
    return data


def resolve_config_value(raw: object) -> str:
    """Literal string, or env lookup only when the value is exactly ``$VAR`` / ``${VAR}``."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        return str(raw).strip()
    s = raw.strip()
    if not s:
        return ""
    m = _ENV_VAR_REF.match(s)
    if m:
        name = m.group(1) or m.group(2)
        return os.environ.get(name, "")
    return s


def require_api_key(cfg: dict[str, Any]) -> str:
    key = resolve_config_value(cfg.get("api_key"))
    if not key:
        print(
            "Error: set non-empty `api_key` in config.yaml (literal or $VAR / ${VAR}).",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def require_github_token(cfg: dict[str, Any]) -> str:
    token = resolve_config_value(cfg.get("github_token"))
    if not token:
        print(
            "Error: set non-empty `github_token` in config.yaml (literal or $VAR / ${VAR}).",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def get_investigation_max_token_usage(cfg: dict[str, Any]) -> int:
    """Cumulative token budget for hypothesis investigation before forced convergence."""
    raw = cfg.get("investigation_max_token_usage", 5_000_000)
    try:
        n = int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"config `investigation_max_token_usage` must be an integer, got {raw!r}"
        ) from e
    if n <= 0:
        raise ValueError(
            f"config `investigation_max_token_usage` must be positive, got {n}"
        )
    return n


def get_model_context_window(cfg: dict[str, Any]) -> int:
    """Maximum input context length for the configured model (``model_context_window`` in YAML; required)."""
    if "model_context_window" not in cfg or cfg.get("model_context_window") is None:
        raise ValueError("config.yaml must set `model_context_window` to a non-null value")
    raw = cfg["model_context_window"]
    try:
        n = int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"config `model_context_window` must be an integer, got {raw!r}"
        ) from e
    if n <= 0:
        raise ValueError(f"config `model_context_window` must be positive, got {n}")
    return n


def load_runtime_settings(cfg: dict[str, Any]) -> tuple[str, str, str, str]:
    """Model settings plus required ``api_key`` and ``github_token`` from *cfg*."""
    api_key = require_api_key(cfg)
    github_token = require_github_token(cfg)
    model_name = str(cfg.get("model_name", "gemini-3-flash-preview"))
    model_provider = str(cfg.get("model_provider", "google_genai"))
    return model_name, model_provider, api_key, github_token
