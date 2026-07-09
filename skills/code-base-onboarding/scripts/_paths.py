"""Shared paths for the self-contained code-base-onboarding skill."""

from __future__ import annotations

import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = SKILL_ROOT / "projects"
DOMAIN_KNOWLEDGE_DIR = SKILL_ROOT / "domain_knowledge"
CONFIG_PATH = SKILL_ROOT / "config.yaml"


def setup_imports() -> None:
    scripts = str(SCRIPTS_DIR)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def resolve_under_skill(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return SKILL_ROOT / candidate
