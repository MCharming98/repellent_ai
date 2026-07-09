#!/usr/bin/env python3
"""Estimate LLM read token usage for onboarding a source repository."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from _paths import CONFIG_PATH, SKILL_ROOT, resolve_under_skill, setup_imports

setup_imports()

from utils import read_file  # noqa: E402
from utils.config import (  # noqa: E402
    get_model_context_window,
    load_config,
    require_api_key,
    resolve_config_value,
)
from utils.files import list_source_files_recursive  # noqa: E402
from utils.text import estimate_token_count  # noqa: E402


def estimate_repository_tokens(repository: Path, cfg: dict) -> dict:
    require_api_key(cfg)
    model_name = str(cfg.get("model_name", "gemini-3-flash-preview"))
    model_provider = str(cfg.get("model_provider", "google_genai"))
    model_context_window = get_model_context_window(cfg)

    read_dir = str(repository.resolve())
    source_files = list_source_files_recursive(read_dir)
    token_count_map: dict[str, int] = {}
    skipped_oversized = 0
    total_files = len(source_files)
    print(f"Estimating file tokens for {total_files} files...")

    for i, rel_path in enumerate(source_files, start=1):
        if rel_path.startswith("Error:"):
            continue
        full_path = os.path.join(read_dir, rel_path)
        content = read_file(full_path)
        if content.startswith("Error:"):
            continue
        n = estimate_token_count(
            content,
            model_provider,
            model_context_window,
            model_name=model_name,
        )
        if n > model_context_window:
            skipped_oversized += 1
            continue
        token_count_map[rel_path] = n
        if i % 25 == 0 or i == total_files:
            print(f"Estimating file tokens: {i}/{total_files} files processed")

    print(f"Estimating file tokens completed ({len(token_count_map)} files counted)")

    total_read_tokens = sum(token_count_map.values())
    return {
        "repository_path": read_dir,
        "source_files": len(token_count_map),
        "total_files_scanned": total_files,
        "skipped_oversized": skipped_oversized,
        "estimated_read_tokens": total_read_tokens,
        "rough_total_budget_low": int(total_read_tokens * 1.5),
        "rough_total_budget_high": total_read_tokens * 2,
        "model_name": model_name,
        "model_provider": model_provider,
        "model_context_window": model_context_window,
    }


def print_summary(summary: dict) -> None:
    print()
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Repository path | `{summary['repository_path']}` |")
    print(f"| Source files | {summary['source_files']} |")
    if summary["skipped_oversized"]:
        print(
            f"| Skipped (exceeds context window) | {summary['skipped_oversized']} |"
        )
    print(f"| Estimated read tokens | {summary['estimated_read_tokens']:,} |")
    print(
        f"| Rough total budget (1.5-2x read) | "
        f"{summary['rough_total_budget_low']:,}-{summary['rough_total_budget_high']:,} |"
    )
    print(
        f"| Model | `{summary['model_name']}` (`{summary['model_provider']}`) |"
    )
    print(f"| Context window | `{summary['model_context_window']:,}` |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate read token usage before running onboarding.",
    )
    parser.add_argument(
        "repository",
        help="Path to the source repository (e.g. projects/myrepo)",
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help="Path to config.yaml (default: skill-root config.yaml)",
    )
    args = parser.parse_args()

    repository = resolve_under_skill(args.repository)
    if not repository.is_dir():
        print(f"Error: repository path is not a directory: {repository}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(Path(args.config))
    if not resolve_config_value(cfg.get("api_key")):
        print(
            "Error: set non-empty `api_key` in config.yaml (literal or $VAR / ${VAR}).",
            file=sys.stderr,
        )
        sys.exit(1)

    summary = estimate_repository_tokens(repository, cfg)
    print_summary(summary)


if __name__ == "__main__":
    main()
