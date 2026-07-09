#!/usr/bin/env python3
"""Clone a GitHub repository into projects/<repo_name>/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _paths import PROJECTS_DIR, SKILL_ROOT, setup_imports

setup_imports()

from utils.git import clone_github_repo, parse_github_repo_url  # noqa: E402


def _repo_name(repo: str) -> str:
    parsed = parse_github_repo_url(repo)
    if parsed:
        return parsed[1]
    s = repo.strip().rstrip("/")
    if "/" in s and "://" not in s and "github.com" not in s:
        parts = [p for p in s.split("/") if p]
        if len(parts) == 2:
            return parts[1].removesuffix(".git")
    raise ValueError(f"Could not determine repo name from: {repo!r}")


def _clone_url(repo: str) -> str:
    parsed = parse_github_repo_url(repo)
    if parsed:
        owner, name = parsed
        return f"https://github.com/{owner}/{name}.git"
    s = repo.strip().rstrip("/")
    if "/" in s and "://" not in s and "github.com" not in s:
        parts = [p for p in s.split("/") if p]
        if len(parts) == 2:
            owner = parts[0]
            name = parts[1].removesuffix(".git")
            return f"https://github.com/{owner}/{name}.git"
    return repo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone a GitHub repository for code-base onboarding.",
    )
    parser.add_argument(
        "repo",
        help="GitHub URL, git@github.com:owner/repo.git, or owner/repo",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination directory (default: projects/<repo_name> under skill root)",
    )
    args = parser.parse_args()

    name = _repo_name(args.repo)
    dest = Path(args.dest) if args.dest else PROJECTS_DIR / name
    if not dest.is_absolute():
        dest = SKILL_ROOT / dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    clone_github_repo(_clone_url(args.repo), str(dest))
    print(f"Cloned {args.repo} to {dest}")


if __name__ == "__main__":
    main()
