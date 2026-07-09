#!/usr/bin/env python3
"""Run the onboarding workflow for a repository."""

from __future__ import annotations

import argparse
import sys

from _paths import resolve_under_skill, setup_imports

setup_imports()

from main import _run_onboard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate domain knowledge documents via OnboardingWorkflow.",
    )
    parser.add_argument(
        "--repository",
        required=True,
        help="Path to source repository (e.g. projects/myrepo)",
    )
    args = parser.parse_args()

    repository = resolve_under_skill(args.repository)
    if not repository.is_dir():
        print(f"Error: repository path is not a directory: {repository}", file=sys.stderr)
        sys.exit(1)

    _run_onboard(str(repository))


if __name__ == "__main__":
    main()
