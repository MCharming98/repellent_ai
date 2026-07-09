"""Onboarding CLI for the self-contained code-base-onboarding skill."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.config import (
    default_config_path,
    get_model_context_window,
    load_config,
    require_api_key,
    skill_root,
)
from workflows.onboarding_workflow import OnboardingWorkflow


def _run_onboard(repository: str) -> None:
    cfg = load_config(default_config_path())
    api_key = require_api_key(cfg)
    model_name = str(cfg.get("model_name", "gemini-3-flash-preview"))
    model_provider = str(cfg.get("model_provider", "google_genai"))

    repo_path = Path(repository)
    if not repo_path.is_absolute():
        repo_path = skill_root() / repository
    repo_path = repo_path.resolve()
    if not repo_path.is_dir():
        print(f"Error: repository path is not a directory: {repo_path}", file=sys.stderr)
        sys.exit(1)

    project_name = repo_path.name
    domain_knowledge_dir = skill_root() / "domain_knowledge" / project_name
    domain_knowledge_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(cfg.get("file_analysis_batch_size", 10))
    model_context_window = get_model_context_window(cfg)

    onboarding_workflow = OnboardingWorkflow(
        source_repository=str(repo_path),
        domain_knowledge=str(domain_knowledge_dir),
        file_analysis_batch_size=batch_size,
        model=model_name,
        model_provider=model_provider,
        api_key=api_key,
        model_context_window=model_context_window,
    )
    onboarding_workflow.build_workflow()
    onboarding_workflow.run()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onboard",
        description="Run onboarding: file analysis, business analysis, contributor analysis.",
    )
    parser.add_argument(
        "--repository",
        required=True,
        help="Path to source repository (e.g. projects/myrepo).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _run_onboard(args.repository)


if __name__ == "__main__":
    main()
