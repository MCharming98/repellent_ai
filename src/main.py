import argparse
import os
import re
import sys
from pathlib import Path

import yaml

from workflows.bug_analysis_workflow import BugAnalysisWorkflow
from workflows.onboarding_workflow import OnboardingWorkflow
from utils import parse_github_issue_url


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.yaml"


# `api_key: $VAR_NAME` or `api_key: ${VAR_NAME}` reads from the process environment.
_ENV_VAR_REF = re.compile(
    r"^\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))$"
)


def _resolve_api_key_from_config(raw: object) -> str:
    """Return API key string: literal value, or env lookup for ``$VAR`` / ``${VAR}``."""
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


def _load_config(path: Path) -> dict:
    if not path.is_file():
        print(f"Error: config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        print(f"Error: config must be a YAML mapping: {path}", file=sys.stderr)
        sys.exit(1)
    return data


def _run_onboard(repository: str) -> None:
    cfg = _load_config(_default_config_path())

    api_key = _resolve_api_key_from_config(cfg.get("api_key"))
    if not api_key:
        print(
            "Error: set `api_key` in config.yaml (literal or $VAR / ${VAR}), "
            "or LLM_API_KEY / GOOGLE_API_KEY in the environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    project_name = Path(repository).resolve().name
    domain_knowledge_dir = f"domain_knowledge/{project_name}"

    batch_size = int(cfg.get("file_analysis_batch_size", 10))
    model_name = cfg.get("model_name", "gemini-3-flash-preview")
    model_provider = cfg.get("model_provider", "google_genai")

    onboarding_workflow = OnboardingWorkflow(
        source_repository=repository,
        domain_knowledge=domain_knowledge_dir,
        file_analysis_batch_size=batch_size,
        model=model_name,
        model_provider=model_provider,
        api_key=api_key,
    )
    onboarding_workflow.build_workflow()
    onboarding_workflow.run()


def _load_runtime_settings() -> tuple[str, str, str]:
    """Load model/provider/api key from config + env fallbacks."""
    cfg = _load_config(_default_config_path())
    api_key = _resolve_api_key_from_config(cfg.get("api_key"))
    if not api_key:
        api_key = os.environ.get("LLM_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print(
            "Error: set `api_key` in config.yaml (literal or $VAR / ${VAR}), "
            "or LLM_API_KEY / GOOGLE_API_KEY in the environment.",
            file=sys.stderr,
        )
        sys.exit(1)
    model_name = cfg.get("model_name", "gemini-3-flash-preview")
    model_provider = cfg.get("model_provider", "google_genai")
    return model_name, model_provider, api_key


def _run_analyze(
    issue_url: str,
    source_dir: str | None,
    domain_knowledge_dir: str | None,
    output_dir: str,
) -> None:
    """Run bug analysis for one GitHub issue URL."""
    _, repo, _ = parse_github_issue_url(issue_url)
    source = source_dir or f"projects/{repo}"
    domain_knowledge = domain_knowledge_dir or f"domain_knowledge/{repo}"
    model_name, model_provider, api_key = _load_runtime_settings()

    workflow = BugAnalysisWorkflow(
        issue_url=issue_url,
        output_dir=output_dir,
        source_dir=source,
        domain_knowledge_dir=domain_knowledge,
        model=model_name,
        model_provider=model_provider,
        api_key=api_key,
    )
    workflow.build_workflow()
    workflow.run()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main",
        description="Repellent AI — run subcommands for onboarding and analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main onboard --repository PATH_TO_REPOSITORY\n"
            "  PYTHONPATH=src python src/main.py onboard --repository PATH_TO_REPOSITORY\n"
            "  python main.py analyze --url=https://github.com/owner/repo/issues/123\n"
        ),
    )
    sub = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        help="Subcommand to run (required).",
    )
    onboard = sub.add_parser(
        "onboard",
        help="Run onboarding: file analysis, business analysis, contributor analysis.",
    )
    onboard.add_argument(
        "--repository",
        required=True,
        help="Path to source repository to analyze.",
    )
    analyze = sub.add_parser(
        "analyze",
        help="Run bug analysis (generator + investigator) for a single GitHub issue URL.",
    )
    analyze.add_argument(
        "--url",
        required=True,
        dest="issue_url",
        help="GitHub issue URL (e.g. https://github.com/owner/repo/issues/123).",
    )
    analyze.add_argument(
        "--source-dir",
        default=None,
        help="Source repository path (default: projects/<repo_from_url>).",
    )
    analyze.add_argument(
        "--domain-knowledge",
        default=None,
        dest="domain_knowledge_dir",
        help="Domain knowledge path (default: domain_knowledge/<repo_from_url>).",
    )
    analyze.add_argument(
        "--output-dir",
        default="issues",
        help="Directory to store fetched issue details (default: issues).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    if args.command == "onboard":
        _run_onboard(args.repository)
        return
    if args.command == "analyze":
        _run_analyze(
            issue_url=args.issue_url,
            source_dir=args.source_dir,
            domain_knowledge_dir=args.domain_knowledge_dir,
            output_dir=args.output_dir,
        )
        return


if __name__ == "__main__":
    main()
