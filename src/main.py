import argparse
import sys
from pathlib import Path

from utils.config import (
    default_config_path,
    load_config,
    load_runtime_settings,
    require_api_key,
)
from workflows.bug_analysis_workflow import BugAnalysisWorkflow
from workflows.onboarding_workflow import OnboardingWorkflow
from utils import parse_github_issue_url


def _run_onboard(repository: str) -> None:
    cfg = load_config(default_config_path())
    api_key = require_api_key(cfg)
    model_name = str(cfg.get("model_name", "gemini-3-flash-preview"))
    model_provider = str(cfg.get("model_provider", "google_genai"))

    project_name = Path(repository).resolve().name
    domain_knowledge_dir = f"domain_knowledge/{project_name}"

    batch_size = int(cfg.get("file_analysis_batch_size", 10))

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


def _run_analyze(
    issue_url: str | None,
    issue_path: str | None,
    source_dir: str | None,
    domain_knowledge_dir: str | None,
    output_dir: str,
) -> None:
    """Run bug analysis from a GitHub issue URL or local issue directory path."""
    cfg = load_config(default_config_path())
    model_name, model_provider, api_key, github_token = load_runtime_settings(cfg)

    if issue_path:
        issue_dir = Path(issue_path).resolve()
        if not issue_dir.is_dir():
            print(f"Error: issue path is not a directory: {issue_dir}", file=sys.stderr)
            sys.exit(1)
        if not (issue_dir / "issue_details.json").is_file():
            print(f"Error: missing issue_details.json in: {issue_dir}", file=sys.stderr)
            sys.exit(1)
        repo = issue_dir.parent.name
        source = source_dir or f"projects/{repo}"
        domain_knowledge = domain_knowledge_dir or f"domain_knowledge/{repo}"
    else:
        if not issue_url:
            print("Error: one of --url or --path is required.", file=sys.stderr)
            sys.exit(1)
        _, repo, _ = parse_github_issue_url(issue_url)
        source = source_dir or f"projects/{repo}"
        domain_knowledge = domain_knowledge_dir or f"domain_knowledge/{repo}"

    workflow = BugAnalysisWorkflow(
        issue_url=issue_url or "",
        issue_path=issue_path or "",
        output_dir=output_dir,
        source_dir=source,
        domain_knowledge_dir=domain_knowledge,
        model=model_name,
        model_provider=model_provider,
        api_key=api_key,
        github_token=github_token,
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
        help="Run bug analysis from a GitHub issue URL or a local issue directory path.",
    )
    issue_src = analyze.add_mutually_exclusive_group(required=True)
    issue_src.add_argument(
        "--url",
        dest="issue_url",
        help="GitHub issue URL (e.g. https://github.com/owner/repo/issues/123).",
    )
    issue_src.add_argument(
        "--path",
        dest="issue_path",
        help="Local issue directory path (e.g. issues/AntennaPod/8311).",
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
            issue_path=args.issue_path,
            source_dir=args.source_dir,
            domain_knowledge_dir=args.domain_knowledge_dir,
            output_dir=args.output_dir,
        )
        return


if __name__ == "__main__":
    main()
