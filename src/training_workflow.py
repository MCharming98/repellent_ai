"""Workflow to fetch closed issues from a GitHub repo for training data."""

import argparse
import asyncio
from pathlib import Path, PurePath
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from issue_investigator import IssueInvestigator

from utils import (
    extract_issue_fields,
    fetch_issue_comments,
    get_closed_issues,
    parse_github_repo_url,
    save_issue_details_to_json,
    save_issues_to_json,
)


class TrainingWorkflow:
    """Fetches closed issues from a GitHub repo and stores them to issues/{repo_name}/issues.json."""

    def __init__(
        self,
        github_url: str,
        output_dir: str = "issues",
        max_issues: int = 100,
        agent_workspace: str | None = None,
        model: str = "gemini-3-flash-preview",
        model_provider: str = "google_genai",
        api_key: str | None = None,
    ):
        self.github_url = github_url
        self.output_dir = output_dir
        self.max_issues = max_issues
        self.agent_workspace = agent_workspace
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key

    class State(TypedDict):
        github_url: str
        owner: str
        repo: str
        issues: list
        output_path: str

    def parse_repo(self, state: State):
        parsed = parse_github_repo_url(state["github_url"])
        if not parsed:
            raise ValueError(f"Invalid GitHub URL: {state['github_url']}")
        owner, repo = parsed
        output_path = str(PurePath(self.output_dir) / repo / "issues.json")
        print(f"Training Workflow: Parsed {owner}/{repo}, output: {output_path}")
        return {"owner": owner, "repo": repo, "output_path": output_path}

    def fetch_issues_metadata(self, state: State):
        print(f"Training Workflow: Fetching up to {self.max_issues} closed issues from {state['owner']}/{state['repo']}...")
        issues = get_closed_issues(
            owner=state["owner"],
            repo=state["repo"],
            state="closed",
            per_page=100,
            max_issues=self.max_issues,
        )
        print(f"Training Workflow: Fetched {len(issues)} issues")
        return {"issues": extract_issue_fields(issues)}

    def save_issues_metadata(self, state: State):
        output_path = Path(state["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_issues_to_json(state["issues"], str(output_path))
        print(f"Training Workflow: Saved to {output_path}")
        return {}

    def fetch_and_save_issue_details(self, state: State):
        owner = state["owner"]
        repo = state["repo"]
        issues = state["issues"]
        base_dir = Path(self.output_dir) / repo

        for i, issue in enumerate(issues):
            issue_number = issue.get("number")
            if issue_number is None:
                continue
            issue_dir = base_dir / str(issue_number)
            issue_dir.mkdir(parents=True, exist_ok=True)
            title = issue.get("title")
            body = issue.get("body")
            comments = fetch_issue_comments(owner, repo, issue_number)
            comments_path = PurePath(issue_dir, "issue_details.json")
            save_issue_details_to_json(title, body, comments, str(comments_path))
            print(f"Training Workflow: [{i + 1}/{len(issues)}] Issue #{issue_number}: saved {len(comments)} comments to {comments_path}")

        return {}

    def run_issue_analysis_agents(self, state: State):
        if not self.agent_workspace or not self.api_key:
            print("Training Workflow: Skipping issue analysis (no --workspace or --api-key)")
            return {}

        repo = state["repo"]
        base_dir = Path(self.output_dir) / repo
        if not base_dir.is_dir():
            return {}

        issue_dirs = sorted(
            d
            for d in base_dir.iterdir()
            if d.is_dir() and (d / "issue_details.json").is_file()
        )
        if not issue_dirs:
            print("Training Workflow: No issue folders with issue_details.json; skipping issue analysis")
            return {}

        print(
            f"Training Workflow: Running {len(issue_dirs)} issue analysis agents in parallel "
            f"(workspace={self.agent_workspace})..."
        )

        agents: list[IssueInvestigator] = []
        for i, issue_dir in enumerate(issue_dirs):
            agent = IssueInvestigator(
                str(i),
                str(issue_dir.resolve()),
                str(Path(self.agent_workspace).resolve()),
                self.model,
                self.model_provider,
                self.api_key,
            )
            agent.build_workflow()
            agents.append(agent)

        async def _run_all():
            await asyncio.gather(*[a.run() for a in agents])

        asyncio.run(_run_all())
        print("Training Workflow: Issue analysis agents finished")
        return {}

    def build_workflow(self):
        workflow = StateGraph(self.State)
        workflow.add_node("parse_repo", self.parse_repo)
        workflow.add_node("fetch_issues_metadata", self.fetch_issues_metadata)
        workflow.add_node("save_issues_metadata", self.save_issues_metadata)
        workflow.add_node("fetch_and_save_issue_details", self.fetch_and_save_issue_details)
        workflow.add_node("run_issue_analysis_agents", self.run_issue_analysis_agents)

        workflow.add_edge(START, "parse_repo")
        workflow.add_edge("parse_repo", "fetch_issues_metadata")
        workflow.add_edge("fetch_issues_metadata", "save_issues_metadata")
        workflow.add_edge("save_issues_metadata", "fetch_and_save_issue_details")
        workflow.add_edge("fetch_and_save_issue_details", "run_issue_analysis_agents")
        workflow.add_edge("run_issue_analysis_agents", END)

        self.workflow = workflow.compile()

    def run(self):
        self.build_workflow()
        self.workflow.invoke({"github_url": self.github_url})


def main():
    parser = argparse.ArgumentParser(
        description="Fetch closed issues and comments from a GitHub repo for training data"
    )
    parser.add_argument(
        "--github-url",
        required=True,
        dest="github_url",
        help="GitHub repository URL (e.g. https://github.com/owner/repo)",
    )
    parser.add_argument(
        "--output-dir",
        default="issues",
        help="Output directory for issues (default: issues)",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=100,
        help="Maximum number of issues to fetch (default: 100)",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Agent workspace with analysis markdown; if set with --api-key, runs issue analysis per issue after fetch",
    )
    parser.add_argument(
        "--model-name",
        default="gemini-3-flash-preview",
        help="LLM model for issue analysis (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--model-provider",
        default="google_genai",
        help="LLM provider for issue analysis (default: google_genai)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for issue analysis (required when --workspace is set)",
    )
    args = parser.parse_args()

    if args.workspace and not args.api_key:
        parser.error("--api-key is required when --workspace is set")

    workflow = TrainingWorkflow(
        github_url=args.github_url,
        output_dir=args.output_dir,
        max_issues=args.max_issues,
        agent_workspace=args.workspace,
        model=args.model_name,
        model_provider=args.model_provider,
        api_key=args.api_key,
    )
    workflow.run()


if __name__ == "__main__":
    main()
