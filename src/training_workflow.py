"""Workflow to fetch closed issues from a GitHub repo for training data."""

import argparse
from pathlib import Path, PurePath
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

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

    def __init__(self, github_url: str, output_dir: str = "issues", max_issues: int = 100):
        self.github_url = github_url
        self.output_dir = output_dir
        self.max_issues = max_issues

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

    def build_workflow(self):
        workflow = StateGraph(self.State)
        workflow.add_node("parse_repo", self.parse_repo)
        workflow.add_node("fetch_issues_metadata", self.fetch_issues_metadata)
        workflow.add_node("save_issues_metadata", self.save_issues_metadata)
        workflow.add_node("fetch_and_save_issue_details", self.fetch_and_save_issue_details)

        workflow.add_edge(START, "parse_repo")
        workflow.add_edge("parse_repo", "fetch_issues_metadata")
        workflow.add_edge("fetch_issues_metadata", "save_issues_metadata")
        workflow.add_edge("save_issues_metadata", "fetch_and_save_issue_details")
        workflow.add_edge("fetch_and_save_issue_details", END)

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
    args = parser.parse_args()

    workflow = TrainingWorkflow(
        github_url=args.github_url,
        output_dir=args.output_dir,
        max_issues=args.max_issues,
    )
    workflow.run()


if __name__ == "__main__":
    main()
