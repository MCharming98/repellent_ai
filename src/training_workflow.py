"""Workflow to fetch closed issues from a GitHub repo for training data."""

from pathlib import Path, PurePath
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from utils import extract_issue_fields, get_closed_issues, parse_github_repo_url, save_issues_to_json


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

    def build_workflow(self):
        workflow = StateGraph(self.State)
        workflow.add_node("parse_repo", self.parse_repo)
        workflow.add_node("fetch_issues_metadata", self.fetch_issues_metadata)
        workflow.add_node("save_issues_metadata", self.save_issues_metadata)

        workflow.add_edge(START, "parse_repo")
        workflow.add_edge("parse_repo", "fetch_issues_metadata")
        workflow.add_edge("fetch_issues_metadata", "save_issues_metadata")
        workflow.add_edge("save_issues_metadata", END)

        self.workflow = workflow.compile()

    def run(self):
        self.build_workflow()
        self.workflow.invoke({"github_url": self.github_url})
