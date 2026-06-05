import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import time
import urllib.error
import urllib.request
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from agents.graph_hypothesis_investigator import GraphHypothesisInvestigator
from agents.hypothesis_generator import HypothesisGenerator
from utils.config import (
    default_config_path,
    get_investigation_max_token_usage,
    load_config,
    require_github_token,
)
from utils import (
    fetch_issue_comments,
    get_issue_labels,
    parse_github_issue_url,
    save_issue_details_to_json,
)

HYPOTHESIS_STAGE_WAIT_SECONDS = 60


class BugAnalysisWorkflow:
    """Fetch one issue by URL, generate diagnosis, then run hypothesis investigation."""

    class State(TypedDict):
        issue_url: str
        issue_path: str
        owner: str
        repo: str
        issue_number: int
        issue_dir: str
        source_dir: str
        domain_knowledge_dir: str
        model: str
        model_provider: str
        api_key: str
        github_token: str

    def __init__(
        self,
        issue_url: str | None,
        issue_path: str | None,
        output_dir: str,
        source_dir: str,
        domain_knowledge_dir: str,
        model: str,
        model_provider: str,
        api_key: str,
        max_token_usage: int,
        github_token: str = "",
    ) -> None:
        self.issue_url = issue_url or ""
        self.issue_path = issue_path or ""
        self.output_dir = output_dir
        self.source_dir = str(Path(source_dir).resolve())
        self.domain_knowledge_dir = str(Path(domain_knowledge_dir).resolve())
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key
        self.github_token = github_token
        self.max_token_usage = max_token_usage
        self.workflow = None

    def _entrypoint(self, state: State) -> str:
        return "load_issue_path" if state.get("issue_path") else "parse_issue_url"

    def load_issue_path(self, state: State) -> dict:
        issue_dir = Path(state["issue_path"]).resolve()
        if not issue_dir.is_dir():
            raise ValueError(f"Issue path is not a directory: {issue_dir}")
        details_path = issue_dir / "issue_details.json"
        if not details_path.is_file():
            raise ValueError(f"Missing issue_details.json in {issue_dir}")
        print(f"Bug Analysis Workflow: using local issue path {issue_dir}")
        return {"issue_dir": str(issue_dir)}

    def parse_issue_url(self, state: State) -> dict:
        owner, repo, issue_number = parse_github_issue_url(state["issue_url"])
        issue_dir = Path(self.output_dir) / repo / str(issue_number)
        print(
            f"Bug Analysis Workflow: parsed issue {owner}/{repo}#{issue_number}, "
            f"output dir: {issue_dir}"
        )
        return {
            "owner": owner,
            "repo": repo,
            "issue_number": issue_number,
            "issue_dir": str(issue_dir.resolve()),
        }

    def fetch_issue_metadata(self, state: State) -> dict:
        owner = state["owner"]
        repo = state["repo"]
        issue_number = state["issue_number"]
        issue_api = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Repellent-AI",
        }
        token = state["github_token"]
        if token:
            headers["Authorization"] = f"token {token}"
        req = urllib.request.Request(issue_api, headers=headers)
        try:
            with urllib.request.urlopen(req) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Issue metadata request failed: {e.code} {e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Issue metadata request failed: {e.reason}") from e

        title = payload.get("title", "")
        body = payload.get("body", "")
        comments = fetch_issue_comments(
            owner, repo, issue_number, token=state["github_token"] or None
        )
        issue_dir = Path(state["issue_dir"])
        issue_dir.mkdir(parents=True, exist_ok=True)
        details_path = issue_dir / "issue_details.json"
        save_issue_details_to_json(title, body, comments, str(details_path))
        print(
            "Bug Analysis Workflow: saved issue details with "
            f"{len(comments)} comments to {details_path}"
        )
        return {}

    def run_hypothesis_generator(self, state: State) -> dict:
        issue_dir = Path(state["issue_dir"])
        print(f"Bug Analysis Workflow: running hypothesis generator for {issue_dir}")
        try:
            repo_labels = get_issue_labels(state["source_dir"])
            print(
                f"Bug Analysis Workflow: loaded {len(repo_labels)} repo label definition(s)"
            )
        except (ValueError, RuntimeError) as e:
            print(f"Bug Analysis Workflow: could not load repo labels ({e}); continuing")
            repo_labels = {}
        agent = HypothesisGenerator(
            issue_dir=str(issue_dir),
            domain_knowledge=state["domain_knowledge_dir"],
            model=state["model"],
            model_provider=state["model_provider"],
            api_key=state["api_key"],
            issue_labels=repo_labels,
        )
        agent.build_workflow()
        asyncio.run(agent.run())
        print("Bug Analysis Workflow: hypothesis generator finished")
        return {}

    def run_hypothesis_investigator(self, state: State) -> dict:
        print("Bug Analysis Workflow: running graph hypothesis investigator")
        investigator = GraphHypothesisInvestigator(
            issue_dir=state["issue_dir"],
            source_dir=state["source_dir"],
            domain_knowledge_dir=state["domain_knowledge_dir"],
            model=state["model"],
            model_provider=state["model_provider"],
            api_key=state["api_key"],
            max_token_usage=self.max_token_usage,
        )
        investigator.build_workflow()
        asyncio.run(investigator.run())
        print("Bug Analysis Workflow: hypothesis investigation finished")
        return {}

    def wait_before_hypothesis_investigator(self, state: State) -> dict:
        print(
            "Bug Analysis Workflow: waiting "
            f"{HYPOTHESIS_STAGE_WAIT_SECONDS}s before hypothesis investigator"
        )
        time.sleep(HYPOTHESIS_STAGE_WAIT_SECONDS)
        return {}

    def build_workflow(self) -> None:
        workflow = StateGraph(self.State)
        workflow.add_node("load_issue_path", self.load_issue_path)
        workflow.add_node("parse_issue_url", self.parse_issue_url)
        workflow.add_node("fetch_issue_metadata", self.fetch_issue_metadata)
        workflow.add_node("run_hypothesis_generator", self.run_hypothesis_generator)
        workflow.add_node(
            "wait_before_hypothesis_investigator",
            self.wait_before_hypothesis_investigator,
        )
        workflow.add_node("run_hypothesis_investigator", self.run_hypothesis_investigator)

        workflow.add_conditional_edges(
            START,
            self._entrypoint,
            {
                "load_issue_path": "load_issue_path",
                "parse_issue_url": "parse_issue_url",
            },
        )
        workflow.add_edge("load_issue_path", "run_hypothesis_generator")
        workflow.add_edge("parse_issue_url", "fetch_issue_metadata")
        workflow.add_edge("fetch_issue_metadata", "run_hypothesis_generator")
        workflow.add_edge(
            "run_hypothesis_generator", "wait_before_hypothesis_investigator"
        )
        workflow.add_edge(
            "wait_before_hypothesis_investigator", "run_hypothesis_investigator"
        )
        workflow.add_edge("run_hypothesis_investigator", END)
        self.workflow = workflow.compile()

    def run(self) -> None:
        if self.workflow is None:
            self.build_workflow()
        self.workflow.invoke(
            {
                "issue_url": self.issue_url,
                "issue_path": self.issue_path,
                "source_dir": self.source_dir,
                "domain_knowledge_dir": self.domain_knowledge_dir,
                "model": self.model,
                "model_provider": self.model_provider,
                "api_key": self.api_key,
                "github_token": self.github_token,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bug analysis workflow for one GitHub issue URL"
    )
    parser.add_argument(
        "--issue-url",
        dest="issue_url",
        required=False,
        help="GitHub issue URL (e.g. https://github.com/owner/repo/issues/123)",
    )
    parser.add_argument(
        "--issue-path",
        dest="issue_path",
        required=False,
        help="Local issue directory path (e.g. issues/AntennaPod/8311)",
    )
    parser.add_argument(
        "--output-dir",
        default="issues",
        help="Base directory to store fetched issue details (default: issues)",
    )
    parser.add_argument(
        "--source-dir",
        dest="source_dir",
        required=True,
        help="Path to source repository used by investigation tools",
    )
    parser.add_argument(
        "--domain-knowledge",
        required=True,
        dest="domain_knowledge_dir",
        help="Domain knowledge directory containing file_analysis.md",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="LLM model (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--model-provider",
        default="google_genai",
        dest="model_provider",
        help="LLM provider (default: google_genai)",
    )
    parser.add_argument("--api-key", required=True, help="API key for LLM provider")
    args = parser.parse_args()

    if not args.issue_url and not args.issue_path:
        parser.error("one of --issue-url or --issue-path is required")

    cfg = load_config(default_config_path())
    github_token = require_github_token(cfg)
    max_token_usage = get_investigation_max_token_usage(cfg)

    workflow = BugAnalysisWorkflow(
        issue_url=args.issue_url or "",
        issue_path=args.issue_path,
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        domain_knowledge_dir=args.domain_knowledge_dir,
        model=args.model,
        model_provider=args.model_provider,
        api_key=args.api_key,
        max_token_usage=max_token_usage,
        github_token=github_token,
    )
    workflow.run()


if __name__ == "__main__":
    main()
