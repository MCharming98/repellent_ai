import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import urllib.error
import urllib.request
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from agents.hypothesis_generator import HypothesisGenerator
from agents.hypothesis_investigator import HypothesisInvestigator
from utils.config import default_config_path, load_config, require_github_token
from utils import fetch_issue_comments, parse_github_issue_url, save_issue_details_to_json


class BugAnalysisWorkflow:
    """Fetch one issue by URL, generate diagnosis, then run hypothesis investigation."""

    class State(TypedDict):
        issue_url: str
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
        issue_url: str,
        output_dir: str,
        source_dir: str,
        domain_knowledge_dir: str,
        model: str,
        model_provider: str,
        api_key: str,
        github_token: str = "",
    ) -> None:
        self.issue_url = issue_url
        self.output_dir = output_dir
        self.source_dir = str(Path(source_dir).resolve())
        self.domain_knowledge_dir = str(Path(domain_knowledge_dir).resolve())
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key
        self.github_token = github_token
        self.workflow = None

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
        agent = HypothesisGenerator(
            issue_dir=str(issue_dir),
            domain_knowledge=state["domain_knowledge_dir"],
            model=state["model"],
            model_provider=state["model_provider"],
            api_key=state["api_key"],
        )
        agent.build_workflow()
        asyncio.run(agent.run())
        print("Bug Analysis Workflow: hypothesis generator finished")
        return {}

    def run_hypothesis_investigator(self, state: State) -> dict:
        print("Bug Analysis Workflow: running hypothesis investigator (reads diagnosis.md)")
        investigator = HypothesisInvestigator(
            issue_dir=state["issue_dir"],
            source_dir=state["source_dir"],
            domain_knowledge_dir=state["domain_knowledge_dir"],
            model=state["model"],
            model_provider=state["model_provider"],
            api_key=state["api_key"],
        )
        investigator.build_workflow()
        asyncio.run(investigator.run())
        print("Bug Analysis Workflow: hypothesis investigation finished")
        return {}

    def build_workflow(self) -> None:
        workflow = StateGraph(self.State)
        workflow.add_node("parse_issue_url", self.parse_issue_url)
        workflow.add_node("fetch_issue_metadata", self.fetch_issue_metadata)
        workflow.add_node("run_hypothesis_generator", self.run_hypothesis_generator)
        workflow.add_node("run_hypothesis_investigator", self.run_hypothesis_investigator)

        workflow.add_edge(START, "parse_issue_url")
        workflow.add_edge("parse_issue_url", "fetch_issue_metadata")
        workflow.add_edge("fetch_issue_metadata", "run_hypothesis_generator")
        workflow.add_edge("run_hypothesis_generator", "run_hypothesis_investigator")
        workflow.add_edge("run_hypothesis_investigator", END)
        self.workflow = workflow.compile()

    def run(self) -> None:
        if self.workflow is None:
            self.build_workflow()
        self.workflow.invoke(
            {
                "issue_url": self.issue_url,
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
        required=True,
        help="GitHub issue URL (e.g. https://github.com/owner/repo/issues/123)",
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

    github_token = require_github_token(load_config(default_config_path()))

    workflow = BugAnalysisWorkflow(
        issue_url=args.issue_url,
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        domain_knowledge_dir=args.domain_knowledge_dir,
        model=args.model,
        model_provider=args.model_provider,
        api_key=args.api_key,
        github_token=github_token,
    )
    workflow.run()


if __name__ == "__main__":
    main()
