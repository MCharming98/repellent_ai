import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from agents.hypothesis_generator import HypothesisGenerator
from agents.hypothesis_investigator import HypothesisInvestigator


class BugAnalysisWorkflow:
    """Generate hypotheses for one issue, then investigate each in parallel."""

    class State(TypedDict):
        issue_dir: str
        source_dir: str
        agent_workspace_dir: str
        model: str
        model_provider: str
        api_key: str
        hypothesis_files: list[str]

    def __init__(
        self,
        issue_dir: str,
        source_dir: str,
        agent_workspace_dir: str,
        model: str,
        model_provider: str,
        api_key: str,
    ) -> None:
        self.issue_dir = str(Path(issue_dir).resolve())
        self.source_dir = str(Path(source_dir).resolve())
        self.agent_workspace_dir = str(Path(agent_workspace_dir).resolve())
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key
        self.workflow = None

    def run_hypothesis_generator(self, state: State) -> dict:
        issue_dir = Path(state["issue_dir"])
        print(f"Bug Analysis Workflow: running hypothesis generator for {issue_dir}")
        agent = HypothesisGenerator(
            id="0",
            issue_dir=str(issue_dir),
            agent_workspace=state["agent_workspace_dir"],
            model=state["model"],
            model_provider=state["model_provider"],
            api_key=state["api_key"],
        )
        agent.build_workflow()
        asyncio.run(agent.run())
        print("Bug Analysis Workflow: hypothesis generator finished")
        return {}

    def collect_hypothesis_files(self, state: State) -> dict:
        issue_dir = Path(state["issue_dir"])

        def _hyp_idx(p: Path) -> int:
            stem = p.stem
            try:
                return int(stem.split("_")[-1])
            except ValueError:
                return 10**9

        files = sorted(issue_dir.glob("hypothesis_*.md"), key=_hyp_idx)
        hypothesis_files = [str(p.resolve()) for p in files]
        print(f"Bug Analysis Workflow: collected {len(hypothesis_files)} hypothesis files")
        return {"hypothesis_files": hypothesis_files}

    def run_hypothesis_investigators(self, state: State) -> dict:
        hypothesis_files = state.get("hypothesis_files") or []
        if not hypothesis_files:
            print("Bug Analysis Workflow: no hypothesis files found; skipping investigation")
            return {}

        print(
            f"Bug Analysis Workflow: running {len(hypothesis_files)} hypothesis investigators in parallel"
        )
        investigators: list[HypothesisInvestigator] = []
        for hyp_path in hypothesis_files:
            investigator = HypothesisInvestigator(
                issue_dir=state["issue_dir"],
                hypothesis_path=hyp_path,
                source_dir=state["source_dir"],
                agent_workspace_dir=state["agent_workspace_dir"],
                model=state["model"],
                model_provider=state["model_provider"],
                api_key=state["api_key"],
            )
            investigator.build_workflow()
            investigators.append(investigator)

        async def _run_all() -> None:
            await asyncio.gather(*[i.run() for i in investigators])

        asyncio.run(_run_all())
        print("Bug Analysis Workflow: hypothesis investigation finished")
        return {}

    def build_workflow(self) -> None:
        workflow = StateGraph(self.State)
        workflow.add_node("run_hypothesis_generator", self.run_hypothesis_generator)
        workflow.add_node("collect_hypothesis_files", self.collect_hypothesis_files)
        workflow.add_node(
            "run_hypothesis_investigators", self.run_hypothesis_investigators
        )

        workflow.add_edge(START, "run_hypothesis_generator")
        workflow.add_edge("run_hypothesis_generator", "collect_hypothesis_files")
        workflow.add_edge("collect_hypothesis_files", "run_hypothesis_investigators")
        workflow.add_edge("run_hypothesis_investigators", END)
        self.workflow = workflow.compile()

    def run(self) -> None:
        if self.workflow is None:
            self.build_workflow()
        self.workflow.invoke(
            {
                "issue_dir": self.issue_dir,
                "source_dir": self.source_dir,
                "agent_workspace_dir": self.agent_workspace_dir,
                "model": self.model,
                "model_provider": self.model_provider,
                "api_key": self.api_key,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bug analysis workflow for one issue directory"
    )
    parser.add_argument(
        "--issue-dir",
        dest="issue_dir",
        required=True,
        help="Path to issue directory containing issue_details.json",
    )
    parser.add_argument(
        "--source-dir",
        dest="source_dir",
        required=True,
        help="Path to source repository used by investigation tools",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Agent workspace directory containing file_analysis.md",
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

    workflow = BugAnalysisWorkflow(
        issue_dir=args.issue_dir,
        source_dir=args.source_dir,
        agent_workspace_dir=args.workspace,
        model=args.model,
        model_provider=args.model_provider,
        api_key=args.api_key,
    )
    workflow.run()


if __name__ == "__main__":
    main()
