"""Investigates a single hypothesis against an issue using domain knowledge."""

import asyncio
import enum
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    # Allow direct execution: python src/agents/graph_hypothesis_investigator.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from constants.hypothesis_investigator_constants import (
    FORCE_CONVERGENCE_PROMPT,
    INVESTIGATION_ANALYSIS_SCHEMA,
    get_hypothesis_investigator_prompt,
)
from utils import checkout, format_key_to_subheading, read_file, write_to_file, pull_from_repo
from utils.config import default_config_path, get_investigation_max_token_usage, load_config
from utils.langchain import (
    _RATE_LIMIT_WAIT_SECONDS,
    aggregate_token_usage_from_messages,
    get_llm_agent,
    is_http_429,
)
from utils.tools import list_files_tool, read_file_tool 

# Stable section order for markdown output (schema ``required`` order).
_INVESTIGATION_MARKDOWN_KEY_ORDER = tuple(INVESTIGATION_ANALYSIS_SCHEMA["required"])

BENCH_CONFIG_JSON = "bench_config.json"
_RATE_LIMIT_MAX_ATTEMPTS = 1
_INVESTIGATION_RECURSION_LIMIT = 15


def _resolved_issue_dir(issue_dir: str | Path) -> Path:
    p = Path(issue_dir).resolve()
    if p.is_file():
        return p.parent
    return p


def _commit_hash_from_bench_config(issue_dir: Path) -> str | None:
    """Read ``commit_hash`` from ``issue_dir`` / ``bench_config.json`` if present and non-empty."""
    path = issue_dir / BENCH_CONFIG_JSON
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    h = cfg.get("commit_hash")
    if isinstance(h, str) and h.strip():
        return h.strip()
    return None

def format_investigation_analysis_to_markdown(data: dict) -> str:
    """
    Render an INVESTIGATION_ANALYSIS_SCHEMA-shaped dict as markdown.

    Each field uses a ### sub-heading. Arrays (e.g. critical_signals,
    hypothesis_resolution, next_steps) render as bullet lists; ``final_verdict``
    renders as a paragraph.
    """
    lines: list[str] = []
    for key in _INVESTIGATION_MARKDOWN_KEY_ORDER:
        if key not in data:
            continue
        val = data[key]
        lines.append(f"### {format_key_to_subheading(key)}")
        lines.append("")
        if isinstance(val, list):
            for item in val:
                lines.append(f"- {item}")
            lines.append("")
        else:
            lines.append(str(val))
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _message_content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts) if parts else str(content)
    return str(content)


def extract_tool_use_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """
    Collect tool execution results from an agent message list (ToolMessage entries).
    """
    records: list[dict[str, Any]] = []
    for msg in messages or []:
        if isinstance(msg, ToolMessage):
            records.append(
                {
                    "name": getattr(msg, "name", None),
                    "content": _message_content_to_str(getattr(msg, "content", "")),
                    "tool_call_id": getattr(msg, "tool_call_id", None),
                    "status": getattr(msg, "status", None),
                }
            )
    return records


class GraphHypothesisInvestigator:
    """Validates or falsifies one hypothesis using issue context and domain knowledge.

    If ``issue_dir`` / ``bench_config.json`` exists and contains a non-empty ``commit_hash``,
    that ref is checked out in ``source_dir`` during ``analyze_hypothesis``; otherwise no
    checkout is performed.
    """

    class State(TypedDict):
        issue_details: dict
        issue_hypotheses: str
        file_analysis: str
        prompt: str
        cwd: str
        start_time: float
        elapsed_time: float
        hypothesis_analysis: dict
        message_history: list[Any]
        
    def __init__(
        self,
        issue_dir: str,
        source_dir: str,
        domain_knowledge_dir: str,
        model: str,
        model_provider: str,
        api_key: str,
        max_token_usage: int,
    ) -> None:
        issue_path = _resolved_issue_dir(issue_dir)
        self.issue_dir = str(issue_path)
        self.source_dir = source_dir
        self.domain_knowledge_dir = domain_knowledge_dir
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key
        self.max_token_usage = max_token_usage
        self.commit_hash = _commit_hash_from_bench_config(issue_path)

        self.investigator_agent = get_llm_agent(
            model,
            model_provider,
            api_key,
            enable_web_search=True,
            tools=[list_files_tool, read_file_tool],
            response_format=ToolStrategy(INVESTIGATION_ANALYSIS_SCHEMA),
        )

        self.convergence_agent = get_llm_agent(
            model,
            model_provider,
            api_key,
            enable_web_search=False,
            tools=[],
            response_format=ToolStrategy(INVESTIGATION_ANALYSIS_SCHEMA),
        )

    def load_context(self, state: State) -> dict:
        """Load issue JSON, hypotheses.md, and file_analysis from domain knowledge."""
        issue_dir = Path(self.issue_dir)
        issue_details_path = issue_dir / "issue_details.json"
        with open(issue_details_path, encoding="utf-8") as f:
            issue_details = json.load(f)
        hypotheses_path = issue_dir / "hypotheses.md"
        hypotheses = read_file(str(hypotheses_path))
        file_analysis = read_file(
            str(Path(self.domain_knowledge_dir) / "file_analysis.md")
        )
        return {
            "issue_details": issue_details,
            "issue_hypotheses": hypotheses,
            "file_analysis": file_analysis,
        }

    def generate_prompt(self, state: State) -> str:
        issue_details = state["issue_details"]
        title = issue_details.get("title") or ""
        body = issue_details.get("body") or ""
        bug_report = f"{title}\n\n{body}".strip()

        prompt = get_hypothesis_investigator_prompt(
            bug_report=bug_report,
            diagnosis_and_hypotheses=state["issue_hypotheses"],
            file_analysis=state["file_analysis"],
        )
        return {"prompt": prompt}

    def checkout_working_directory_and_commit(self, state: State) -> dict:
        """Check out the working commit if it exists."""
        cwd = os.getcwd()
        os.chdir(self.source_dir)
        pull_from_repo()
        print(f"Hypothesis investigator: changed working directory to: {self.source_dir} and updated to the latest commit")
        
        if self.commit_hash:
            print(
                f"Hypothesis investigator: checking out commit {self.commit_hash}"
            )
            checkout(self.commit_hash)
        return {"cwd": cwd}

    def start_perf_counter(self, state: State) -> dict:
        """Start the performance counter."""
        return {"start_time": time.perf_counter()}

    def perform_investigation(self, state: State) -> dict:
        """Run the investigator agent, resuming with full message history on recursion limit."""
        print("Hypothesis investigator: running analysis agent...")
        messages: list[Any] = [HumanMessage(content=state["prompt"])]
        result: dict[str, Any] = {}
        token_usage = 0

        while token_usage < self.max_token_usage:
            try:
                for chunk in self.investigator_agent.stream(
                    {"messages": messages},
                    config={"recursion_limit": _INVESTIGATION_RECURSION_LIMIT},
                    stream_mode="values",
                ):
                    if isinstance(chunk, dict):
                        result = chunk
                        messages = chunk.get("messages")
                break
            except GraphRecursionError as exc:
                agent_messages = messages
                print(
                    f"Hypothesis investigator: resuming with {len(agent_messages)} message(s) "
                    f"({sum(isinstance(m, HumanMessage) for m in agent_messages)} human, "
                    f"{sum(isinstance(m, AIMessage) for m in agent_messages)} ai, "
                    f"{sum(isinstance(m, ToolMessage) for m in agent_messages)} tool)"
                )
            finally:
                token_usage_dict = aggregate_token_usage_from_messages(messages)
                print(
                    "Hypothesis investigator: analysis tokens "
                    f"in={token_usage_dict['input_tokens']} "
                    f"out={token_usage_dict['output_tokens']} "
                    f"total={token_usage_dict['total_tokens']}"
                )
                token_usage = int(token_usage_dict['total_tokens'])
        structured_response = result.get("structured_response") if isinstance(result, dict) else None
        if not structured_response:
            # Invoke the convergence agent with the full message history
            print("Hypothesis investigator: invoking convergence agent")
            result = self.convergence_agent.invoke({"messages": messages + [HumanMessage(content=FORCE_CONVERGENCE_PROMPT.strip())]})
            structured_response = result.get("structured_response") if isinstance(result, dict) else None
        if not structured_response:
            raise RuntimeError("Hypothesis investigator: no structured response found after convergence agent")
        return {
            "hypothesis_analysis": structured_response,
        }

    def calculate_elapsed_time(self, state: State) -> dict:
        """Calculate the elapsed time."""
        elapsed = time.perf_counter() - state["start_time"]
        return {"elapsed_time": elapsed}

    def restore_working_directory_and_commit(self, state: State) -> dict:
        """Restore git state while cwd is still ``source_dir`` (``checkout`` uses cwd)."""
        if self.commit_hash:
            checkout("latest")
        os.chdir(state["cwd"])
        print(f"Hypothesis investigator: restored working directory to: {state['cwd']}")
        return {}

    def append_hypothesis_analysis_to_file(self, state: State) -> dict:
        """Append structured investigation output as markdown to ``diagnosis.md`` in the issue dir."""
        issue_dir = Path(self.issue_dir)
        diagnosis_path = str(issue_dir / "diagnosis.md")
        md = format_investigation_analysis_to_markdown(state["hypothesis_analysis"])
        block = "## Investigation analysis\n\n" + md
        write_to_file(diagnosis_path, block, mode="a")
        print(f"Hypothesis investigator: appended investigation analysis to {diagnosis_path}")
        return {}

    def build_workflow(self) -> None:
        workflow = StateGraph(self.State)
        workflow.add_node("load_context", self.load_context)
        workflow.add_node("generate_prompt", self.generate_prompt)
        workflow.add_node("checkout_working_directory_and_commit", self.checkout_working_directory_and_commit)
        workflow.add_node("start_perf_counter", self.start_perf_counter)
        workflow.add_node("perform_investigation", self.perform_investigation)
        workflow.add_node("calculate_elapsed_time", self.calculate_elapsed_time)
        workflow.add_node("restore_working_directory_and_commit", self.restore_working_directory_and_commit)
        workflow.add_node("append_hypothesis_analysis_to_file", self.append_hypothesis_analysis_to_file)
        
        workflow.add_edge(START, "load_context")
        workflow.add_edge("load_context", "generate_prompt")
        workflow.add_edge("generate_prompt", "checkout_working_directory_and_commit")
        workflow.add_edge("checkout_working_directory_and_commit", "start_perf_counter")
        workflow.add_edge("start_perf_counter", "perform_investigation")
        workflow.add_edge("perform_investigation", "calculate_elapsed_time")
        workflow.add_edge("calculate_elapsed_time", "restore_working_directory_and_commit")
        workflow.add_edge("restore_working_directory_and_commit", "append_hypothesis_analysis_to_file")
        workflow.add_edge("append_hypothesis_analysis_to_file", END)
        self.workflow = workflow.compile()

    async def run(self) -> State:
        if self.workflow is None:
            self.build_workflow()
        diagnosis_path = Path(self.issue_dir) / "diagnosis.md"
        if diagnosis_path.exists():
            print(
                f"Hypothesis investigator: '{diagnosis_path}' already exists; "
                "skipping investigation."
            )
            return {"tool_use": []}
        final_state = await self.workflow.ainvoke({})
        return final_state


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Graph-based hypothesis investigator (issue context + tools)"
    )
    parser.add_argument(
        "--issue-dir",
        required=True,
        help="Path to issue dir or issue_details.json",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Source repository dir used as working dir during investigation",
    )
    parser.add_argument(
        "--domain-knowledge",
        required=True,
        dest="domain_knowledge_dir",
        help="Domain knowledge dir (contains file_analysis.md)",
    )
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--model-provider", default="google_genai")
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()

    cfg = load_config(default_config_path())
    max_token_usage = get_investigation_max_token_usage(cfg)

    investigator = GraphHypothesisInvestigator(
        issue_dir=args.issue_dir,
        source_dir=args.source_dir,
        domain_knowledge_dir=args.domain_knowledge_dir,
        model=args.model,
        model_provider=args.model_provider,
        api_key=args.api_key,
        max_token_usage=max_token_usage,
    )
    investigator.build_workflow()
    asyncio.run(investigator.run())


if __name__ == "__main__":
    main()
