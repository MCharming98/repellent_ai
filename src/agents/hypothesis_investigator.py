"""Investigates a single hypothesis against an issue using workspace knowledge."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    # Allow direct execution: python src/agents/hypothesis_investigator.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from constants.hypothesis_investigator_constants import (
    INVESTIGATION_ANALYSIS_SCHEMA,
    get_hypothesis_investigator_prompt,
)
from utils import read_file, write_to_file
from utils.langchain import get_llm_agent
from utils.tools import list_files_tool, list_source_files_recursive_tool, read_file_tool, write_to_file_tool 

# Stable section order for markdown output (schema ``required`` order).
_INVESTIGATION_MARKDOWN_KEY_ORDER = tuple(INVESTIGATION_ANALYSIS_SCHEMA["required"])

def _investigation_key_to_subheading(key: str) -> str:
    return key.replace("_", " ").strip().title()

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
        lines.append(f"### {_investigation_key_to_subheading(key)}")
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


def aggregate_token_usage_from_messages(messages: list[Any]) -> dict[str, int] | None:
    """
    Sum token counts from AIMessage.usage_metadata across agent steps (if present).
    """
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    found = False
    for msg in messages or []:
        if not isinstance(msg, AIMessage):
            continue
        um = getattr(msg, "usage_metadata", None)
        if not isinstance(um, dict) or not um:
            continue
        found = True
        # LangChain normalizes common keys; providers may vary slightly.
        it = um.get("input_tokens")
        ot = um.get("output_tokens")
        tt = um.get("total_tokens")
        if it is not None:
            input_tokens += int(it)
        if ot is not None:
            output_tokens += int(ot)
        if tt is not None:
            total_tokens += int(tt)
    if not found:
        return None
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


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


class HypothesisInvestigator:
    """Validates or falsifies one hypothesis using issue context and agent workspace data."""

    class State(TypedDict):
        issue_details: dict
        issue_diagnosis: str
        file_analysis: str
        hypothesis_analysis: dict
        tool_use: list[dict[str, Any]]

    def __init__(
        self,
        issue_dir: str,
        source_dir: str,
        agent_workspace_dir: str,
        model: str,
        model_provider: str,
        api_key: str,
    ) -> None:
        self.issue_dir = issue_dir
        self.source_dir = source_dir
        self.agent_workspace_dir = agent_workspace_dir
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key

        self.agent = get_llm_agent(
            model,
            model_provider,
            api_key,
            enable_web_search=True,
            tools=[list_source_files_recursive_tool, read_file_tool, write_to_file_tool],
            response_format=ToolStrategy(INVESTIGATION_ANALYSIS_SCHEMA),
        )
        self.workflow = None

    def load_context(self, state: State) -> dict:
        """Load issue JSON, diagnosis.md (including hypotheses), and workspace file analysis."""
        issue_dir = Path(self.issue_dir)
        if issue_dir.is_file():
            issue_dir = issue_dir.parent
        issue_details_path = issue_dir / "issue_details.json"
        with open(issue_details_path, encoding="utf-8") as f:
            issue_details = json.load(f)
        diagnosis_path = issue_dir / "diagnosis.md"
        diagnosis = read_file(str(diagnosis_path))
        file_analysis = read_file(
            str(Path(self.agent_workspace_dir) / "file_analysis.md")
        )
        return {
            "issue_details": issue_details,
            "issue_diagnosis": diagnosis,
            "file_analysis": file_analysis,
        }

    def analyze_hypothesis(self, state: State) -> dict:
        """Run the investigator agent with structured investigation output."""
        issue_details = state["issue_details"]
        title = issue_details.get("title") or ""
        body = issue_details.get("body") or ""
        bug_report = f"{title}\n\n{body}".strip()

        prompt = get_hypothesis_investigator_prompt(
            bug_report=bug_report,
            diagnosis_and_hypotheses=state["issue_diagnosis"],
            file_analysis=state["file_analysis"],
        )
        print("Hypothesis investigator: running analysis agent...")
        start_time = time.perf_counter()
        prev_cwd = os.getcwd()
        os.chdir(self.source_dir)
        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
        finally:
            os.chdir(prev_cwd)
        elapsed = time.perf_counter() - start_time
        messages_for_usage = result.get("messages") or []
        usage = aggregate_token_usage_from_messages(messages_for_usage)
        if usage:
            print(
                "Hypothesis investigator: analysis completed in "
                f"{elapsed:.2f}s | tokens in={usage['input_tokens']} "
                f"out={usage['output_tokens']} total={usage['total_tokens']}"
            )
        else:
            print(
                f"Hypothesis investigator: analysis completed in {elapsed:.2f}s "
                "(token usage not available on messages)"
            )

        sr = result.get("structured_response")
        if not isinstance(sr, dict):
            raise ValueError(
                f"Hypothesis investigator: expected structured_response dict, got {type(sr)}"
            )
        if "critical_signals" not in sr and "text" in sr:
            try:
                sr = json.loads(sr["text"])
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(
                    f"Hypothesis investigator: could not parse structured_response: {e}"
                ) from e

        messages = messages_for_usage
        tool_use = extract_tool_use_from_messages(messages)

        return {"hypothesis_analysis": sr, "tool_use": tool_use}

    def append_hypothesis_analysis_to_file(self, state: State) -> dict:
        """Append structured investigation output as markdown to ``diagnosis.md`` in the issue dir."""
        issue_dir = Path(self.issue_dir)
        if issue_dir.is_file():
            issue_dir = issue_dir.parent
        diagnosis_path = str(issue_dir / "diagnosis.md")
        md = format_investigation_analysis_to_markdown(state["hypothesis_analysis"])
        block = "\n\n---\n\n## Investigation analysis\n\n" + md
        write_to_file(diagnosis_path, block, mode="a")
        print(f"Hypothesis investigator: appended investigation analysis to {diagnosis_path}")
        return {}

    def build_workflow(self) -> None:
        workflow = StateGraph(self.State)
        workflow.add_node("load_context", self.load_context)
        workflow.add_node("analyze_hypothesis", self.analyze_hypothesis)
        workflow.add_node("append_hypothesis_analysis_to_file", self.append_hypothesis_analysis_to_file)
        workflow.add_edge(START, "load_context")
        workflow.add_edge("load_context", "analyze_hypothesis")
        workflow.add_edge("analyze_hypothesis", "append_hypothesis_analysis_to_file")
        workflow.add_edge("append_hypothesis_analysis_to_file", END)
        self.workflow = workflow.compile()

    async def run(self) -> State:
        if self.workflow is None:
            self.build_workflow()
        final_state = await self.workflow.ainvoke({})
        # print(f"Hypothesis investigator: tool use: {final_state['tool_use']}")
        return final_state


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Investigate one hypothesis against issue context"
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
        "--workspace",
        required=True,
        dest="agent_workspace_dir",
        help="Agent workspace dir (contains file_analysis.md)",
    )
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--model-provider", default="google_genai")
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()

    investigator = HypothesisInvestigator(
        issue_dir=args.issue_dir,
        source_dir=args.source_dir,
        agent_workspace_dir=args.agent_workspace_dir,
        model=args.model,
        model_provider=args.model_provider,
        api_key=args.api_key,
    )
    investigator.build_workflow()
    asyncio.run(investigator.run())


if __name__ == "__main__":
    main()