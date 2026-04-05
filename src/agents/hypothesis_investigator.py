"""Investigates a single hypothesis against an issue using workspace knowledge."""

import asyncio
import json
import sys
import time
from pathlib import Path

from langchain.agents.structured_output import ToolStrategy
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from constants.hypothesis_investigator_constants import (
    INVESTIGATION_ANALYSIS_SCHEMA,
    get_hypothesis_investigator_prompt,
)
from utils import read_file
from utils.langchain import get_llm_agent
from utils.tools import read_file_tool, write_to_file_tool


class HypothesisInvestigator:
    """Validates or falsifies one hypothesis using issue context and agent workspace data."""

    class State(TypedDict):
        issue_details: dict
        hypothesis_details: str
        file_analysis: str
        hypothesis_analysis: dict

    def __init__(
        self,
        issue_path: str,
        hypothesis_path: str,
        agent_workspace_dir: str,
        model: str,
        model_provider: str,
        api_key: str,
    ) -> None:
        self.issue_path = issue_path
        self.hypothesis_path = hypothesis_path
        self.agent_workspace_dir = agent_workspace_dir
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key

        self.agent = get_llm_agent(
            model,
            model_provider,
            api_key,
            enable_web_search=True,
            tools=[read_file_tool, write_to_file_tool],
            response_format=ToolStrategy(INVESTIGATION_ANALYSIS_SCHEMA),
        )
        self.workflow = None

    def load_context(self, state: State) -> dict:
        """Load issue JSON, hypothesis text, and workspace file analysis."""
        issue_dir = Path(self.issue_path)
        if issue_dir.is_file():
            issue_dir = issue_dir.parent
        issue_details_path = issue_dir / "issue_details.json"
        with open(issue_details_path, encoding="utf-8") as f:
            issue_details = json.load(f)
        hypothesis_details = read_file(self.hypothesis_path)
        file_analysis = read_file(
            str(Path(self.agent_workspace_dir) / "file_analysis.md")
        )
        return {
            "issue_details": issue_details,
            "hypothesis_details": hypothesis_details,
            "file_analysis": file_analysis,
        }

    def analyze_hypothesis(self, state: State) -> dict:
        """Run the investigator agent with structured investigation output."""
        issue_details = state["issue_details"]
        title = issue_details.get("title") or ""
        body = issue_details.get("body") or ""
        bug_report = f"{title}\n\n{body}".strip()

        prompt = get_hypothesis_investigator_prompt(
            bug_report,
            state["hypothesis_details"],
            state["file_analysis"],
        )
        print("Hypothesis investigator: running analysis agent...")
        start_time = time.perf_counter()
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        elapsed = time.perf_counter() - start_time
        print(f"Hypothesis investigator: analysis completed in {elapsed:.2f}s")

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
        return {"hypothesis_analysis": sr}

    def build_workflow(self) -> None:
        workflow = StateGraph(self.State)
        workflow.add_node("load_context", self.load_context)
        workflow.add_node("analyze_hypothesis", self.analyze_hypothesis)
        workflow.add_edge(START, "load_context")
        workflow.add_edge("load_context", "analyze_hypothesis")
        workflow.add_edge("analyze_hypothesis", END)
        self.workflow = workflow.compile()

    async def run(self) -> State:
        if self.workflow is None:
            self.build_workflow()
        final_state = await self.workflow.ainvoke({})
        return final_state