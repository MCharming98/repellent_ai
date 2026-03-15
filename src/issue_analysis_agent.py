"""Agent for analyzing GitHub issues using project knowledge from agent workspace."""

import json
from pathlib import Path

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from utils import read_file


class IssueAnalysisAgent:
    """Analyzes an issue using project knowledge from the agent workspace."""

    class State(TypedDict):
        issue_details_path: str
        agent_workspace: str
        issue_details: dict
        structure_analysis: str
        business_analysis: str
        contributor_analysis: str

    def __init__(self, issue_details_path: str, agent_workspace: str):
        self.issue_details_path = Path(issue_details_path)
        self.agent_workspace = Path(agent_workspace)

    def load_state(self) -> State:
        """Load issue details and project knowledge into state."""
        with open(self.issue_details_path, "r", encoding="utf-8") as f:
            issue_details = json.load(f)
        return {"issue_details": issue_details}

    def load_project_knowledge(self) -> dict:
        """Load project knowledge from agent workspace (file_analysis, business_analysis, contributor_analysis)."""
        structure_analysis = read_file(str(self.agent_workspace / "file_analysis.md"))
        business_analysis = read_file(str(self.agent_workspace / "business_analysis.md"))
        contributor_analysis = read_file(str(self.agent_workspace / "contributor_analysis.md"))
        return {"structure_analysis": structure_analysis, "business_analysis": business_analysis, "contributor_analysis": contributor_analysis}

    def build_workflow(self):
        self.workflow = StateGraph(self.State)
        self.workflow.add_node("load_state", self.load_state)
        self.workflow.add_node("load_project_knowledge", self.load_project_knowledge)

        self.workflow.add_edge(START, "load_state")
        self.workflow.add_edge("load_state", "load_project_knowledge")
        self.workflow.add_edge("load_project_knowledge", END)
        self.workflow = self.workflow.compile()

    async def run(self):
        final_state = await self.workflow.ainvoke({
            "issue_details_path": self.issue_details_path,
            "agent_workspace": self.agent_workspace,
        })
        return final_state