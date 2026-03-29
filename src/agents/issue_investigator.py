"""Agent for analyzing GitHub issues using project knowledge from agent workspace."""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import time
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.structured_output import ToolStrategy
from utils import read_file, extract_image_markdown, fetch_image_as_data_url, write_to_file


def _format_issue_analysis_markdown(data: dict) -> str:
    """Render structured issue analysis JSON as markdown."""
    section_order = [
        "symptom_observed",
        "divergence_point",
        "issue_type",
        "diagnose_hypothesis",
    ]
    blocks: list[str] = []
    for key in section_order:
        if key not in data:
            continue
        val = data[key]
        blocks.append(f"## {key}\n")
        if key in ("symptom_observed", "divergence_point", "issue_type"):
            if isinstance(val, dict):
                blocks.append(str(val.get("analysis", "")).strip())
                blocks.append("")
                blocks.append(f"Confidence score: {val.get('confidence_score', '')}")
        elif key == "diagnose_hypothesis" and isinstance(val, list):
            for i, item in enumerate(val):
                if not isinstance(item, dict):
                    continue
                blocks.append(f"### Hypothesis {i + 1}\n")
                blocks.append(str(item.get("hypothesis", "")).strip())
                blocks.append("")
                actions = item.get("investigation_actions") or []
                if actions:
                    blocks.append("Investigation actions")
                    for action in actions:
                        blocks.append(f"- {action}")
                    blocks.append("")
                blocks.append(f"Confidence score: {item.get('confidence_score', '')}")
                blocks.append("")
        blocks.append("")
    return "\n".join(blocks).strip() + "\n"


ISSUE_ANALYSIS_SCHEMA = {
    "type": "object",
    "description": "Issue analysis of the issue",
    "properties": {
        "symptom_observed": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "The analysis of the symptom of the issue",
                },
                "confidence_score": {
                    "type": "number",
                    "description": "The confidence score of the symptom",
                    "minimum": 0,
                    "maximum": 100,
                },
            },
            "required": ["analysis", "confidence_score"],
        },
        "divergence_point": {
            "type": "object",
            "description": "The divergence point between the expected and the actual behavior/CUJ in one sentence",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "The analysis of the expected behavior/CUJ and the divergence point between the expected and the actual behavior/CUJ",
                },
                "confidence_score": {
                    "type": "number",
                    "description": "The confidence score of the divergence point",
                    "minimum": 0,
                    "maximum": 100,
                },
            },
            "required": ["analysis", "confidence_score"],
        },
        "issue_type": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "The analysis and rationale of the issue type: e.g. bug, expected behavior, UX issue, or a feature request",
                },
                "confidence_score": {
                    "type": "number",
                    "description": "The confidence score of the issue type",
                    "minimum": 0,
                    "maximum": 100,
                },
            },
            "required": ["analysis", "confidence_score"],
        },
        "diagnose_hypothesis": {
            "type": "array",
            "description": "The list of diagnose hypothesis and recommended actions for the issue",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis": {
                        "type": "string",
                        "description": "The diagnose hypothesis",
                    },
                    "investigation_actions": {
                        "type": "array",
                        "description": "The list of actions to further investigate into the diagnose hypothesis",
                        "items": {
                            "type": "string",
                            "description": "The detailed recommend action",
                        },
                    },
                    "confidence_score": {
                        "type": "number",
                        "description": "The confidence score of the diagnose hypothesis",
                        "minimum": 0,
                        "maximum": 100,
                    },
                },
                "required": ["hypothesis", "investigation_actions", "confidence_score"],
            },
        },
    },
    "required": ["symptom_observed", "divergence_point", "issue_type", "diagnose_hypothesis"],
}

class IssueInvestigator:
    """Analyzes an issue using project knowledge from the agent workspace."""

    class State(TypedDict):
        issue_directory: str
        agent_workspace: str
        issue_details: dict
        issue_images: list[str]
        file_analysis: str
        business_analysis: str
        contributor_analysis: str
        issue_analysis_json: dict
        issue_analysis: str
        write_status: bool

    def __init__(self, id: str, issue_directory: str, agent_workspace: str, model: str, model_provider: str, api_key: str):
        self.id = id
        self.issue_directory = Path(issue_directory)
        self.agent_workspace = Path(agent_workspace)
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key
        model_kwargs = {"api_key": api_key}
        # Default to Google Developer API instead of Vertex AI
        if model_provider == "google_genai":
            model_kwargs["google_api_key"] = api_key
        self.agent = create_agent(
            model=init_chat_model(model, model_provider=model_provider, **model_kwargs),
            response_format=ToolStrategy(ISSUE_ANALYSIS_SCHEMA)
        )

    def load_issue(self, state: State) -> dict:
        """Load issue details from JSON into state."""
        path = Path(state["issue_directory"], "issue_details.json")
        with open(path, "r", encoding="utf-8") as f:
            issue_details = json.load(f)
        return {"issue_details": issue_details}

    def load_issue_images(self, state: State) -> dict:
        """Load image URLs from issue body into state."""
        issue_details = state["issue_details"]
        body = issue_details.get("body") or ""
        images = extract_image_markdown(body)
        return {"issue_images": images}

    def load_project_knowledge(self, state: State) -> dict:
        """Load project knowledge from agent workspace (file_analysis, business_analysis, contributor_analysis)."""
        workspace = Path(state["agent_workspace"])
        file_analysis = read_file(str(workspace / "file_analysis.md"))
        business_analysis = read_file(str(workspace / "business_analysis.md"))
        contributor_analysis = read_file(str(workspace / "contributor_analysis.md"))
        return {"file_analysis": file_analysis, "business_analysis": business_analysis, "contributor_analysis": contributor_analysis}

    def analyze_issue(self, state: State) -> dict:
        """Analyze issue using project knowledge."""
        print(f"Issue Investigator #{self.id}: Analyzing {state['issue_directory']}")
        start_time = time.perf_counter()
        issue_details = state["issue_details"]
        issue_images = state.get("issue_images") or []
        file_analysis = state["file_analysis"]
        business_analysis = state["business_analysis"]
        contributor_analysis = state["contributor_analysis"]
        prompt = f"""
            You are an experienced software engineer who is talented in bug triageing. 
            Read the following issue report, combining the title, description, 
            and attachments, provide an analysis report with the following 6 sections:
            1. Symptom Observed
                - In technical terms, explain the observed symptom of the issue in one sentence.
                - Assign your symptom analysis a confidence score.
            2. Behavior Divergence Point
                - List the expected behavior or CUJ the user was supposed to go through.
                - Explain the divergence point between the expected and the actual behavior in one sentence.
                - Assign your divergence point analysis a confidence score.
            3. Issue Type
                - Hypothesize the type of the issue: a bug, expected behavior, UX issue, or a feature request.
                - Explain your rationale in one sentence.
                - Assign your issue type analysis a confidence score.
            4. Diagnose Hypothesis and Investigation Actions
                - List up to 5 hypotheses that are mutually distinct in root cause, not variations of the same issue.
                - For each hypothesis, provide the following:
                    1. Mechanism analysis:
                        - A step-by-step causal chain explaining how the system transitions from a correct state to the observed failure.
                        - Reference specific components (functions, services, data flow).
                        - If the diagnose points to source code, provide the file name and the function/class name, if applicable, by referring to the structural analysis.
                    2. Observable implications analysis:
                        - What logs, metrics, or behaviors must be true if this hypothesis is correct?
                    3. Investigation actions:
                        - Provide 5 concrete actions that would confirm or falsify this hypothesis.
                        - Sample actions include but are not limited to: code inspection, log query, unit test, web search, ask user, etc.
                    4. Confidence score:
                        - Based on completeness of mechanism, testability, and clear actionable steps (not intuition).
                - Constraints:
                    - Do NOT output vague causes (e.g., "race condition", "bug in logic")
                    without explaining the exact mechanism.
                    - Prefer hypotheses that can be tested quickly.
                    - Each hypothesis must be falsifiable.

            Rules and Guidelines:
            - Your language should be techinical-oriented, so engineers can quickly understand and investigate.
            - Your analysis should be concise and straight to the point.
            - Refer to the provided domain knowledge documents for domain knowledge.

            Issue Title: {issue_details["title"]}

            Issue Description: {issue_details["body"]}
            
            Domain knowledge documents:
            File analysis: {file_analysis}

            Business analysis: {business_analysis}
        """
        image_blocks = []
        for image_url in issue_images:
            data_url = fetch_image_as_data_url(image_url)
            if data_url:
                image_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
        message = {
            "role": "user",
            "content": [{"type": "text", "text": prompt}, *image_blocks],
        }
        result = self.agent.invoke(
            {"messages": [message]}
        )
        elapsed = time.perf_counter() - start_time
        print(f"Issue Investigator #{self.id}: analysis completed in {elapsed:.2f}s")
        sr = result.get("structured_response")
        if not isinstance(sr, dict):
            raise ValueError(
                f"Issue Investigator #{self.id}: expected structured_response dict, got {type(sr)}"
            )
        if "symptom_observed" not in sr and "text" in sr:
            try:
                sr = json.loads(sr["text"])
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(
                    f"Issue Investigator #{self.id}: could not parse structured_response: {e}"
                ) from e
        return {"issue_analysis_json": sr}

    def format_issue_analysis_markdown(self, state: State) -> dict:
        md = _format_issue_analysis_markdown(state["issue_analysis_json"])
        return {"issue_analysis": md}

    def write_analysis_to_file(self, state: State) -> dict:
        output_path = Path(state["issue_directory"]) / "issue_analysis.md"
        print(f"Issue Investigator #{self.id}: Writing analysis to {output_path}")
        write_to_file(str(output_path), state["issue_analysis"], "w")
        return {"write_status": True}

    def build_workflow(self):
        self.workflow = StateGraph(self.State)
        self.workflow.add_node("load_issue", self.load_issue)
        self.workflow.add_node("load_issue_images", self.load_issue_images)
        self.workflow.add_node("load_project_knowledge", self.load_project_knowledge)
        self.workflow.add_node("analyze_issue", self.analyze_issue)
        self.workflow.add_node("format_issue_analysis_markdown", self.format_issue_analysis_markdown)
        self.workflow.add_node("write_analysis_to_file", self.write_analysis_to_file)

        self.workflow.add_edge(START, "load_issue")
        self.workflow.add_edge("load_issue", "load_issue_images")
        self.workflow.add_edge("load_issue_images", "load_project_knowledge")
        self.workflow.add_edge("load_project_knowledge", "analyze_issue")
        self.workflow.add_edge("analyze_issue", "format_issue_analysis_markdown")
        self.workflow.add_edge("format_issue_analysis_markdown", "write_analysis_to_file")
        self.workflow.add_edge("write_analysis_to_file", END)
        self.workflow = self.workflow.compile()

    async def run(self):
        final_state = await self.workflow.ainvoke({
            "issue_directory": str(self.issue_directory),
            "agent_workspace": str(self.agent_workspace),
        })
        return final_state


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a GitHub issue using project knowledge from an agent workspace"
    )
    parser.add_argument(
        "--issue-details",
        required=True,
        dest="issue_details",
        help="Path to issue_details.json",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Agent workspace directory (file_analysis.md, business_analysis.md, contributor_analysis.md)",
    )
    parser.add_argument(
        "--model_name",
        default="gemini-3-flash-preview",
        help="LLM model (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--model_provider",
        default="google_genai",
        help="LLM provider (default: google_genai)",
    )
    parser.add_argument("--api-key", required=True, help="API key for the LLM provider")
    args = parser.parse_args()

    agent = IssueInvestigator(
        id='0',
        issue_directory=args.issue_details,
        agent_workspace=args.workspace,
        model=args.model_name,
        model_provider=args.model_provider,
        api_key=args.api_key,
    )
    agent.build_workflow()
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()