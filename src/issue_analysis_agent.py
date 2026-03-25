"""Agent for analyzing GitHub issues using project knowledge from agent workspace."""

import argparse
import asyncio
import json
from pathlib import Path
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.structured_output import ToolStrategy
from utils import read_file, extract_image_markdown, fetch_image_as_data_url, write_to_file


ISSUE_ANALYSIS_SCHEMA = {
    "type": "object",
    "description": "Issue analysis of the issue",
    "properties": {
        "text": {
            "type": "string",
            "description": "The full issue analysis in markdown format",
        }
    },
    "required": ["text"],
}

class IssueAnalysisAgent:
    """Analyzes an issue using project knowledge from the agent workspace."""

    class State(TypedDict):
        issue_details_path: str
        agent_workspace: str
        issue_details: dict
        issue_images: list[str]
        file_analysis: str
        business_analysis: str
        contributor_analysis: str
        issue_analysis: str
        write_status: bool

    def __init__(self, issue_details_path: str, agent_workspace: str, model: str, model_provider: str, api_key: str):
        self.issue_details_path = Path(issue_details_path)
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
        path = Path(state["issue_details_path"])
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
                - In the section body, explain the symptom of the issue in one sentence.
            2. CUJ and Divergence Point
                - In the section body:
                    - List the CUJ the user went through.
                    - Exaplain the divergence point between the expected and the actual CUJ in one sentence.
                    - Assign the CUJ and divergence point a confidence score for each.
                    - Enclose each body in the <details> tag.
                    - Bold “CUJ” and “Divergence Point” in the <summary> tag as title
            3. Issue Type
                - Hypothsize the type of the issue: a bug, expected behavior, UX issue, or a feature request.
                - In the section body:
                    - Assign your hypothesis a confidence score
                    - Explain your rationale in one sentence.
                    - Enclose the rationale in the <details> tag.
            4. Diagnose Hypothesis
                - Hypothesize 3 diagnoses with the highest confidence score of what is directly causing the issue.
                - In the section body:
                    - Explain each hypothesis in one sentence.
                    - Assign each hypothesis a confidence score.
                    - If the diagnose points to source code, provide the file name and the. function/class name, if applicable, by referring to the structural analysis.
                    - Enclose each body in the <details> tag
                    - Put the hypothesis and confidence score in the <summary> tag as title
                        - Format: Hypothesis 1/2/3: [your hypothesis] (Confidence [your confidence %])
            5. Recommened Actions
                - Hypothesize 3 recommended actions with the highest confidence score to further investigate into the issue.
                - If missing important information to produce confident diagnose hypothesis, you can recommend the engineer to ask issue reporter to provide the missing information.
                - In the section body:
                    - State each recommended action in one sentence
                    - Assign each recommended action a confidence score.
                    - If suggesting checking any source code logic, provide the file name and the function/class name, if applicable, by referring to the structural analysis.
                    - Enclose each body in the <details> tag.
                    - Put the action and the confidence score in the <summary> as the title.
                        - Format: Action 1/2/3: [your recommendation] (Confidence [your confidence %])
            6. Suggested Engineers
                - Select 3 suggested engineers with the highest confidence score to further triage this issue.
                - In the section body:
                    - State your rationale in one sentence
                    - Assign each engineer suggestion action a confidence score.
                    - Enclose each body in the <details> tag.
                    - Put the engineer’s name and the confidence score in the <summary> as the title.
                        - Format: Engineer 1/2/3: [engineer’s name] (Confidence [your confidence %])

            Rules and Guidelines:
            - Your language should be techinical-oriented, so engineers can quickly understand and investigate.
            - Your analysis should be concise and straight to the point.
            - Refer to the provided domain knowledge documents for domain knowledge.
            - If provided, refer to the lessons learned document when generating analyis to avoid making known mistakes in your analysis.

            Issue Title: {issue_details["title"]}

            Issue Description: {issue_details["body"]}
            
            Domain knowledge documents:
            File analysis: {file_analysis}

            Business analysis: {business_analysis}

            Contributor analysis: {contributor_analysis}
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
        return {"issue_analysis": result["structured_response"]["text"]}

    def write_analysis_to_file(self, state: State) -> dict:
        out_dir = Path(state["issue_details_path"]).parent
        output_path = out_dir / "issue_analysis.md"
        print(f"Issue Analysis Agent: Writing analysis to {output_path}")
        write_to_file(str(output_path), state["issue_analysis"], "w")
        return {"write_status": True}

    def build_workflow(self):
        self.workflow = StateGraph(self.State)
        self.workflow.add_node("load_issue", self.load_issue)
        self.workflow.add_node("load_issue_images", self.load_issue_images)
        self.workflow.add_node("load_project_knowledge", self.load_project_knowledge)
        self.workflow.add_node("analyze_issue", self.analyze_issue)
        self.workflow.add_node("write_analysis_to_file", self.write_analysis_to_file)

        self.workflow.add_edge(START, "load_issue")
        self.workflow.add_edge("load_issue", "load_issue_images")
        self.workflow.add_edge("load_issue_images", "load_project_knowledge")
        self.workflow.add_edge("load_project_knowledge", "analyze_issue")
        self.workflow.add_edge("analyze_issue", "write_analysis_to_file")
        self.workflow.add_edge("write_analysis_to_file", END)
        self.workflow = self.workflow.compile()

    async def run(self):
        final_state = await self.workflow.ainvoke({
            "issue_details_path": str(self.issue_details_path),
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

    agent = IssueAnalysisAgent(
        issue_details_path=args.issue_details,
        agent_workspace=args.workspace,
        model=args.model_name,
        model_provider=args.model_provider,
        api_key=args.api_key,
    )
    agent.build_workflow()
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()