"""Agent for analyzing GitHub issues using project knowledge from agent workspace."""

import json
from pathlib import Path

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from utils import read_file, extract_image_markdown


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

    def __init__(self, issue_details_path: str, agent_workspace: str):
        self.issue_details_path = Path(issue_details_path)
        self.agent_workspace = Path(agent_workspace)

    def load_issue(self) -> State:
        """Load issue details and project knowledge into state."""
        with open(self.issue_details_path, "r", encoding="utf-8") as f:
            issue_details = json.load(f)
        return {"issue_details": issue_details}

    def load_issue_images(self) -> State:
        """Load issue attachments from issue details."""
        issue_details = self.state["issue_details"]
        images = extract_image_markdown(issue_details["body"])
        return {"issue_images": images}

    def load_project_knowledge(self) -> dict:
        """Load project knowledge from agent workspace (file_analysis, business_analysis, contributor_analysis)."""
        file_analysis = read_file(str(self.agent_workspace / "file_analysis.md"))
        business_analysis = read_file(str(self.agent_workspace / "business_analysis.md"))
        contributor_analysis = read_file(str(self.agent_workspace / "contributor_analysis.md"))
        return {"file_analysis": file_analysis, "business_analysis": business_analysis, "contributor_analysis": contributor_analysis}

    def analyze_issue(self) -> State:
        """Analyze issue using project knowledge."""
        issue_details = self.state["issue_details"]
        issue_images = self.state["issue_images"]
        file_analysis = self.state["file_analysis"]
        business_analysis = self.state["business_analysis"]
        contributor_analysis = self.state["contributor_analysis"]
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
        return {}

    def build_workflow(self):
        self.workflow = StateGraph(self.State)
        self.workflow.add_node("load_issue", self.load_issue)
        self.workflow.add_node("load_project_knowledge", self.load_project_knowledge)

        self.workflow.add_edge(START, "load_issue")
        self.workflow.add_edge("load_issue", "load_project_knowledge")
        self.workflow.add_edge("load_project_knowledge", END)
        self.workflow = self.workflow.compile()

    async def run(self):
        final_state = await self.workflow.ainvoke({
            "issue_details_path": self.issue_details_path,
            "agent_workspace": self.agent_workspace,
        })
        return final_state