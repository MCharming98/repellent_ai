"""Investigates a single hypothesis against an issue using workspace knowledge."""

from langchain.agents.structured_output import ToolStrategy
from constants.hypothesis_investigator_constants import INVESTIGATION_ANALYSIS_SCHEMA
from utils.langchain import get_llm_agent


class HypothesisInvestigator:
    """Validates or falsifies one hypothesis using issue context and agent workspace data."""

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
            response_format=ToolStrategy(INVESTIGATION_ANALYSIS_SCHEMA),
        )