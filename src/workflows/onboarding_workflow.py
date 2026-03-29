from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import asyncio

from utils import *

from .business_analysis_workflow import BusinessAnalysisWorkflow
from .contributor_analysis_workflow import ContributorAnalysisWorkflow
from .structural_analysis_workflow import StructuralAnalysisWorkflow

class OnboardingWorkflow():
    def __init__(self, source_repository: str, agent_workspace: str, structural_analysis_batch_size: int, model: str, model_provider: str, api_key: str):
        self.source_repository = source_repository
        self.agent_workspace = agent_workspace
        self.structural_analysis_batch_size = structural_analysis_batch_size
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key

    class State(TypedDict):
        source_repository: str
        agent_workspace: str
        structural_analysis_batch_size: int
        model: str
        model_provider: str
        api_key: str
        structural_analysis_workflow: StructuralAnalysisWorkflow
        contributor_analysis_workflow: ContributorAnalysisWorkflow
        business_analysis_workflow: BusinessAnalysisWorkflow

    # Nodes
    def run_structural_analysis(self, state: State):
        structural_analysis_workflow = StructuralAnalysisWorkflow(state["source_repository"], state["agent_workspace"], state["structural_analysis_batch_size"], state["model"], state["model_provider"], state["api_key"])
        structural_analysis_workflow.build_workflow()
        async def _run():
            return await structural_analysis_workflow.run()
        asyncio.run(_run())
        return {"structural_analysis_workflow": structural_analysis_workflow}

    def run_business_analysis(self,state: State):
        business_analysis_workflow = BusinessAnalysisWorkflow(state["structural_analysis_workflow"].write_directory, state["agent_workspace"], state["model"], state["model_provider"], state["api_key"])
        business_analysis_workflow.build_workflow()
        async def _run():
            return await business_analysis_workflow.run()
        asyncio.run(_run())
        return {"business_analysis_workflow": business_analysis_workflow}

    def run_contributor_analysis(self, state: State):
        contributor_analysis_workflow = ContributorAnalysisWorkflow(state["source_repository"], state["agent_workspace"], state["model"], state["model_provider"], state["api_key"])
        contributor_analysis_workflow.build_workflow()
        async def _run():
            return await contributor_analysis_workflow.run()
        asyncio.run(_run())
        return {"contributor_analysis_workflow": contributor_analysis_workflow}

    def print_summary(self, state: State):
        print(f"Onboarding Workflow: Structural Analysis Status: {state['structural_analysis_workflow'].status}")
        print(f"Onboarding Workflow: Business Analysis Status: {state['business_analysis_workflow'].status}")
        print(f"Onboarding Workflow: Contributor Analysis Status: {state['contributor_analysis_workflow'].status}")
        print(f"Onboarding Workflow: Completed")

    def build_workflow(self):
        # Build workflow
        workflow = StateGraph(self.State)
        workflow.add_node("run_structural_analysis", self.run_structural_analysis)
        workflow.add_node("run_business_analysis", self.run_business_analysis)
        workflow.add_node("run_contributor_analysis", self.run_contributor_analysis)
        workflow.add_node("print_summary", self.print_summary)

        workflow.add_edge(START, "run_structural_analysis")
        workflow.add_edge("run_structural_analysis", "run_business_analysis")
        workflow.add_edge("run_business_analysis", "run_contributor_analysis")
        workflow.add_edge("run_contributor_analysis", "print_summary")
        workflow.add_edge("print_summary", END)
        self.workflow = workflow.compile()

    def run(self):
        self.workflow.invoke({
            "source_repository": self.source_repository,
            "agent_workspace": self.agent_workspace,
            "structural_analysis_batch_size": self.structural_analysis_batch_size,
            "model": self.model,
            "model_provider": self.model_provider,
            "api_key": self.api_key,
        })