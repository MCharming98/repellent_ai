import asyncio

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from utils import *
from utils.langchain import merge_token_usage_totals

from .business_analysis_workflow import BusinessAnalysisWorkflow
from .contributor_analysis_workflow import ContributorAnalysisWorkflow
from .file_analysis_workflow import FileAnalysisWorkflow


class OnboardingWorkflow:
    def __init__(self, source_repository: str, domain_knowledge: str, file_analysis_batch_size: int, model: str, model_provider: str, api_key: str):
        self.source_repository = source_repository
        self.domain_knowledge = domain_knowledge
        self.file_analysis_batch_size = file_analysis_batch_size
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key

    class State(TypedDict):
        source_repository: str
        domain_knowledge: str
        file_analysis_batch_size: int
        model: str
        model_provider: str
        api_key: str
        file_analysis_workflow: FileAnalysisWorkflow
        contributor_analysis_workflow: ContributorAnalysisWorkflow
        business_analysis_workflow: BusinessAnalysisWorkflow

    # Nodes
    def run_file_analysis(self, state: State):
        file_analysis_workflow = FileAnalysisWorkflow(
            state["source_repository"],
            state["domain_knowledge"],
            state["file_analysis_batch_size"],
            state["model"],
            state["model_provider"],
            state["api_key"],
        )
        file_analysis_workflow.build_workflow()

        async def _run():
            return await file_analysis_workflow.run()

        asyncio.run(_run())
        return {"file_analysis_workflow": file_analysis_workflow}

    def run_business_analysis(self, state: State):
        business_analysis_workflow = BusinessAnalysisWorkflow(
            state["file_analysis_workflow"].write_directory,
            state["domain_knowledge"],
            state["model"],
            state["model_provider"],
            state["api_key"],
        )
        business_analysis_workflow.build_workflow()

        async def _run():
            return await business_analysis_workflow.run()

        asyncio.run(_run())
        return {"business_analysis_workflow": business_analysis_workflow}

    def run_contributor_analysis(self, state: State):
        contributor_analysis_workflow = ContributorAnalysisWorkflow(
            state["source_repository"],
            state["domain_knowledge"],
            state["model"],
            state["model_provider"],
            state["api_key"],
        )
        contributor_analysis_workflow.build_workflow()

        async def _run():
            return await contributor_analysis_workflow.run()

        asyncio.run(_run())
        return {"contributor_analysis_workflow": contributor_analysis_workflow}

    def print_summary(self, state: State):
        print(f"Onboarding Workflow: File Analysis Status: {state['file_analysis_workflow'].status}")
        print(f"Onboarding Workflow: Business Analysis Status: {state['business_analysis_workflow'].status}")
        print(f"Onboarding Workflow: Contributor Analysis Status: {state['contributor_analysis_workflow'].status}")
        fa = state["file_analysis_workflow"].token_usage
        bu = state["business_analysis_workflow"].token_usage
        co = state["contributor_analysis_workflow"].token_usage
        print(
            "Onboarding Workflow: Token usage by stage — "
            f"file_analysis: in={fa['input_tokens']} out={fa['output_tokens']} total={fa['total_tokens']}; "
            f"business_analysis: in={bu['input_tokens']} out={bu['output_tokens']} total={bu['total_tokens']}; "
            f"contributor_analysis: in={co['input_tokens']} out={co['output_tokens']} total={co['total_tokens']}"
        )
        agg = merge_token_usage_totals(None, fa)
        agg = merge_token_usage_totals(agg, bu)
        agg = merge_token_usage_totals(agg, co)
        print(
            "Onboarding Workflow: Aggregated token usage — "
            f"input={agg['input_tokens']} output={agg['output_tokens']} total={agg['total_tokens']}"
        )
        print("Onboarding Workflow: Completed")

    def build_workflow(self):
        workflow = StateGraph(self.State)
        workflow.add_node("run_file_analysis", self.run_file_analysis)
        workflow.add_node("run_business_analysis", self.run_business_analysis)
        workflow.add_node("run_contributor_analysis", self.run_contributor_analysis)
        workflow.add_node("print_summary", self.print_summary)

        workflow.add_edge(START, "run_file_analysis")
        workflow.add_edge("run_file_analysis", "run_business_analysis")
        workflow.add_edge("run_business_analysis", "run_contributor_analysis")
        workflow.add_edge("run_contributor_analysis", "print_summary")
        workflow.add_edge("print_summary", END)
        self.workflow = workflow.compile()

    def run(self):
        self.workflow.invoke({
            "source_repository": self.source_repository,
            "domain_knowledge": self.domain_knowledge,
            "file_analysis_batch_size": self.file_analysis_batch_size,
            "model": self.model,
            "model_provider": self.model_provider,
            "api_key": self.api_key,
        })
