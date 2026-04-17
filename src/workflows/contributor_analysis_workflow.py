import threading
import time
from pathlib import PurePath
from typing_extensions import NotRequired, TypedDict
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from utils import *

from constants.contributor_analysis_constants import (
    CONTRIBUTOR_ANALYSIS_SCHEMA,
    get_contributor_analysis_prompt_intro,
)
from utils.langchain import aggregate_token_usage_from_messages, merge_token_usage_totals


class ContributorAnalysisWorkflow():
    def __init__(self, read_directory: str, write_directory: str, model: str, model_provider: str, api_key: str):
        self.read_directory = read_directory
        self.write_directory = write_directory
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key

    class State(TypedDict):
        read_directory: str
        source_code_files: list[str]
        contributors: dict[str, list[str]]
        contributor_analysis: str
        write_directory: str
        token_usage: NotRequired[dict[str, int]]

    # Nodes
    def list_source_code_files(self, state: State):
        files = list_source_files_recursive(state['read_directory'])
        print(f"Contributor Analysis Workflow: List {len(files)} source code files in {state['read_directory']}")
        return {"source_code_files": files}

    def collect_contributors(self, state: State):
        print("Contributor Analysis Workflow: Collecting contributors for each file")
        contributors = {}
        for file in state['source_code_files']:
            contributors[file] = get_contributors(state['read_directory'], file)
        return {"contributors": contributors}

    def generate_contributor_analysis(self, state: State):
        start_time = time.perf_counter()
        stop_event = threading.Event()

        def _print_elapsed():
            while not stop_event.is_set():
                elapsed = time.perf_counter() - start_time
                print(f"\rContributor Analysis Workflow: Generating contributor analysis... {elapsed:.1f}s", end="")
                stop_event.wait(1)

        thread = threading.Thread(target=_print_elapsed, daemon=True)
        thread.start()
        prompt = get_contributor_analysis_prompt_intro()
        for file, file_contributors in state['contributors'].items():
            prompt += f"""
            \n
            - File: {state['read_directory']+ file}
            - Contributors(number of commits, name, account): {file_contributors}
            \n
            """
        model_kwargs = {"api_key": self.api_key}
        # Default to Google Developer API instead of Vertex AI
        if self.model_provider == "google_genai":
            model_kwargs["google_api_key"] = self.api_key
        agent = create_agent(
            model=init_chat_model(self.model, model_provider=self.model_provider, **model_kwargs),
            response_format=ToolStrategy(CONTRIBUTOR_ANALYSIS_SCHEMA),
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        stop_event.set()
        thread.join()
        elapsed = time.perf_counter() - start_time
        print(f"\rContributor Analysis Workflow: Generating contributor analysis completed in {elapsed:.2f}s")
        usage = aggregate_token_usage_from_messages(result.get("messages") or [])
        token_usage = merge_token_usage_totals(None, usage)
        return {
            "contributor_analysis": result["structured_response"]["text"],
            "token_usage": token_usage,
        }

    def write_analysis_to_file(self, state: State):
        print(f"Contributor Analysis Workflow: Writing contributor analysis to {state['write_directory']}contributor_analysis.md")
        write_to_file(str(PurePath(state["write_directory"], "contributor_analysis.md")), state["contributor_analysis"], 'w')
        return {}

    def build_workflow(self):
        # Build workflow
        workflow = StateGraph(self.State)
        workflow.add_node("list_source_code_files", self.list_source_code_files)
        workflow.add_node("get_contributors", self.collect_contributors)
        workflow.add_node("generate_contributor_analysis", self.generate_contributor_analysis)
        workflow.add_node("write_analysis_to_file", self.write_analysis_to_file)

        workflow.add_edge(START, "list_source_code_files")
        workflow.add_edge("list_source_code_files", "get_contributors")
        workflow.add_edge("get_contributors", "generate_contributor_analysis")
        workflow.add_edge("generate_contributor_analysis", "write_analysis_to_file")
        workflow.add_edge("write_analysis_to_file", END)

        # Compile workflow
        self.workflow = workflow.compile()

    async def run(self):
        print(f"Contributor Analysis Workflow: Run workflow")
        await self.workflow.ainvoke({
            "read_directory": self.read_directory,
            "write_directory": self.write_directory,
        })
        self.status = True