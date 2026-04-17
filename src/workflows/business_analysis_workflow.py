import threading
import time
from pathlib import PurePath
from typing_extensions import NotRequired, TypedDict
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from utils import *

from constants.business_analysis_constants import (
    BUSINESS_ANALYSIS_SCHEMA,
    get_business_analysis_prompt,
)
from utils.langchain import aggregate_token_usage_from_messages, merge_token_usage_totals


class BusinessAnalysisWorkflow():
    def __init__(self, read_directory: str, write_directory: str, model: str, model_provider: str, api_key: str):
        self.read_directory = read_directory
        self.write_directory = write_directory
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key

    class State(TypedDict):
        read_directory: str
        business_analysis: str
        write_directory: str
        token_usage: NotRequired[dict[str, int]]

    # Nodes
    def generate_business_analysis(self, state: State):
        start_time = time.perf_counter()
        stop_event = threading.Event()

        def _print_elapsed():
            while not stop_event.is_set():
                elapsed = time.perf_counter() - start_time
                print(f"\rBusiness Analysis Workflow: Generating analysis... {elapsed:.1f}s", end="")
                stop_event.wait(1)

        thread = threading.Thread(target=_print_elapsed, daemon=True)
        thread.start()
        input_path = PurePath(state["read_directory"], "file_analysis.md")
        input = read_file(str(input_path))
        prompt = get_business_analysis_prompt(input)
        agent = create_agent(
            model=init_chat_model(self.model, model_provider=self.model_provider, api_key=self.api_key),
            response_format=ToolStrategy(BUSINESS_ANALYSIS_SCHEMA),
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        stop_event.set()
        thread.join()
        elapsed = time.perf_counter() - start_time
        print(f"\rBusiness Analysis Workflow: Generating analysis completed in {elapsed:.2f}s")
        usage = aggregate_token_usage_from_messages(result.get("messages") or [])
        token_usage = merge_token_usage_totals(None, usage)
        return {
            "business_analysis": result["structured_response"]["text"],
            "token_usage": token_usage,
        }

    def write_analysis_to_file(self, state: State):
        print(f"Business Analysis Workflow: Writing analysis to {state['write_directory']}/business_analysis.md")
        output_path = PurePath(state["write_directory"], "business_analysis.md")
        write_to_file(str(output_path), state["business_analysis"], 'w')
        return {}

    def build_workflow(self):
        # Build workflow
        workflow = StateGraph(self.State)
        workflow.add_node("generate_business_analysis", self.generate_business_analysis)
        workflow.add_node("write_analysis_to_file", self.write_analysis_to_file)

        workflow.add_edge(START, "generate_business_analysis")
        workflow.add_edge("generate_business_analysis", "write_analysis_to_file")
        workflow.add_edge("write_analysis_to_file", END)

        # Compile workflow
        self.workflow = workflow.compile()

    async def run(self):
        print(f"Business Analysis Workflow: Run workflow")
        await self.workflow.ainvoke({
            "read_directory": self.read_directory,
            "write_directory": self.write_directory,
        })
        self.status = True