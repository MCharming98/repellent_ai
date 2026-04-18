import asyncio
import os
import threading
import time
from typing_extensions import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.file_analyzer import FileAnalyzer
from constants.file_analyzer_constants import get_file_analyzer_prompt_header
from utils import *
from utils.langchain import merge_token_usage_totals
from utils.text import estimate_token_count

# Reserved tokens for output / safety margin when sizing file batches.
DEFAULT_FILE_ANALYSIS_TOKEN_BUFFER = 10_000


class FileAnalysisWorkflow:
    def __init__(
        self,
        read_directory: str,
        write_directory: str,
        file_batch_size: int,
        model: str,
        model_provider: str,
        api_key: str,
        model_context_window: int,
    ):
        self.read_directory = read_directory
        self.file_batch_size = file_batch_size
        self.write_directory = write_directory
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key
        self.model_context_window = model_context_window
        self.token_usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    class State(TypedDict):
        read_directory: str
        file_batch_size: int
        write_directory: str
        model: str
        model_provider: str
        model_context_window: int
        api_key: str
        source_code_files: list[str]
        token_count_map: dict[str, int]
        file_batches: list[list[str]]
        agents: list[FileAnalyzer]
        token_usage: NotRequired[dict[str, int]]

    # Nodes
    def list_source_code_files(self, state: State):
        files = list_source_files_recursive(state['read_directory'])
        print(f"File Analysis Workflow: List {len(files)} source code files in {state['read_directory']}")
        return {"source_code_files": files}

    def estimate_file_token_count(self, state: State):
        token_count_map: dict[str, int] = {}
        read_dir = state["read_directory"]
        model_provider = state["model_provider"]
        model_name = state["model"]
        for rel_path in state["source_code_files"]:
            if rel_path.startswith("Error:"):
                continue
            full_path = os.path.join(read_dir, rel_path)
            content = read_file(full_path)
            if content.startswith("Error:"):
                token_count_map[rel_path] = 0
                continue
            n = estimate_token_count(
                content,
                model_provider,
                model_name=model_name,
            )
            token_count_map[rel_path] = n
        print(
            f"File Analysis Workflow: Estimated tokens for {len(token_count_map)} files "
        )
        return {"token_count_map": token_count_map}

    def create_file_batches(self, state: State):
        """Compute per-batch token budget for file content."""
        buffer = DEFAULT_FILE_ANALYSIS_TOKEN_BUFFER
        prompt_text = get_file_analyzer_prompt_header(1)
        prompt_tokens = estimate_token_count(
            state["model_provider"],
            prompt_text,
            model_name=state["model"],
        )
        limit = state["model_context_window"] - prompt_tokens - buffer
        if limit < 0:
            raise ValueError(
                f"File batch token limit is negative ({limit}): model_context_window="
                f"{state['model_context_window']} is too small for prompt (~{prompt_tokens} tokens) "
                f"plus buffer ({buffer}). Increase model_context_window or reduce the prompt/buffer."
            )

        file_batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0
        for rel_path, file_tokens in sorted(state["token_count_map"].items()):
            if current_batch and current_tokens + file_tokens > limit:
                file_batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(rel_path)
            current_tokens += file_tokens
        if current_batch:
            file_batches.append(current_batch)

        return {
            "file_batches": file_batches,
        }

    def create_agents(self, state: State):
        id = 0
        agents = []
        for batch in state["file_batches"]:
            if not batch:
                continue
            agent = FileAnalyzer(str(id), state['read_directory'], batch, state['write_directory'], state['model'], state['model_provider'], state['api_key'])
            agents.append(agent)
            id += 1
        print(f"File Analysis Workflow: Created {len(agents)} agents")
        return {"agents": agents}

    def build_agents(self, state: State):
        for agent in state['agents']:
            agent.build_workflow()
        return {"agents": state['agents']}

    def run_agents(self, state: State):
        start_time = time.perf_counter()
        stop_event = threading.Event()

        def _print_elapsed():
            while not stop_event.is_set():
                elapsed = time.perf_counter() - start_time
                print(f"\rFile Analysis Workflow: Running {len(state['agents'])} agents... {elapsed:.1f}s", end="")
                stop_event.wait(1)

        thread = threading.Thread(target=_print_elapsed, daemon=True)
        thread.start()

        async def _run():
            results = await asyncio.gather(*[agent.run() for agent in state["agents"]])
            merged = merge_token_usage_totals(None, None)
            for i, r in enumerate(results):
                if isinstance(r, dict) and r.get("token_usage"):
                    merged = merge_token_usage_totals(merged, r["token_usage"])
                if isinstance(r, dict) and not r.get("write_status", False):
                    print(
                        f"Warning: File Analysis Workflow: agent index {i} returned "
                        "write_status=False"
                    )
            self.token_usage = merged

        asyncio.run(_run())
        stop_event.set()
        thread.join()
        elapsed = time.perf_counter() - start_time
        print(f"\rFile Analysis Workflow: Running {len(state['agents'])} agents completed in {elapsed:.2f}s")
        return {"agents": state["agents"], "token_usage": self.token_usage}

    def build_workflow(self):
        workflow = StateGraph(self.State)
        workflow.add_node("list_source_code_files", self.list_source_code_files)
        workflow.add_node("estimate_file_token_count", self.estimate_file_token_count)
        workflow.add_node("create_file_batches", self.create_file_batches)
        workflow.add_node("create_agents", self.create_agents)
        workflow.add_node("build_agents", self.build_agents)
        workflow.add_node("run_agents", self.run_agents)

        workflow.add_edge(START, "list_source_code_files")
        workflow.add_edge("list_source_code_files", "estimate_file_token_count")
        workflow.add_edge("estimate_file_token_count", "create_file_batches")
        workflow.add_edge("create_file_batches", "create_agents")
        workflow.add_edge("create_agents", "build_agents")
        workflow.add_edge("build_agents", "run_agents")
        workflow.add_edge("run_agents", END)
        self.workflow = workflow.compile()

    async def run(self):
        print("File Analysis Workflow: Run workflow")
        await self.workflow.ainvoke({
            "read_directory": self.read_directory,
            "file_batch_size": self.file_batch_size,
            "write_directory": self.write_directory,
            "model": self.model,
            "model_provider": self.model_provider,
            "model_context_window": self.model_context_window,
            "api_key": self.api_key,
            "source_code_files": [],
            "token_count_map": {},
            "file_batches": [],
            "agents": [],
        })
        self.status = True
