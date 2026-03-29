import threading
import time
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from utils import *
from file_analyzer import FileAnalyzer
import asyncio

class StructuralAnalysisWorkflow():
    def __init__(self, read_directory: str, write_directory: str, file_batch_size: int, model: str, model_provider: str, api_key: str):
        self.read_directory = read_directory
        self.file_batch_size = file_batch_size
        self.write_directory = write_directory
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key

    class State(TypedDict):
        read_directory: str
        file_batch_size: int
        write_directory: str
        model: str
        model_provider: str
        api_key: str
        source_code_files: list[str]
        agents: list[FileAnalyzer]

    # Nodes
    def list_source_code_files(self, state: State):
        files = list_source_files_recursive(state['read_directory'])
        print(f"Structural Analysis Workflow: List {len(files)} source code files in {state['read_directory']}")
        return {"source_code_files": files}

    def create_agents(self, state: State):
        id = 0
        num_agents = len(state['source_code_files']) // state['file_batch_size'] + 1       
        agents = []
        for i in range(num_agents):
            files = state['source_code_files'][i * state['file_batch_size'] : min((i + 1) * state['file_batch_size'], len(state['source_code_files']))]
            if len(files) == 0:
                continue
            agent = FileAnalyzer(str(id), state['read_directory'], files, state['write_directory'], state['model'], state['model_provider'], state['api_key'])
            agents.append(agent)
            id += 1
        print(f"Structural Analysis Workflow: Created {len(agents)} agents")
        return {"agents": agents}

    def build_agents(self, state: State):
        for agent in state['agents']:
            agent.build_workflow()
        #print(f"Structural Analysis Workflow: Built {len(state['agents'])} agents")
        return {"agents": state['agents']}

    def run_agents(self, state: State):
        start_time = time.perf_counter()
        stop_event = threading.Event()

        def _print_elapsed():
            while not stop_event.is_set():
                elapsed = time.perf_counter() - start_time
                print(f"\rStructural Analysis Workflow: Running {len(state['agents'])} agents... {elapsed:.1f}s", end="")
                stop_event.wait(1)

        thread = threading.Thread(target=_print_elapsed, daemon=True)
        thread.start()
        async def _run():
            run_tasks = [asyncio.create_task(agent.run()) for agent in state['agents']]
            await asyncio.gather(*run_tasks)

        asyncio.run(_run())
        stop_event.set()
        thread.join()
        elapsed = time.perf_counter() - start_time
        print(f"\rStructural Analysis Workflow: Running {len(state['agents'])} agents completed in {elapsed:.2f}s")
        return {"agents": state['agents']}

    def build_workflow(self):
        # Build workflow
        workflow = StateGraph(self.State)
        workflow.add_node("list_source_code_files", self.list_source_code_files)
        workflow.add_node("create_agents", self.create_agents)
        workflow.add_node("build_agents", self.build_agents)
        workflow.add_node("run_agents", self.run_agents)

        workflow.add_edge(START, "list_source_code_files")
        workflow.add_edge("list_source_code_files", "create_agents")
        workflow.add_edge("create_agents", "build_agents")
        workflow.add_edge("build_agents", "run_agents")
        workflow.add_edge("run_agents", END)
        # Compile workflow
        self.workflow = workflow.compile()

    async def run(self):
        print(f"Structural Analysis Workflow: Run workflow")
        await self.workflow.ainvoke({
            "read_directory": self.read_directory,
            "file_batch_size": self.file_batch_size,
            "write_directory": self.write_directory,
            "model": self.model,
            "model_provider": self.model_provider,
            "api_key": self.api_key,
        })
        self.status = True