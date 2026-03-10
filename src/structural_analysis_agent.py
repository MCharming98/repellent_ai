import time
from pathlib import PurePath
from typing import List
from typing_extensions import TypedDict
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from utils import *
from langchain.agents.structured_output import ToolStrategy

FILE_ANALYSIS_SCHEMA = {
    "type": "object",
    "description": "The summary of a file and its functions",
    "properties": {
        "files": {
            "type": "array",
            "description": "The array of files and their analysis",
            "items": {
                "type": "object",
                "description": "The entry of a file and its analysis",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the source code file"
                    },
                    "file_contributors": {
                        "type": "string",
                        "description": "The summary of the contributors of the file"
                    },
                    "file_analysis": {
                        "type": "string",
                        "description": "The detailed summary of what the file and its functions do"
                    },
                },
                "required": ["file_path", "file_analysis"]
            },
        },
    },
    "required": ["files"]
}

class StructuralAnalysisAgent:
    def __init__(self, id: str, read_directory: str, files: List[str], write_directory: str, model: str, model_provider: str, api_key: str):
        self.id = id
        self.read_directory = read_directory
        self.files = files
        self.write_directory = write_directory
        model_kwargs = {"api_key": api_key}
        if model_provider == "google_genai":
            model_kwargs["google_api_key"] = api_key
        self.agent = create_agent(
            model=init_chat_model(model, model_provider=model_provider, **model_kwargs),
            response_format=ToolStrategy(FILE_ANALYSIS_SCHEMA)
        )

    class State(TypedDict):
        read_directory: str
        write_directory: str
        source_code_files: list[str]
        file_analysis: dict[str, str]
        write_status: bool

    def analyze_files(self, state: State):
        start_time = time.perf_counter()
        print(f"Structural analysis agent #{self.id}: analyzing {len(state['source_code_files'])} files")
        file_analysis = {}
        source_files = state['source_code_files']
        batch_size = len(source_files)

        for batch_start in range(0, len(source_files), batch_size):
            batch = source_files[batch_start : batch_start + batch_size]
            contributors = {}
            for file in batch:
                contributors[file] = get_contributors(state['read_directory'], file)
            prompt = f""" 
                    Task: Given {batch_size} source code file paths and contents, your task is to read through the source code and do the following:
                    - Write your response in markdown format
                    - In one sentence, summarize the overall high-level responsibilities of the file itself
                    - A subsection for file contributors, including their name and account
                      - Identify one primary contributor with the most commits
                      - Identify one secondary contributor with the second most commits
                    - A subsection for functions, including function names and their responsibilities in a couple of words
                      - Wrap the subsection in h2(double #) format
                      - Omit the subsection if there is no function
                    - Create a new entry in the files array for each file
                    - Fill the file_path field with the file path in the entry, in h1(single #) format
                    - Fill the file_contributors field with the contributors summary in the entry
                    - Fill the file_analysis field with the analysis summary in the entry

                    Input:
                    """
            for file_path in batch:
                file_content: str = read_file(str(PurePath(state["read_directory"], file_path)))
                file_contributors: list[str] = contributors[file_path]
                prompt += f"""
                - File path: {file_path}
                - File contributors(number of commits, name, account): {file_contributors}
                - File content: {file_content}

                """
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            for file in result["structured_response"]["files"]:
                file_path = file.get("file_path", "")
                if "file_contributors" not in file:
                    print(f"Warning: File contributors missing for file {file_path}")
                if "file_analysis" not in file:
                    print(f"Warning: File analysis missing for file {file_path}")
                file_contributors = file.get("file_contributors", "")
                file_analysis_content = file.get("file_analysis", "")
                file_analysis[file_path] = file_contributors + "\n" + file_analysis_content
        elapsed = time.perf_counter() - start_time
        print(f"\nStructural analysis agent #{self.id}: completed in {elapsed:.2f}s")
        return {"file_analysis": file_analysis}

    def write_analysis_to_file(self, state: State):
        print(f"Structural analysis agent #{self.id}: writing {len(state['file_analysis'])} file analysis to {state['write_directory']}")
        write_directory = state['write_directory']
        file_analysis = state['file_analysis']
        for file_path, analysis in file_analysis.items():
            output_path = PurePath(write_directory, "file_analysis.md")
            write_to_file(str(output_path), f"{file_path}\n{analysis}\n\n", 'a')
        return {"write_status": True}

    def build_workflow(self):
        # print(f"Agent {self.id}: build workflow")
        self.workflow = StateGraph(self.State)
        self.workflow.add_node("analyze_files", self.analyze_files)
        self.workflow.add_node("write_analysis_to_file", self.write_analysis_to_file)

        self.workflow.add_edge(START, "analyze_files")
        self.workflow.add_edge("analyze_files", "write_analysis_to_file")
        self.workflow.add_edge("write_analysis_to_file", END)
        self.workflow = self.workflow.compile()

    async def run(self):
        # print(f"Agent {self.id}: run workflow")
        final_state = await self.workflow.ainvoke({
            "read_directory": self.read_directory,
            "write_directory": self.write_directory,
            "source_code_files": self.files,
        })
        return final_state