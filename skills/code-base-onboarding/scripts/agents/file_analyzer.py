import time
from pathlib import PurePath
from typing import Any, List

from typing_extensions import NotRequired, TypedDict
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from utils import *
from langchain.agents.structured_output import ToolStrategy

from constants.file_analyzer_constants import FILE_ANALYSIS_SCHEMA, get_file_analyzer_prompt_header
from utils.langchain import aggregate_token_usage_from_messages, merge_token_usage_totals, is_http_429, _RATE_LIMIT_WAIT_SECONDS

def file_analysis_entry_to_markdown(entry: dict[str, Any]) -> str:
    """
    Render one ``FILE_ANALYSIS_SCHEMA`` file object (a single ``files[]`` item) as markdown.
    """
    lines: list[str] = []
    fp = str(entry.get("file_path", "")).strip()
    lines.append(f"## `{fp}`")
    lines.append("")
    summary = str(entry.get("file_summary", "")).strip()
    lines.append("### Summary")
    lines.append(summary if summary else "_N/A_")
    lines.append("")
    lines.append("### Contributors")
    primary = str(entry.get("primary_contributor", "")).strip()
    secondary = str(entry.get("secondary_contributor", "")).strip()
    lines.append(f"- **Primary:** {primary if primary else '_N/A_'}")
    if secondary:
        lines.append(f"- **Secondary:** {secondary}")
    lines.append("")
    classes = entry.get("classes")
    if isinstance(classes, list) and classes:
        lines.append("### Classes")
        for c in classes:
            if not isinstance(c, dict):
                continue
            cn = str(c.get("class_name", "")).strip()
            cs = str(c.get("class_summary", "")).strip()
            if cn:
                lines.append(f"- **`{cn}`**: {cs if cs else '_N/A_'}")
        lines.append("")
    functions = entry.get("functions")
    if isinstance(functions, list) and functions:
        lines.append("### Functions")
        for fn in functions:
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("function_name", "")).strip()
            fsum = str(fn.get("function_summary", "")).strip()
            if name:
                lines.append(f"- **`{name}`**: {fsum if fsum else '_N/A_'}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class FileAnalyzer:
    def __init__(self, id: str, read_directory: str, files: List[str], write_directory: str, model: str, model_provider: str, api_key: str):
        self.id = id
        self.read_directory = read_directory
        self.files = files
        self.write_directory = write_directory
        self.write_status = False
        model_kwargs = {"api_key": api_key}
        # Default to Google Developer API instead of Vertex AI
        if model_provider == "google_genai":
            model_kwargs["google_api_key"] = api_key
        self.agent = create_agent(
            model=init_chat_model(model, model_provider=model_provider, **model_kwargs),
            response_format=ToolStrategy(FILE_ANALYSIS_SCHEMA)
        )

    class State(TypedDict):
        id: str
        read_directory: str
        write_directory: str
        source_code_files: list[str]
        file_analysis: dict[str, str]
        write_status: bool
        token_usage: NotRequired[dict[str, int]]

    def analyze_files(self, state: State):
        print(f"File analyzer #{state['id']}: analyzing {len(state['source_code_files'])} files")
        file_analysis = {}
        source_files = state['source_code_files']
        batch_size = len(source_files)
        token_usage: dict[str, int] = dict(
            state.get("token_usage")
            or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        )

        for batch_start in range(0, len(source_files), batch_size):
            batch = source_files[batch_start : batch_start + batch_size]
            contributors = {}
            for file in batch:
                contributors[file] = get_contributors(state['read_directory'], file)
            prompt = get_file_analyzer_prompt_header(batch_size)
            for file_path in batch:
                file_content: str = read_file(str(PurePath(state["read_directory"], file_path)))
                file_contributors: list[str] = contributors[file_path]
                prompt += f"""
                - File path: {file_path}
                - File contributors(number of commits, name, account): {file_contributors}
                - File content: {file_content}

                """
            try:
                result = self.agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]}
                )
            except Exception as e:
                if is_http_429(e):
                    print(
                        f"File analyzer #{state['id']}: HTTP 429; waiting {_RATE_LIMIT_WAIT_SECONDS:.0f}s "
                    )
                    time.sleep(_RATE_LIMIT_WAIT_SECONDS)
                    return {"file_analysis": file_analysis, "token_usage": token_usage}
                print(f"Error: File analyzer #{state['id']}: {e}")
                return {"write_status": False, "token_usage": token_usage}
            usage = aggregate_token_usage_from_messages(result.get("messages") or [])
            if usage:
                token_usage = merge_token_usage_totals(token_usage, usage)
            structured = result.get("structured_response") or {}
            for file in structured.get("files") or []:
                if not isinstance(file, dict):
                    continue
                file_path = str(file.get("file_path", "")).strip()
                if not file_path:
                    print("Warning: skipping file entry with empty file_path")
                    continue
                file_analysis[file_path] = file_analysis_entry_to_markdown(file)
        print(f"File analyzer #{state['id']}: completed")
        return {"file_analysis": file_analysis, "token_usage": token_usage}

    def write_analysis_to_file(self, state: State):
        print(f"File analyzer #{state['id']}: writing {len(state['file_analysis'])} file analysis to {state['write_directory']}")
        write_directory = state['write_directory']
        file_analysis = state['file_analysis']
        output_path = PurePath(write_directory, "file_analysis.md")
        for file_path in sorted(file_analysis.keys()):
            markdown_block = file_analysis[file_path]
            write_to_file(str(output_path), markdown_block + "\n", 'a')
        return {"write_status": True, "token_usage": state.get("token_usage", {})}

    def fetch_missing_files(self, state: State):
        """If any paths in this agent's batch lack analysis, narrow ``source_code_files`` and retry."""
        expected = state['source_code_files']
        fa = state.get("file_analysis") or {}
        if len(fa) >= len(expected):
            return {}
        missing = [f for f in expected if f not in fa]
        if not missing:
            return {}
        print(
            f"File analyzer #{state['id']}: {len(missing)} missing file(s) of {len(expected)}; "
            "re-running analyze_files"
        )
        return {"source_code_files": missing, "file_analysis": {}}

    def check_for_missing_files(self, state: State) -> bool:
        """Check if there are any missing files in the file analysis."""
        return len(state.get("file_analysis") or {}) < len(state['source_code_files'])

    def build_workflow(self):
        # print(f"Agent {state['id']}: build workflow")
        self.workflow = StateGraph(self.State)
        self.workflow.add_node("analyze_files", self.analyze_files)
        self.workflow.add_node("write_analysis_to_file", self.write_analysis_to_file)
        self.workflow.add_node("fetch_missing_files", self.fetch_missing_files)

        self.workflow.add_edge(START, "analyze_files")
        self.workflow.add_edge("analyze_files", "write_analysis_to_file")
        self.workflow.add_conditional_edges(
            "write_analysis_to_file",
            self.check_for_missing_files,
            {True: "fetch_missing_files", False: END},
        )
        self.workflow.add_edge("fetch_missing_files", "analyze_files")
        self.workflow = self.workflow.compile()

    async def run(self):
        # print(f"Agent {state['id']}: run workflow")
        final_state = await self.workflow.ainvoke({
            "id": self.id,
            "read_directory": self.read_directory,
            "write_directory": self.write_directory,
            "source_code_files": self.files,
            "file_analysis": {},
            "write_status": False,
        })
        self.write_status = final_state.get("write_status", False)
        return final_state