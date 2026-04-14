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
                        "description": "The path of the source code file",
                    },
                    "file_contributors": {
                        "type": "string",
                        "description": "The summary of the contributors of the file",
                    },
                    "file_analysis": {
                        "type": "string",
                        "description": "The detailed summary of what the file and its functions do",
                    },
                },
                "required": ["file_path", "file_analysis"],
            },
        },
    },
    "required": ["files"],
}


def get_file_analyzer_prompt_header(batch_size: int) -> str:
    """Opening instructions for batch file analysis; per-file blocks are appended by the agent."""
    return f"""
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
