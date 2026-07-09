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
                    "file_summary": {
                        "type": "string",
                        "description": "A high-level summary of what the file is responsible for",
                    },
                    "primary_contributor": {
                        "type": "string",
                        "description": "The name of the primary contributor of the file",
                    },
                    "secondary_contributor": {
                        "type": "string",
                        "description": "The name of the secondary contributor of the file",
                    },
                    "classes": {
                        "type": "array",
                        "description": "The list of classes in the file",
                        "items": {
                            "type": "object",
                            "description": "The entry of a class",
                            "properties": {
                                "class_name": {
                                    "type": "string",
                                    "description": "The name of the class",
                                },
                                "class_summary": {
                                    "type": "string",
                                    "description": "A high-level summary of what the class is responsible for",
                                },
                            },
                            "required": ["class_name", "class_summary"],
                        },
                    },
                    "functions": {
                        "type": "array",
                        "description": "The list of functions in the file",
                        "items": {
                            "type": "object",
                            "description": "The entry of a function",
                            "properties": {
                                "function_name": {
                                    "type": "string",
                                    "description": "The name of the function",
                                },
                                "function_summary": {
                                    "type": "string",
                                    "description": "A high-level summary of what the function is responsible for",
                                },
                            },
                            "required": ["function_name", "function_summary"],
                        },
                    },
                },
                "required": ["file_path", "file_summary", "primary_contributor"],
            },
        },
    },
    "required": ["files"],
}


def get_file_analyzer_prompt_header(batch_size: int) -> str:
    """Opening instructions for batch file analysis; per-file blocks are appended by the agent."""
    return f"""
                    Task: Given {batch_size} source code file paths and contents, your task is to read through the source code for each file and analze the following:
                    - A high-level summary of what the file is responsible for
                    - Identify one primary contributor, including their name and account, with the most commits
                    - If applicable, identify one secondary contributor, including their name and account, with the second most commits
                    - If applicable, list all the classes, including class names and their high-level responsibilities
                    - If applicable, list all the functions, including function names and their high-level responsibilities

                    Input:
                    """
