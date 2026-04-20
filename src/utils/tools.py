from . import files
from typing import Literal
from langchain.tools import tool

@tool
def list_files_tool(path: str) -> list[str]:
    """
    List all files and directories in the specified path.

    Args:
        path (str): The directory path to list files from. Can be absolute or relative.

    Returns:
    """
    files_list = files.list_files(path)
    print(f"list_files_tool: listing {len(files_list)} files in {path}")
    return files_list

@tool
def list_source_files_recursive_tool(path: str) -> list[str]:
    """
    Recursively list all source code file paths in the specified directory and subdirectories.
    Only source code files are included, excluding binary files, build configs, logs, docs, etc.

    Args:
        path (str): The directory path to recursively list files from. Can be absolute or relative.

    Returns:
        List[str]: List of relative file paths. Returns an error message list if the path
        doesn't exist or is not a directory.
    """
    files_list = files.list_source_files_recursive(path)
    print(f"list_source_files_recursive_tool: listing {len(files_list)} source files in {path}")
    return files_list

@tool
def read_file_tool(path: str) -> str:
    """
    Read the contents of a file and return it as a string.

    Args:
        path (str): The file path to read from. Can be absolute or relative.

    Returns:
        str: The file contents, or an error message if the file doesn't exist or cannot be read.
    """
    file_contents = files.read_file(path)
    if file_contents.startswith("Error:"):
        print(f"read_file_tool: error — {path} — {file_contents}")
    else:
        print(f"read_file_tool: ok — {path} ({len(file_contents)} chars)")
    return file_contents

@tool
def write_to_file_tool(path: str, content: str, mode: Literal['w', 'a'] = 'a') -> str:
    """
    Write content to a file. Creates parent directories if needed.

    Args:
        path (str): The file path to write to. Can be absolute or relative.
        content (str): The content to write to the file.
        mode (Literal['w', 'a']): The mode to write the file in. 'w' for overwrite, 'a' for append.

    Returns:
        str: The file path, or an error message if the file cannot be written to.
    """
    print(f"write_to_file_tool: writing to file {path}")
    return files.write_to_file(path, content, mode)