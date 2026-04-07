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
    print(f"list_files: listing files in {path}")
    return files.list_files(path)

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
    print(f"list_source_files_recursive_tool: listing source files in {path}")
    return files.list_source_files_recursive(path)

@tool
def read_file_tool(path: str) -> str:
    """
    Read the contents of a file and return it as a string.

    Args:
        path (str): The file path to read from. Can be absolute or relative.

    Returns:
        str: The file contents, or an error message if the file doesn't exist or cannot be read.
    """
    print(f"read_file_tool: reading file {path}")
    return files.read_file(path)

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