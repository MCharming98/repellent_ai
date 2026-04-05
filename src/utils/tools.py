import files
from typing import Literal
from langchain.tools import tool

@tool
def read_file_tool(path: str) -> str:
    """
    Read the contents of a file and return it as a string.

    Args:
        path (str): The file path to read from. Can be absolute or relative.

    Returns:
        str: The file contents, or an error message if the file doesn't exist or cannot be read.
    """
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
    return files.write_to_file(path, content, mode)