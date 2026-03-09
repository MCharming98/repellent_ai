"""Utility functions for file and directory operations."""

import os
import subprocess
from pathlib import Path
from typing import List, Literal
from langchain.tools import tool

def get_current_working_directory() -> str:
    """
    Get the current working directory path.
    
    Returns:
        str: The absolute path of the current working directory.
    
    Example:
        >>> cwd = get_current_working_directory()
        >>> print(cwd)
        '/Users/chenmeng/bug_agent'
    """
    # print("get_current_working_directory()")
    # Get the current working directory as an absolute path
    return os.getcwd()


def list_files(path: str) -> List[str]:
    """
    List all files and directories in the specified path.
    
    Args:
        path (str): The directory path to list files from. Can be absolute or relative.
    
    Returns:
        str: A formatted string containing the list of files and directories,
             one per line. Returns an error message if the path doesn't exist
             or is not a directory.
    
    Example:
        >>> files = list_files('/Users/chenmeng/bug_agent')
        >>> print(files)
        'analyze_engineers.py\\nAntennaPod\\narchive\\n...'
    """
    # print(f"list_files(path={path})")
    # Convert string path to Path object for easier manipulation
    dir_path = Path(path)
    
    # Check if the path exists
    if not dir_path.exists():
        return f"Error: Path '{path}' does not exist."
    
    # Check if the path is a directory
    if not dir_path.is_dir():
        return f"Error: '{path}' is not a directory."
    
    # Get all items in the directory and sort them
    items = sorted(dir_path.iterdir())
    
    # Format items as a string, one per line
    # Show directories with a trailing slash for clarity
    file_list = []
    for item in items:
        if item.is_dir():
            file_list.append(f"{item.name}/")
        else:
            file_list.append(item.name)
    
    return file_list

def list_source_files_recursive(path: str) -> List[str]:
    """
    Recursively list all source code file paths in the specified directory and all subdirectories.
    Only source code files are included, excluding binary files, build configs, logs, docs, etc.
    
    Args:
        path (str): The directory path to recursively list files from.
                   Can be absolute or relative.
    
    Returns:
        str: A formatted string containing the list of all source code file paths,
             one per line. Paths are relative to the input directory.
             Returns an error message if the path doesn't exist
             or is not a directory.
    
    Example:
        >>> files = list_files_recursive('/Users/chenmeng/bug_agent')
        >>> print(files)
        'analyze_engineers.py\\nAntennaPod/src/Main.java\\nAntennaPod/src/Utils.kt\\n...'
    """
    # print(f"list_source_files_recursive(path={path})")
    # Define source code file extensions
    SOURCE_EXTENSIONS = {
        # Python
        '.py', '.pyx', '.pyi',
        # Java and JVM languages
        '.java', '.kt', '.scala', '.groovy', '.clj', '.cljs',
        # JavaScript/TypeScript
        '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs',
        # C/C++
        '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx', '.hh',
        # Objective-C/Swift
        '.m', '.mm', '.swift',
        # Go
        '.go',
        # Rust
        '.rs',
        # Ruby
        '.rb', '.rake',
        # PHP
        '.php', '.phtml',
        # C# / .NET
        '.cs', '.fs', '.fsx', '.vb',
        # Shell scripts
        '.sh', '.bash', '.zsh', '.fish', '.ps1',
        # SQL
        '.sql',
        # R
        '.r', '.R',
        # Other languages
        '.dart', '.lua', '.pl', '.pm', '.tcl', '.vim', '.el', '.erl', '.ex', '.exs',
        '.jl', '.nim', '.cr', '.v', '.zig', '.odin', '.hs', '.ml', '.mli', '.fsi',
        '.d', '.pas', '.pp', '.ada', '.adb', '.ads',
    }
    
    # Define directory patterns to exclude (build artifacts, dependencies, etc.)
    EXCLUDED_DIRS = {
        '__pycache__', 'node_modules', '.git', '.svn', '.hg', 'build', 'dist', 'target',
        'bin', 'obj', 'out', '.gradle', '.idea', '.vscode', '.vs', '.settings',
        '.classpath', '.project', 'venv', 'env', '.venv', '.env', '.pytest_cache',
        '.mypy_cache', '.tox', '.coverage', '.nyc_output', 'coverage', '.cache',
        '.parcel-cache', '.next', '.nuxt', '.output', '.temp', '.tmp', 'logs', 'Logs',
        '.DS_Store', '.idea', '.vscode', '.vs', 'vendor', 'bower_components',
        'gradle', '.gradle', 'build', 'out', 'target', 'dist', 'lib', 'libs',
        'node_modules', '.npm', '.yarn', 'pods', '.cocoapods', 'DerivedData',
        'Pods', '.build', '.swiftpm', 'Carthage', '.carthage',
    }
    
    # Define file extensions to exclude (binary files, build artifacts, configs, docs, logs)
    EXCLUDED_EXTENSIONS = {
        # Binary executables
        '.exe', '.dll', '.so', '.dylib', '.bin',
        # Compiled code
        '.class', '.o', '.obj', '.pyc', '.pyo', '.pyd', '.jar', '.war', '.ear',
        # Archives
        '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar', '.xz',
        # Logs
        '.log',
        # Documentation
        '.md', '.rst', '.txt', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.bmp', '.tiff', '.webp', '.heic',
        # Media
        '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.wav', '.ogg', '.flac', '.aac',
        # Fonts
        '.woff', '.woff2', '.ttf', '.otf', '.eot',
        # Databases
        '.db', '.sqlite', '.sqlite3', '.db3', '.mdb', '.accdb',
        # Build configs and lock files
        '.lock', '.min.js', '.min.css', '.map', '.bundle', '.chunk',
        # IDE and editor files
        '.swp', '.swo', '.swn', '.bak', '.tmp',
    }
    
    # Convert string path to Path object for easier manipulation
    dir_path = Path(path)
    
    # Check if the path exists
    if not dir_path.exists():
        return f"Error: Path '{path}' does not exist."
    
    # Check if the path is a directory
    if not dir_path.is_dir():
        return f"Error: '{path}' is not a directory."
    
    # Recursively get all source code files in the directory and subdirectories
    file_paths = []
    # Iterate through all files recursively
    for file_path in dir_path.rglob('*'):
        # Only process files, not directories
        if not file_path.is_file():
            continue
        
        # Get relative path from the input directory
        relative_path = file_path.relative_to(dir_path)
        
        # Check if any parent directory should be excluded
        should_exclude = False
        for part in relative_path.parts:
            if part in EXCLUDED_DIRS:
                should_exclude = True
                break
        
        if should_exclude:
            continue
        
        # Get file extension (lowercase for case-insensitive matching)
        file_ext = file_path.suffix.lower()
        
        # Exclude files with excluded extensions
        if file_ext in EXCLUDED_EXTENSIONS:
            continue
        
        # Only include files with source code extensions
        if file_ext in SOURCE_EXTENSIONS:
            file_paths.append(str(relative_path))
    
    # Sort the file paths for consistent output
    file_paths.sort()
    
    return file_paths

def list_cwd_source_files_recursive() -> List[str]:
    """
    List all files and directories in the current working directory and all subdirectories.
    """
    return list_source_files_recursive(os.getcwd())

def read_file(path: str) -> str:
    """
    Read the contents of a file and return it as a string.
    
    Args:
        path (str): The file path to read from. Can be absolute or relative.
    
    Returns:
        str: The complete contents of the file as a string.
             Returns an error message if the file doesn't exist,
             is not readable, or is a directory.
    
    Raises:
        IOError: If the file cannot be read due to permission issues.
    
    Example:
        >>> content = read_file('/Users/chenmeng/bug_agent/analyze_engineers.py')
        >>> print(content[:50])
        'import os\nfrom pathlib import Path\n...'
    """
    #print(f"read_file(path={path})")
    # Convert string path to Path object
    file_path = Path(path)
    
    # Check if the path exists
    if not file_path.exists():
        return f"Error: File '{path}' does not exist."
    
    # Check if the path is a file (not a directory)
    if not file_path.is_file():
        return f"Error: '{path}' is not a file."
    
    # Read the file content with UTF-8 encoding
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try reading as binary and return a message
        return f"Error: File '{path}' contains binary data and cannot be read as text."
    except IOError as e:
        return f"Error: Cannot read file '{path}': {str(e)}"

def write_to_file(path: str, content: str, mode: Literal['w', 'a'] = 'a') -> None:
    """
    Write content to a file, create the file if it doesn't exist.
    If mode is 'a', the content will be appended to the file.
    If mode is 'w', the file will be overwritten.

    Args:
        path (str): The file path to write to. Can be absolute or relative.
                   Parent directories will be created if they don't exist.
        content (str): The content to write to the file.
    
    Returns:
        None
    
    Raises:
        IOError: If the file cannot be written due to permission issues.
    
    Example:
        >>> write_to_file('/Users/chenmeng/bug_agent/test.txt', 'Hello, World!')
        >>> read_file('/Users/chenmeng/bug_agent/test.txt')
        'Hello, World!'
    """
    # print(f"write_to_file(path={path}, content={len(content)} bytes, mode={mode})")
    # Convert string path to Path object
    file_path = Path(path)
    
    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the file if it doesn't exist
    if not file_path.exists():
        file_path.touch()

    # Write content to the file with UTF-8 encoding
    try:
        with open(file_path, mode, encoding='utf-8') as f:
            f.write(content)
    except IOError as e:
        raise IOError(f"Cannot write to file '{path}': {str(e)}")

def get_contributors(git_repo_path: str, file_path: str) -> List[str]:
    """
    Get contributors for a file or directory using git shortlog.
    
    Args:
        path (str): The file or directory path to get contributors for.
    
    Returns:
        List[str]: List of contributor lines from git shortlog, split by newline.
    """
    result = subprocess.run(
        ['git', 'shortlog', '-n', '-s', '-e', '--', file_path],
        capture_output=True,
        text=True,
        cwd=git_repo_path
    )
    if result.returncode != 0:
        return []
    return result.stdout.strip().split('\n') if result.stdout.strip() else []