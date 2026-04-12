"""File and directory operations."""

import base64
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, List, Literal, Optional

def get_current_working_directory() -> str:
    """
    Get the current working directory path.

    Returns:
        str: The absolute path of the current working directory.
    """
    return os.getcwd()


def list_files(path: str) -> List[str]:
    """
    List all files and directories in the specified path.

    Args:
        path (str): The directory path to list files from. Can be absolute or relative.

    Returns:
        List[str]: List of file and directory names. Returns an error message string
        if the path doesn't exist or is not a directory.
    """
    dir_path = Path(path)

    if not dir_path.exists():
        return [f"Error: Path '{path}' does not exist."]

    if not dir_path.is_dir():
        return [f"Error: '{path}' is not a directory."]

    items = sorted(dir_path.iterdir())
    file_list = []
    for item in items:
        if item.is_dir():
            file_list.append(f"{item.name}/")
        else:
            file_list.append(item.name)

    return file_list


def list_source_files_recursive(path: str) -> List[str]:
    """
    Recursively list all source code file paths in the specified directory and subdirectories.
    Only source code files are included, excluding binary files, build configs, logs, docs, etc.

    Args:
        path (str): The directory path to recursively list files from. Can be absolute or relative.

    Returns:
        List[str]: List of relative file paths. Returns an error message list if the path
        doesn't exist or is not a directory.
    """
    SOURCE_EXTENSIONS = {
        '.py', '.pyx', '.pyi',
        '.java', '.kt', '.scala', '.groovy', '.clj', '.cljs',
        '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs',
        '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx', '.hh',
        '.m', '.mm', '.swift', '.go', '.rs',
        '.rb', '.rake', '.php', '.phtml',
        '.cs', '.fs', '.fsx', '.vb',
        '.sh', '.bash', '.zsh', '.fish', '.ps1',
        '.sql', '.r', '.R',
        '.dart', '.lua', '.pl', '.pm', '.tcl', '.vim', '.el', '.erl', '.ex', '.exs',
        '.jl', '.nim', '.cr', '.v', '.zig', '.odin', '.hs', '.ml', '.mli', '.fsi',
        '.d', '.pas', '.pp', '.ada', '.adb', '.ads',
    }

    EXCLUDED_DIRS = {
        '__pycache__', 'node_modules', '.git', '.svn', '.hg', 'build', 'dist', 'target',
        'bin', 'obj', 'out', '.gradle', '.idea', '.vscode', '.vs', '.settings',
        '.classpath', '.project', 'venv', 'env', '.venv', '.env', '.pytest_cache',
        '.mypy_cache', '.tox', '.coverage', '.nyc_output', 'coverage', '.cache',
        '.parcel-cache', '.next', '.nuxt', '.output', '.temp', '.tmp', 'logs', 'Logs',
        '.DS_Store', 'vendor', 'bower_components', 'gradle', 'lib', 'libs',
        '.npm', '.yarn', 'pods', '.cocoapods', 'DerivedData', 'Pods', '.build',
        '.swiftpm', 'Carthage', '.carthage',
    }

    EXCLUDED_EXTENSIONS = {
        '.exe', '.dll', '.so', '.dylib', '.bin',
        '.class', '.o', '.obj', '.pyc', '.pyo', '.pyd', '.jar', '.war', '.ear',
        '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar', '.xz', '.log',
        '.md', '.rst', '.txt', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.bmp', '.tiff', '.webp', '.heic',
        '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.wav', '.ogg', '.flac', '.aac',
        '.woff', '.woff2', '.ttf', '.otf', '.eot',
        '.db', '.sqlite', '.sqlite3', '.db3', '.mdb', '.accdb',
        '.lock', '.min.js', '.min.css', '.map', '.bundle', '.chunk',
        '.swp', '.swo', '.swn', '.bak', '.tmp',
    }

    dir_path = Path(path)

    if not dir_path.exists():
        return [f"Error: Path '{path}' does not exist."]

    if not dir_path.is_dir():
        return [f"Error: '{path}' is not a directory."]

    file_paths = []
    for file_path in dir_path.rglob('*'):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(dir_path)

        should_exclude = any(part in EXCLUDED_DIRS for part in relative_path.parts)
        if should_exclude:
            continue

        file_ext = file_path.suffix.lower()
        if file_ext in EXCLUDED_EXTENSIONS:
            continue
        if file_ext in SOURCE_EXTENSIONS:
            file_paths.append(str(relative_path))

    file_paths.sort()
    return file_paths


def list_cwd_source_files_recursive() -> List[str]:
    """List all source code files in the current working directory recursively."""
    return list_source_files_recursive(os.getcwd())

def read_file(path: str) -> str:
    """
    Read the contents of a file and return it as a string.

    Args:
        path (str): The file path to read from. Can be absolute or relative.

    Returns:
        str: The file contents, or an error message if the file doesn't exist or cannot be read.
    """
    file_path = Path(path)

    if not file_path.exists():
        return f"Error: File '{path}' does not exist."

    if not file_path.is_file():
        return f"Error: '{path}' is not a file."

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        return f"Error: File '{path}' contains binary data and cannot be read as text."
    except IOError as e:
        return f"Error: Cannot read file '{path}': {str(e)}"


def write_to_file(path: str, content: Any, mode: Literal['w', 'a'] = 'a') -> None:
    """
    Write content to a file. Creates parent directories as needed.
    If mode is 'a', append; if 'w', overwrite.
    Non-string content is coerced with str().
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        file_path.touch()

    try:
        with open(file_path, mode, encoding='utf-8') as f:
            if isinstance(content, str):
                text = content
            else:
                print(
                    f"Warning: Content is not a string but {type(content)}, converting to string"
                )
                text = str(content)
            f.write(text)
    except IOError as e:
        raise IOError(f"Cannot write to file '{path}': {str(e)}") from e


_MAX_ATTACHMENT_TEXT_BYTES = 512 * 1024


def fetch_url_text(
    url: str,
    *,
    max_bytes: int = _MAX_ATTACHMENT_TEXT_BYTES,
    timeout: float = 60.0,
) -> Optional[str]:
    """
    Download a URL and return text (UTF-8), truncating to ``max_bytes``.

    Skips responses whose ``Content-Type`` is clearly ``image/*`` (use
    :func:`fetch_image_as_data_url` for images).

    Returns:
        Decoded text, or ``None`` on failure or if skipped as non-text.
    """
    if not url:
        return None
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "RepellentAI/1.0 (issue attachment fetch)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_ct = resp.headers.get("Content-Type", "")
            ct = raw_ct.split(";")[0].strip().lower()
            if ct.startswith("image/"):
                return None
            data = resp.read(max_bytes + 1)
    except (urllib.error.URLError, OSError) as e:
        print(f"Warning: could not fetch attachment text {url!r}: {e}")
        return None
    truncated = len(data) > max_bytes
    chunk = data[:max_bytes] if truncated else data
    try:
        text = chunk.decode("utf-8")
    except UnicodeDecodeError:
        text = chunk.decode("utf-8", errors="replace")
    if truncated:
        text += "\n\n... [truncated: attachment exceeded byte limit]"
    return text


_DEFAULT_FETCH_MAX_BYTES = 20 * 1024 * 1024


def fetch_url_bytes(
    url: str,
    *,
    max_bytes: int = _DEFAULT_FETCH_MAX_BYTES,
    timeout: float = 120.0,
) -> Optional[tuple[bytes, str]]:
    """
    Download a URL and return ``(raw_bytes, content_type)`` where ``content_type`` is the
    header value without parameters.

    Reads at most ``max_bytes`` bytes (truncated if the response is larger).
    """
    if not url:
        return None
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "RepellentAI/1.0 (issue attachment fetch)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_ct = resp.headers.get("Content-Type", "") or ""
            ct = raw_ct.split(";")[0].strip()
            data = resp.read(max_bytes + 1)
    except (urllib.error.URLError, OSError) as e:
        print(f"Warning: could not fetch URL bytes {url!r}: {e}")
        return None
    if len(data) > max_bytes:
        data = data[:max_bytes]
        print(f"Warning: truncated download to {max_bytes} bytes: {url!r}")
    return (data, ct)


_MAX_IMAGE_BYTES = 20 * 1024 * 1024
def _sniff_image_mime(data: bytes) -> Optional[str]:
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def is_image_bytes(data: bytes, content_type: str) -> bool:
    """True if bytes are an image by Content-Type or magic-byte sniffing."""
    ct = content_type.split(";")[0].strip().lower()
    if ct.startswith("image/"):
        return True
    return _sniff_image_mime(data) is not None


def is_text_bytes(data: bytes, content_type: str) -> bool:
    """Heuristic: treat as text for logs/markdown/json and readable UTF-8."""
    ct = content_type.split(";")[0].strip().lower()
    if ct.startswith("text/"):
        return True
    if ct in ("application/json", "application/xml", "application/javascript"):
        return True
    if "json" in ct or "xml" in ct:
        return True
    if is_image_bytes(data, content_type):
        return False
    try:
        s = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if not s:
        return False
    printable = sum(1 for c in s if c.isprintable() or c in "\n\r\t")
    return printable / len(s) > 0.88


def guess_attachment_extension(data: bytes, content_type: str) -> str:
    """Pick a filename extension (with leading dot) for saved attachment bytes."""
    ct = content_type.split(";")[0].strip().lower()
    if ct.startswith("image/"):
        sub = ct.split("/", 1)[-1]
        return {
            "png": ".png",
            "jpeg": ".jpg",
            "jpg": ".jpg",
            "gif": ".gif",
            "webp": ".webp",
        }.get(sub, ".img")
    sniff = _sniff_image_mime(data)
    if sniff:
        return {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }.get(sniff, ".img")
    if is_text_bytes(data, content_type):
        return ".txt"
    return ".bin"


def fetch_image_as_data_url(url: str, timeout: float = 30.0) -> Optional[str]:
    """
    Download an image URL and return a data: URL with an explicit image/* MIME type.

    GitHub user-attachments URLs often have no file extension; Gemini then fails with
    ``Unsupported MIME type:`` when given a bare https URL.
    """
    if not url:
        return None
    if url.startswith("data:"):
        return url
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "RepellentAI/1.0 (issue image fetch)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            raw_ct = resp.headers.get("Content-Type", "")
            ct = raw_ct.split(";")[0].strip().lower()
    except (urllib.error.URLError, OSError) as e:
        print(f"Warning: could not fetch image {url!r}: {e}")
        return None
    if len(data) > _MAX_IMAGE_BYTES:
        print(f"Warning: skipping image over {_MAX_IMAGE_BYTES} bytes: {url!r}")
        return None
    mime: Optional[str] = None
    if ct.startswith("image/"):
        mime = ct
    if mime is None:
        mime = _sniff_image_mime(data)
    if mime is None:
        print(f"Warning: could not determine image MIME type for {url!r}")
        return None
    b64 = base64.standard_b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"
