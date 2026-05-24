"""Git and GitHub operations."""

import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

GITHUB_API_BASE = "https://api.github.com"

ISSUE_FIELDS = ("number", "title", "url", "state", "body", "comments_url", "events_url", "html_url")


def parse_github_repo_url(url: str) -> Optional[Tuple[str, str]]:
    """
    Parse owner and repo name from a GitHub URL.

    Supports formats:
        - https://github.com/owner/repo
        - http://github.com/owner/repo
        - github.com/owner/repo
        - https://github.com/owner/repo.git
        - https://github.com/owner/repo/
        - git@github.com:owner/repo.git

    Args:
        url: A GitHub repository URL or identifier.

    Returns:
        Tuple of (owner, repo) if valid, None otherwise.
    """
    url = url.strip().rstrip("/")
    if not url:
        return None

    # git@github.com:owner/repo.git
    ssh_match = re.match(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", url)
    if ssh_match:
        return (ssh_match.group(1), ssh_match.group(2))

    # Ensure we have something that looks like a host
    if "github.com" not in url:
        return None

    # Add scheme if missing for urllib parsing
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urllib.parse.urlparse(url)
    if parsed.hostname and "github.com" in parsed.hostname:
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(parts) >= 2:
            repo = parts[-1].removesuffix(".git")
            owner = parts[-2]
            return (owner, repo)
    return None


def _github_clone_url(url: str) -> str:
    """Normalize a GitHub repo URL or web path into an ``https`` clone URL ending in ``.git``."""
    s = url.strip().rstrip("/")
    if not s:
        raise ValueError("Repository URL is empty")
    if s.startswith("git@github.com:"):
        return s if s.endswith(".git") else s + ".git"
    parsed = parse_github_repo_url(s)
    if parsed:
        owner, repo = parsed
        return f"https://github.com/{owner}/{repo}.git"
    if "github.com" in s:
        if not s.startswith(("http://", "https://")):
            s = "https://" + s
        return s if s.endswith(".git") else s + ".git"
    raise ValueError(f"Unrecognized GitHub repository URL: {url!r}")


def checkout(commit_hash: str) -> None:
    """
    Check out a git ref in the current working directory.

    Args:
        commit_hash: Git ref to check out (branch, tag, or SHA).
            Special input ``latest`` restores the previous HEAD: tries ``git switch -``,
            then ``git checkout ORIG_HEAD`` if the reflog has no ``@{-1}``.

    Raises:
        RuntimeError: If ``git checkout`` fails.
    """
    ref = commit_hash.strip()
    if not ref:
        return

    if ref.lower() == "latest":
        checkout_process = subprocess.run(
            ["git", "switch", "-"],
            capture_output=True,
            text=True,
        )
    else:
        checkout_process = subprocess.run(
            ["git", "checkout", ref],
            capture_output=True,
            text=True,
        )
    if checkout_process.returncode != 0:
        err = (checkout_process.stderr or checkout_process.stdout or "").strip()
        raise RuntimeError(f"git checkout {ref!r} failed: {err}")


def clone_github_repo(url: str, path: str) -> None:
    """
    Clone a GitHub repository into ``path``.

    Args:
        url: Repository URL or GitHub web path (https, ``git@github.com:...``, or ``owner/repo``-style).
        path: Destination directory for the clone (must not exist or must be empty).
    Raises:
        ValueError: If ``url`` cannot be turned into a clone URL.
        FileExistsError: If ``path`` exists and is not an empty directory.
        RuntimeError: If ``git clone`` fails.
    """
    target = Path(path).expanduser().resolve()
    if target.exists():
        if target.is_file() or any(target.iterdir()):
            raise FileExistsError(f"Clone destination exists and is not empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    clone_url = _github_clone_url(url)
    clone = subprocess.run(
        ["git", "clone", clone_url, str(target)],
        capture_output=True,
        text=True,
    )
    if clone.returncode != 0:
        err = (clone.stderr or clone.stdout or "").strip()
        raise RuntimeError(f"git clone failed: {err}")

    return


def pull_from_repo(repo_path: str | None = None) -> None:
    """
    Run ``git pull`` in a repository.

    Args:
        repo_path: Git repository root. If omitted, uses the current working directory.

    Raises:
        RuntimeError: If ``git pull`` fails.
    """
    kwargs: dict[str, Any] = {"capture_output": True, "text": True}
    if repo_path is not None:
        kwargs["cwd"] = str(Path(repo_path).expanduser().resolve())

    pull = subprocess.run(["git", "pull"], **kwargs)
    if pull.returncode != 0:
        err = (pull.stderr or pull.stdout or "").strip()
        raise RuntimeError(f"git pull failed: {err}")


def extract_issue_fields(
    issues: Union[List[Dict[str, Any]], str, Path],
) -> List[Dict[str, Any]]:
    """
    Extract url, comments_url, events_url, number, title, and state from issues JSON.

    Args:
        issues: Either a list of issue dicts, or a path to a JSON file containing
                an array of issues.

    Returns:
        List of dicts with keys: url, comments_url, events_url, number, title, state.
    """
    if isinstance(issues, (str, Path)):
        with open(issues, "r", encoding="utf-8") as f:
            issues = json.load(f)

    if not isinstance(issues, list):
        raise TypeError(f"Expected list of issues, got {type(issues)}")

    return [{k: issue.get(k) for k in ISSUE_FIELDS} for issue in issues]


# Matches ![Image](url) and [Image](url) only
_IMAGE_MARKDOWN_PATTERN = re.compile(
    r"!?\[Image\]\(([^)]+)\)"
)


# Markdown links to GitHub user-uploaded files (logs, dumps, etc.), not only [Image](...).
_USER_ATTACHMENT_LINK_PATTERN = re.compile(
    r"\[([^\]]*)\]\((https://github\.com/user-attachments/(?:files|assets)/[^)\s]+)\)",
    re.IGNORECASE,
)


def extract_github_user_attachment_links(text: str) -> List[Tuple[str, str]]:
    """
    Extract (link label, URL) pairs for github.com/user-attachments/files|assets/... links.

    Used for crash logs and other non-image uploads linked as ``[name.ext](url)`` in issue bodies
    or comments.

    Args:
        text: Markdown (issue body, comment bodies, etc.).

    Returns:
        De-duplicated list of (label, url); label may be empty.
    """
    if not text:
        return []
    seen: set[str] = set()
    out: List[Tuple[str, str]] = []
    for m in _USER_ATTACHMENT_LINK_PATTERN.finditer(text):
        label, url = m.group(1).strip(), m.group(2).strip()
        if url in seen:
            continue
        seen.add(url)
        out.append((label or url, url))
    return out


def extract_image_markdown(text: str) -> List[str]:
    """
    Extract image URLs from markdown where the link text is exactly "Image".

    Matches ![Image](url) and [Image](url) (e.g. GitHub user-attachments).

    Args:
        text: Markdown string to parse.

    Returns:
        List of URL strings for each match.
    """
    if not text:
        return []
    return [m.group(1) for m in _IMAGE_MARKDOWN_PATTERN.finditer(text)]


def parse_github_issue_url(url: str) -> Tuple[str, str, int]:
    """
    Parse a GitHub issue URL to extract owner, repo, and issue number.

    Supports both web URLs and API URLs:
        - https://github.com/owner/repo/issues/123
        - https://api.github.com/repos/owner/repo/issues/123
        - https://api.github.com/repos/owner/repo/issues/123/comments

    Returns:
        Tuple of (owner, repo, issue_number).

    Raises:
        ValueError: If URL format is not recognized.
    """
    web_pattern = r"github\.com/([^/]+)/([^/]+)/issues/(\d+)"
    api_pattern = r"api\.github\.com/repos/([^/]+)/([^/]+)/issues/(\d+)"

    for pattern in (web_pattern, api_pattern):
        match = re.search(pattern, url)
        if match:
            return match.group(1), match.group(2), int(match.group(3))

    raise ValueError(f"Could not parse GitHub issue URL: {url}")


def fetch_issue_comments(
    owner: str,
    repo: str,
    issue_number: int,
    per_page: int = 100,
    token: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch comments from a GitHub issue via the GitHub Issues Comments API.

    Args:
        owner: Repository owner (username or organization).
        repo: Repository name.
        issue_number: Issue number.
        per_page: Number of comments per page (max 100).
        token: GitHub API token (optional; unauthenticated if omitted or empty).

    Returns:
        List of comment dicts with user.login and body.
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues/{issue_number}/comments"
    api_token = token or ""

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Repellent-AI",
    }
    if api_token:
        headers["Authorization"] = f"token {api_token}"

    params = {"per_page": min(per_page, 100), "page": 1}
    all_comments = []

    while True:
        try:
            query_params = urllib.parse.urlencode(params)
            full_url = f"{url}?{query_params}"
            req = urllib.request.Request(full_url, headers=headers)

            with urllib.request.urlopen(req) as response:
                if response.getcode() != 200:
                    break
                comments = json.loads(response.read().decode("utf-8"))

            extracted = [
                {
                    "user": {"login": c.get("user", {}).get("login", "")},
                    "body": c.get("body", ""),
                }
                for c in comments
            ]
            all_comments.extend(extracted)

            if len(comments) < params["per_page"]:
                break

            params["page"] += 1

        except urllib.error.HTTPError:
            break
        except urllib.error.URLError:
            break
        except Exception:
            break

    return all_comments


def save_issue_details_to_json(title: str, body: str, comments: List[Dict], filename: str) -> None:
    """Save issue details to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "title": title,
            "body": body,
            "comments": comments,
        }, f, indent=2, ensure_ascii=False)


def get_contributors(git_repo_path: str, file_path: str) -> List[str]:
    """
    Get contributors for a file using git shortlog.

    Args:
        git_repo_path (str): Path to the git repository root.
        file_path (str): Path to the file (relative to repo root).

    Returns:
        List[str]: List of contributor lines from git shortlog (commits, name, email).
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


def get_closed_issues(
    owner: str,
    repo: str,
    state: str = "closed",
    per_page: int = 100,
    max_issues: Optional[int] = None,
    token: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch closed issues from a GitHub repository via the GitHub Issues API.

    Args:
        owner: Repository owner (username or organization).
        repo: Repository name.
        state: Issue state (default: "closed").
        per_page: Number of issues per page (max 100).
        max_issues: Maximum number of issues to fetch (None for all).
        token: GitHub API token (optional; unauthenticated if omitted or empty).

    Returns:
        List of issue dictionaries.
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues"
    api_token = token or ""

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Repellent-AI",
    }
    if api_token:
        headers["Authorization"] = f"token {api_token}"

    params = {"state": state, "per_page": min(per_page, 100), "page": 1}
    all_issues = []

    while True:
        try:
            query_params = urllib.parse.urlencode({
                "state": state,
                "per_page": min(per_page, 100),
                "page": params["page"],
            })
            full_url = f"{url}?{query_params}"
            req = urllib.request.Request(full_url, headers=headers)

            with urllib.request.urlopen(req) as response:
                if response.getcode() != 200:
                    break
                issues = json.loads(response.read().decode("utf-8"))

            actual_issues = [i for i in issues if "pull_request" not in i]
            all_issues.extend(actual_issues)

            if max_issues and len(all_issues) >= max_issues:
                all_issues = all_issues[:max_issues]
                break

            if len(issues) < params["per_page"]:
                break

            params["page"] += 1

        except urllib.error.HTTPError as e:
            print(f"get_closed_issues: HTTP Error: {e.code}: {e.reason}")
            break
        except urllib.error.URLError as e:
            print(f"get_closed_issues: URL Error: {e.reason}")
            break
        except Exception as e:
            print(f"get_closed_issues: Error: {e}")
            break

    return all_issues


def format_issue(issue: Dict) -> str:
    """Format an issue dictionary as a readable string."""
    created = datetime.strptime(issue["created_at"], "%Y-%m-%dT%H:%M:%SZ")
    closed = (
        datetime.strptime(issue["closed_at"], "%Y-%m-%dT%H:%M:%SZ")
        if issue.get("closed_at")
        else None
    )

    labels = ", ".join([lb["name"] for lb in issue.get("labels", [])])
    assignees = ", ".join([a["login"] for a in issue.get("assignees", [])])

    result = f"""
Issue #{issue["number"]}: {issue["title"]}
URL: {issue["html_url"]}
State: {issue["state"]}
Created: {created.strftime("%Y-%m-%d %H:%M:%S")}
"""

    if closed:
        result += f"Closed: {closed.strftime("%Y-%m-%d %H:%M:%S")}\n"
    if labels:
        result += f"Labels: {labels}\n"
    if assignees:
        result += f"Assignees: {assignees}\n"
    if issue.get("body"):
        body_preview = issue["body"][:200].replace("\n", " ")
        result += f"Description: {body_preview}...\n"

    result += "-" * 80
    return result


def save_issues_to_json(issues: List[Dict], filename: str) -> None:
    """Save issues to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)


def save_issues_to_text(issues: List[Dict], filename: str) -> None:
    """Save issues to a text file using format_issue."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Closed Issues Report\n")
        f.write(f"Total Issues: {len(issues)}\n")
        f.write("=" * 80 + "\n\n")
        for issue in issues:
            f.write(format_issue(issue) + "\n")
