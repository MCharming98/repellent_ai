"""Git and GitHub operations."""

import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Dict, List, Optional

GITHUB_API_BASE = "https://api.github.com"


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
        token: GitHub API token. Uses GITHUB_TOKEN env var if not provided.

    Returns:
        List of issue dictionaries.
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues"
    api_token = token or os.getenv("GITHUB_TOKEN", "")

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

        except urllib.error.HTTPError:
            break
        except urllib.error.URLError:
            break
        except Exception:
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
