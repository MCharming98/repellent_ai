import json
import sys
from pathlib import Path

# Allow running as `python src/.../onboarding.py` without PYTHONPATH=src
_src_root = Path(__file__).resolve().parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from typing import Any, Dict, List, Set, Tuple, Union

from datasets import Dataset, DatasetDict, load_dataset

from utils import clone_github_repo


def repo_set_from_dataset(dataset: Union[DatasetDict, Dataset]) -> Set[Tuple[str, str]]:
    """Collect unique ``owner/repo`` pairs from all rows (all splits if a ``DatasetDict``)."""
    if isinstance(dataset, DatasetDict):
        split_datasets = dataset.values()
    else:
        split_datasets = (dataset,)
    return {
        tuple(example["repo"].split("/", 1))
        for split_dataset in split_datasets
        for example in split_dataset
        if isinstance(example.get("repo"), str) and "/" in example["repo"]
    }


def issue_details_from_swe_bench(
    problem_statement: str,
    hints_text: str,
) -> Dict[str, Any]:
    """
    Build an ``issue_details``-shaped dict (see ``issues/.../issue_details.json``).

    The problem statement is split on the first newline: title before, body after.
    Hints are stored as a single synthetic comment with ``login`` set to ``N/A``.
    """
    if "\n" in problem_statement:
        title, body = problem_statement.split("\n", 1)
    else:
        title, body = problem_statement, ""

    comments: List[Dict[str, Any]] = [
        {
            "user": {"login": "N/A"},
            "body": hints_text,
        }
    ]

    return {
        "title": title,
        "body": body,
        "comments": comments,
    }


def parse_instance_id(instance_id: str) -> Tuple[str, str, str]:
    """
    Parse a SWE-bench ``instance_id`` of the form ``owner__repo-<issue_id>``.

    The issue number is separated from ``owner__repo`` by the last ``-`` in the string
    (so repo names may contain hyphens).

    Example: ``django__django-11099`` → ("django", "django", "11099").
    """
    s = instance_id.strip()
    if "__" not in s:
        raise ValueError(f"instance_id must contain '__' (owner__repo): {instance_id!r}")
    head, issue_part = s.rsplit("-", 1)
    if "__" not in head:
        raise ValueError(f"instance_id must match owner__repo-<issue_id>: {instance_id!r}")
    owner, repo = head.split("__", 1)
    if not owner or not repo or not issue_part:
        raise ValueError(f"Invalid instance_id: {instance_id!r}")
    return owner, repo, issue_part


def _iter_dataset_rows(dataset: Union[DatasetDict, Dataset]):
    if isinstance(dataset, DatasetDict):
        for split_dataset in dataset.values():
            for row in split_dataset:
                yield row
    else:
        for row in dataset:
            yield row


def export_swe_bench_issues(
    dataset: Union[DatasetDict, Dataset],
    issues_root: Path,
) -> None:
    """
    For each row, create ``issues/<repo>/<issue_id>/`` with ``issue_details.json``
    and a plain ``commit_hash`` file (content from ``base_commit``).
    """
    for example in _iter_dataset_rows(dataset):
        repo = example.get("repo") or ""
        instance_id = example.get("instance_id") or ""
        if not repo or "/" not in repo or not instance_id:
            continue
        print(f"Exporting issue {instance_id}")
        _, repo, issue_id = parse_instance_id(instance_id)
        issue_dir = issues_root.joinpath(repo, issue_id)
        issue_dir.mkdir(parents=True, exist_ok=True)

        problem_statement = example.get("problem_statement") or ""
        hints_text = example.get("hints_text") or ""
        if not isinstance(problem_statement, str):
            problem_statement = str(problem_statement)
        if not isinstance(hints_text, str):
            hints_text = str(hints_text)

        details = issue_details_from_swe_bench(problem_statement, hints_text)
        issue_details_path = issue_dir / "issue_details.json"
        with open(issue_details_path, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
            f.write("\n")

        base_commit = example.get("base_commit")
        commit_str = base_commit.strip() if isinstance(base_commit, str) else ""
        (issue_dir / "commit_hash").write_text(commit_str + "\n", encoding="utf-8")


# Load main dataset
sbf = load_dataset('SWE-bench/SWE-bench')
# Load verified variant
sbv = load_dataset('SWE-bench/SWE-bench_Verified', split='test')
# Load lite variant
sbl = load_dataset('SWE-bench/SWE-bench_Lite')

# Fetch all repositories and clone them to the projects directory
"""
repo_set = repo_set_from_dataset(sbl)
_projects_root = Path(__file__).resolve().parent.parent.parent / "projects"
for owner, repo in sorted(repo_set):
    print(f"Cloning {owner}/{repo}")
    dest = _projects_root / repo
    if dest.exists() and any(dest.iterdir()):
        print(f"skip (already present): {owner}/{repo}")
        continue
    url = f"https://github.com/{owner}/{repo}"
    clone_github_repo(url, str(dest))
"""

# Export SWE-bench issues to the issues directory
_issues_root = Path(__file__).resolve().parent.parent.parent / "issues"
export_swe_bench_issues(sbl, _issues_root)
