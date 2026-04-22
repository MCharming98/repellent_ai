import argparse
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
    For each row, create ``issues/<repo>/<issue_id>/`` with ``issue_details.json``,
    ``bench_config.json`` (``instance_id`` and ``commit_hash`` from ``base_commit``),
    and a plain ``commit_hash`` file.

    Existing files are left unchanged (each path is written only if it does not
    already exist as a file). If all three outputs already exist, the row is skipped
    without re-reading or rebuilding JSON.
    """
    for example in _iter_dataset_rows(dataset):
        repo = example.get("repo") or ""
        instance_id = example.get("instance_id") or ""
        if not repo or "/" not in repo or not instance_id:
            continue
        _, repo, issue_id = parse_instance_id(instance_id)
        issue_dir = issues_root.joinpath(repo, issue_id)
        issue_details_path = issue_dir / "issue_details.json"
        bench_config_path = issue_dir / "bench_config.json"
        commit_hash_path = issue_dir / "commit_hash"

        if (
            issue_details_path.is_file()
            and bench_config_path.is_file()
            and commit_hash_path.is_file()
        ):
            print(f"Skipping issue {instance_id} (outputs already exist)")
            continue

        print(f"Exporting issue {instance_id}")
        issue_dir.mkdir(parents=True, exist_ok=True)

        problem_statement = example.get("problem_statement") or ""
        hints_text = example.get("hints_text") or ""
        if not isinstance(problem_statement, str):
            problem_statement = str(problem_statement)
        if not isinstance(hints_text, str):
            hints_text = str(hints_text)

        if not issue_details_path.is_file():
            details = issue_details_from_swe_bench(problem_statement, hints_text)
            with open(issue_details_path, "w", encoding="utf-8") as f:
                json.dump(details, f, indent=2, ensure_ascii=False)
                f.write("\n")

        base_commit = example.get("base_commit")
        commit_str = base_commit.strip() if isinstance(base_commit, str) else ""
        bench_config = {"instance_id": instance_id, "commit_hash": commit_str}
        if not bench_config_path.is_file():
            with open(bench_config_path, "w", encoding="utf-8") as f:
                json.dump(bench_config, f, indent=2, ensure_ascii=False)
                f.write("\n")

        if not commit_hash_path.is_file():
            commit_hash_path.write_text(commit_str + "\n", encoding="utf-8")


def export_eval_template(dataset: Union[DatasetDict, Dataset], dataset_name: str) -> Path:
    """
    Build an empty SWE-bench ``predictions`` JSON (a list of dicts) for every row.

    Each entry has ``instance_id``, ``model_name_or_path``, and ``model_patch`` (the
    latter two empty strings). Writes ``bench-predictions/<dataset_name>.json`` under
    the repository root (``dataset_name`` may contain ``/``; it is sanitized for the
    filesystem).
    """
    bench_predictions_root = (
        Path(__file__).resolve().parent.parent.parent / "bench-predictions"
    )
    bench_predictions_root.mkdir(parents=True, exist_ok=True)
    safe_stem = dataset_name.replace("/", "__").replace("\\", "__")
    out_path = bench_predictions_root / f"{safe_stem}.json"

    seen: Set[str] = set()
    rows: List[Dict[str, str]] = []
    for example in _iter_dataset_rows(dataset):
        instance_id = example.get("instance_id") or ""
        if not isinstance(instance_id, str) or not instance_id.strip():
            continue
        if instance_id in seen:
            continue
        seen.add(instance_id)
        rows.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": "",
                "model_patch": "",
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SWE-bench issue artifacts from a Hugging Face dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=("default", "lite", "verified"),
        default="default",
        help=(
            "Dataset variant: default -> SWE-bench, lite -> SWE-bench_Lite, "
            "verified -> SWE-bench_Verified."
        ),
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional dataset split to load (e.g. dev, test).",
    )
    parser.add_argument(
        "--issues-root",
        default=str(Path(__file__).resolve().parent.parent.parent / "issues"),
        help="Output root for issue folders. Default: <repo_root>/issues",
    )
    parser.add_argument(
        "--clone-repos",
        action="store_true",
        help="Also clone all repos referenced by the selected dataset/split.",
    )
    parser.add_argument(
        "--projects-root",
        default=str(Path(__file__).resolve().parent.parent.parent / "projects"),
        help="Destination root used with --clone-repos. Default: <repo_root>/projects",
    )
    parser.add_argument(
        "--export-eval-template",
        action="store_true",
        help="Also create bench-predictions/<sanitized_dataset_name>.json.",
    )
    args = parser.parse_args()

    dataset_map = {
        "default": "SWE-bench/SWE-bench",
        "lite": "SWE-bench/SWE-bench_Lite",
        "verified": "SWE-bench/SWE-bench_Verified",
    }
    dataset_name = dataset_map[args.dataset]

    split = args.split.strip() if isinstance(args.split, str) else ""
    if split:
        dataset = load_dataset(dataset_name, split=split)
    else:
        dataset = load_dataset(dataset_name)

    if args.clone_repos:
        repo_set = repo_set_from_dataset(dataset)
        projects_root = Path(args.projects_root).resolve()
        for owner, repo in sorted(repo_set):
            print(f"Cloning {owner}/{repo}")
            dest = projects_root / repo
            if dest.exists() and any(dest.iterdir()):
                print(f"skip (already present): {owner}/{repo}")
                continue
            url = f"https://github.com/{owner}/{repo}"
            clone_github_repo(url, str(dest))

    issues_root = Path(args.issues_root).resolve()
    export_swe_bench_issues(dataset, issues_root)

    if args.export_eval_template:
        template_name = args.dataset if not split else f"{args.dataset}_{split}"
        out_path = export_eval_template(dataset, template_name)
        print(f"Wrote eval template: {out_path}")


if __name__ == "__main__":
    main()