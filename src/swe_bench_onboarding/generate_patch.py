"""Generate a SWE-bench patch via the Cursor headless Agent CLI and record it in predictions JSON.

Uses **print mode** (``-p`` / ``--print``) with ``--force`` so the agent applies edits in
non-interactive runs, as described in the Cursor docs:
https://cursor.com/docs/cli/headless
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

# Allow running as `python src/.../generate_patch.py` without PYTHONPATH=src
_src_root = Path(__file__).resolve().parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

ISSUE_DETAILS_JSON = "issue_details.json"
BENCH_CONFIG_JSON = "bench_config.json"


def _status(msg: str, *, quiet: bool) -> None:
    if not quiet:
        print(f"[generate_patch] {msg}", file=sys.stderr, flush=True)


def _build_agent_prompt(issue_details_text: str, diagnosis_text: str | None = None) -> str:
    prompt = (
        "You are fixing a bug in this git repository at the checked-out commit.\n"
        "Read the issue details and implement the minimal correct fix by editing "
        "source files. Do not run `git commit`.\n"
        "After your edits, the harness will record the change as a git unified diff.\n\n"
        f"Issue details:\n{issue_details_text}\n"
    )
    if diagnosis_text:
        prompt += f"\nDiagnosis context:\n{diagnosis_text}\n"
    return prompt


def _find_agent_executable() -> str:
    env_bin = os.environ.get("CURSOR_AGENT_BIN", "").strip()
    if env_bin and Path(env_bin).expanduser().is_file():
        return str(Path(env_bin).expanduser())
    found = shutil.which("agent")
    if found:
        return found
    raise RuntimeError(
        "Cursor Agent CLI not found: expected `agent` on PATH "
        "(set CURSOR_AGENT_BIN to a full path if needed)."
    )


def _run_agent(repo_path: Path, prompt: str) -> subprocess.CompletedProcess[str]:
    """Invoke Cursor Agent in headless print mode so file changes are applied (not only proposed).

    Per https://cursor.com/docs/cli/headless — combine ``-p``/``--print`` with ``--force`` (or
    ``--yolo``) for scripted edits; ``--trust`` is required for headless use of ``--workspace``.
    """
    agent = _find_agent_executable()
    return subprocess.run(
        [
            agent,
            "-p",
            "--trust",
            "--force",
            "--workspace",
            str(repo_path),
            prompt,
        ],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _git(repo_path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _git_diff_patch(repo_path: Path) -> str:
    p = _git(repo_path, "diff", "--no-color", check=False)
    if p.returncode != 0:
        raise RuntimeError(
            f"git diff failed: {p.stderr or p.stdout}"
        )
    return p.stdout or ""


def _git_discard_worktree(repo_path: Path) -> None:
    """Reset tracked files to ``HEAD`` so a later ``git switch`` is not blocked by local edits."""
    p = _git(repo_path, "reset", "--hard", "HEAD", check=False)
    if p.returncode != 0:
        raise RuntimeError(
            f"git reset --hard HEAD failed: {p.stderr or p.stdout}"
        )


def _update_bench_predictions(
    bench_predictions_path: Path,
    instance_id: str,
    model_name: str,
    model_patch: str,
) -> None:
    raw = bench_predictions_path.read_text(encoding="utf-8")
    data: Union[List[Dict[str, Any]], Dict[str, Any]] = json.loads(raw)

    if isinstance(data, dict):
        pred = data.get(instance_id)
        if pred is None:
            raise ValueError(
                f"No prediction entry for instance_id {instance_id!r} in {bench_predictions_path}"
            )
        if not isinstance(pred, dict):
            raise ValueError(f"Prediction for {instance_id!r} must be a dict")
        pred["model_name_or_path"] = model_name
        pred["model_patch"] = model_patch
    elif isinstance(data, list):
        updated = False
        for row in data:
            if isinstance(row, dict) and row.get("instance_id") == instance_id:
                row["model_name_or_path"] = model_name
                row["model_patch"] = model_patch
                updated = True
                break
        if not updated:
            raise ValueError(
                f"No prediction entry for instance_id {instance_id!r} in {bench_predictions_path}"
            )
    else:
        raise ValueError("bench predictions file must be a JSON list or dict")

    bench_predictions_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def generate_patch_for_bench_instance(
    repo_path: Union[Path, str],
    issue_dir_path: Union[Path, str],
    bench_predictions_path: Union[Path, str],
    diagnosis_path: Union[Path, str, None] = None,
    *,
    model_name: str = "cursor",
    quiet: bool = False,
) -> str:
    """
    Check out the bench commit in ``repo_path``, run Cursor Agent to implement the fix,
    capture the result as ``git diff`` text, and write ``model_patch`` / ``model_name_or_path``
    for the corresponding ``instance_id`` in ``bench_predictions_path``.

    ``issue_dir_path`` must contain ``issue_details.json`` and ``bench_config.json`` (see onboarding export layout).
    After the diff is captured, runs ``git reset --hard HEAD`` to discard agent edits, then
    ``git switch -`` so the repo returns to the previous branch/HEAD.

    Requires the ``agent`` binary on ``PATH`` (install: https://cursor.com/docs/cli/installation).
    Set ``CURSOR_API_KEY`` for authentication in scripts
    (https://cursor.com/docs/cli/reference/authentication). Override the binary with
    ``CURSOR_AGENT_BIN`` if needed.

    Set ``quiet`` to True to suppress progress lines on stderr.
    """
    repo_path = Path(repo_path).resolve()
    issue_dir = Path(issue_dir_path).resolve()
    bench_predictions_path = Path(bench_predictions_path).resolve()
    diagnosis_file = Path(diagnosis_path).resolve() if diagnosis_path else None

    if not issue_dir.is_dir():
        raise NotADirectoryError(f"issue_dir_path is not a directory: {issue_dir}")

    issue_details_path = issue_dir / ISSUE_DETAILS_JSON
    bench_config_path = issue_dir / BENCH_CONFIG_JSON
    if not issue_details_path.is_file():
        raise FileNotFoundError(
            f"Missing {ISSUE_DETAILS_JSON} under issue_dir_path: {issue_details_path}"
        )
    if not bench_config_path.is_file():
        raise FileNotFoundError(
            f"Missing {BENCH_CONFIG_JSON} under issue_dir_path: {bench_config_path}"
        )

    issue_details = json.loads(issue_details_path.read_text(encoding="utf-8"))
    bench_config = json.loads(bench_config_path.read_text(encoding="utf-8"))
    diagnosis_text: str | None = None
    if diagnosis_file is not None:
        if not diagnosis_file.is_file():
            raise FileNotFoundError(f"diagnosis_path is not a file: {diagnosis_file}")
        diagnosis_text = diagnosis_file.read_text(encoding="utf-8")

    instance_id = bench_config.get("instance_id")
    commit_hash = bench_config.get("commit_hash")
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("bench_config must contain a non-empty string 'instance_id'")
    if not isinstance(commit_hash, str) or not commit_hash.strip():
        raise ValueError("bench_config must contain a non-empty string 'commit_hash'")

    if not repo_path.is_dir():
        raise FileNotFoundError(f"repo_path is not a directory: {repo_path}")

    ch = commit_hash.strip()
    _status(
        f"instance_id={instance_id.strip()!r} repo={repo_path} issue_dir={issue_dir}",
        quiet=quiet,
    )
    if diagnosis_file is not None:
        _status(f"using diagnosis file: {diagnosis_file}", quiet=quiet)
    _status(f"bench commit {ch[:12]}…", quiet=quiet)

    _status("git checkout -f <bench commit>", quiet=quiet)
    checkout = _git(repo_path, "checkout", "-f", ch, check=False)
    if checkout.returncode != 0:
        raise RuntimeError(
            f"git checkout failed: {checkout.stderr or checkout.stdout}"
        )
    _status("checkout complete", quiet=quiet)

    try:
        prompt = _build_agent_prompt(str(issue_details), diagnosis_text=diagnosis_text)
        _status(
            "running Cursor Agent (-p --force); this may take a long time …",
            quiet=quiet,
        )
        proc = _run_agent(repo_path, prompt)
        if proc.returncode != 0:
            raise RuntimeError(
                "Cursor Agent exited with non-zero status "
                f"{proc.returncode}.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        _status("agent finished; capturing git diff", quiet=quiet)
        patch = _git_diff_patch(repo_path).strip("\n")
        if not patch:
            raise RuntimeError(
                "git diff is empty after agent run; no patch was produced. "
                f"Agent stdout:\n{proc.stdout}\nAgent stderr:\n{proc.stderr}"
            )

        _status(f"writing predictions → {bench_predictions_path}", quiet=quiet)
        _update_bench_predictions(
            bench_predictions_path,
            instance_id.strip(),
            model_name,
            patch + "\n",
        )
        _status("predictions file updated", quiet=quiet)
        return patch + "\n"
    finally:
        # Patch is already captured in memory / JSON; drop working-tree edits so
        # ``git switch -`` is not rejected with "local changes would be overwritten".
        _status("git reset --hard HEAD (discard agent edits)", quiet=quiet)
        try:
            _git_discard_worktree(repo_path)
        except RuntimeError as e:
            warnings.warn(str(e), RuntimeWarning, stacklevel=2)
        _status("git switch - (restore previous HEAD)", quiet=quiet)
        sw = _git(repo_path, "switch", "-", check=False)
        if sw.returncode != 0:
            warnings.warn(
                f"git switch - failed (working tree may still be at bench commit): "
                f"{sw.stderr or sw.stdout}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            _status("restored previous branch/HEAD", quiet=quiet)


def main() -> int:
    """CLI entrypoint (see ``if __name__ == '__main__'``). Run from repo root with ``PYTHONPATH=src`` if needed."""
    parser = argparse.ArgumentParser(
        description=(
            "Check out a bench commit, run Cursor headless Agent (-p --force), record the git diff "
            "in predictions JSON, then git switch - back to the previous HEAD."
        ),
        epilog=(
            "Headless CLI: https://cursor.com/docs/cli/headless — set CURSOR_API_KEY for "
            "non-interactive auth."
        ),
    )
    parser.add_argument(
        "--repo_path",
        type=Path,
        help="Path to the cloned git repository",
    )
    parser.add_argument(
        "--issue_dir",
        type=Path,
        help=(
            f"Directory containing {ISSUE_DETAILS_JSON} and {BENCH_CONFIG_JSON} "
            "(e.g. issues/<repo>/<issue_id>/ from onboarding export)"
        ),
    )
    parser.add_argument(
        "--bench_predictions_path",
        type=Path,
        help="Path to SWE-bench predictions JSON to update",
    )
    parser.add_argument(
        "--diagnosis_path",
        type=Path,
        default=None,
        help="Optional path to diagnosis markdown/text file included in the prompt",
    )
    parser.add_argument(
        "--model-name",
        default="cursor",
        help="Value written to model_name_or_path (default: %(default)s)",
    )
    parser.add_argument(
        "--print-patch",
        action="store_true",
        help="Print the unified diff to stdout after a successful run",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress [generate_patch] status lines on stderr",
    )
    args = parser.parse_args()
    try:
        patch = generate_patch_for_bench_instance(
            args.repo_path,
            args.issue_dir,
            args.bench_predictions_path,
            diagnosis_path=args.diagnosis_path,
            model_name=args.model_name,
            quiet=args.quiet,
        )
    except (OSError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    preds = args.bench_predictions_path.resolve()
    if not args.quiet:
        print(f"[generate_patch] done — recorded patch in {preds}", file=sys.stderr)
    if args.print_patch:
        print(patch, end="" if patch.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
