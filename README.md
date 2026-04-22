# Repellent AI

## What it is

**Repellent AI** is an autonomous analysis stack for software repositories. It uses LLM-backed agents to build structured **domain knowledge** from a codebase (how files fit together, business context, and who contributes where). On top of that knowledge, it can run **bug analysis** for a single GitHub issue: it triages the report, proposes hypotheses, and investigates them against the checked-out source and the onboarding artifacts.

The CLI lives in `src/main.py`. From the repository root, use `PYTHONPATH=src` so imports resolve.

## Setup

1. **Python** — Use an environment where project dependencies are installed (see your usual workflow or `requirements`/lockfiles if present).
2. **Configuration** — Copy or edit `config.yaml` at the repo root:
   - **`api_key`** — LLM provider key (literal or `$VAR` / `${VAR}`, e.g. `$GEMINI_API_KEY`).
   - **`github_token`** — GitHub token for API access (required for the **`analyze`** subcommand; literal or `$GITHUB_TOKEN`).
   - **`model_name`**, **`model_provider`**, **`model_context_window`** — Model and context limits used by the workflows.
3. **Optional** — `file_analysis_batch_size` (default `10` in code) controls batching during file analysis.

Run all examples from this repository’s root unless you note paths otherwise.

## CLI overview

```bash
PYTHONPATH=src python src/main.py onboard --repository <PATH_TO_CLONED_REPO>
PYTHONPATH=src python src/main.py analyze --url <GITHUB_ISSUE_URL> [--output-dir issues] [--source-dir ...] [--domain-knowledge ...]
PYTHONPATH=src python src/main.py analyze --path <LOCAL_ISSUE_DIR> [--source-dir ...] [--domain-knowledge ...]
```

- **`onboard`** — Builds domain knowledge for one local repository clone.
- **`analyze`** — Runs bug analysis for one issue (by URL or by local issue folder). Expects that repository’s onboarding output to already exist at the default **`domain_knowledge/<repo>/`** (see below).

---

## Onboarding workflow

**Purpose:** For a single project directory, run the full **onboarding** pipeline so later issue analysis has `file_analysis.md` and related summaries.

**Command:**

```bash
PYTHONPATH=src python src/main.py onboard --repository projects/MyProject
```

**What runs (in order):**

1. **File analysis** — Scans source files, runs parallel file-level agents, writes `file_analysis.md` under `domain_knowledge/<project_name>/`.
2. **Business analysis** — Reads file analysis; writes `business_analysis.md` (product context, CUJs, linked files).
3. **Contributor analysis** — Git-aware contributor picture; writes `contributor_analysis.md`.

`<project_name>` is the **basename** of the path passed to `--repository` (e.g. `projects/sqlfluff` → `domain_knowledge/sqlfluff/`).

**Outputs (under `domain_knowledge/<project_name>/`):**

| File | Role |
|------|------|
| `file_analysis.md` | Per-file structure and responsibilities (heaviest input for **analyze**) |
| `business_analysis.md` | Business and user-journey context |
| `contributor_analysis.md` | Contributor rollup |

Implementation: `workflows/onboarding_workflow.py` (orchestrates `file_analysis_workflow`, `business_analysis_workflow`, `contributor_analysis_workflow`).

---

## Analyze workflow

**Purpose:** For **one** issue, produce a **diagnosis** grounded in issue text/comments, domain knowledge, and the source tree. Used for triage and hypothesis exploration toward root cause.

**Prerequisites:**

- Onboarding has been run for that product so **`domain_knowledge/<repo>/file_analysis.md`** (and siblings) exist. Defaults assume the clone lives at **`projects/<repo>`** and knowledge at **`domain_knowledge/<repo>`**, where `<repo>` is the GitHub repository name parsed from the issue URL, or the parent folder name of the issue directory when using `--path` (e.g. `issues/sqlfluff/1625` → repo `sqlfluff`).
- **`analyze`** loads `api_key` and **`github_token`** from `config.yaml` (both required for this subcommand today).

**Start from a GitHub issue URL**

Fetches issue metadata and comments into the issues tree, then runs hypothesis generation and investigation.

```bash
PYTHONPATH=src python src/main.py analyze \
  --url https://github.com/owner/repo/issues/123 \
  --output-dir issues
```

- Issue payload is written under **`issues/<repo>/<issue_number>/`** (e.g. `issue_details.json`).
- **`diagnosis.md`** is created/updated in that folder (hypothesis generator writes it; investigator **appends** investigation sections).

**Start from a local issue directory**

Use this when you already have **`issue_details.json`** inside the issue folder (e.g. you copied it or fetched the issue elsewhere).

```bash
PYTHONPATH=src python src/main.py analyze --path issues/sqlfluff/1625
```

**Useful overrides**

| Flag | Meaning |
|------|--------|
| `--source-dir` | Root of the cloned repository (default: `projects/<repo>`). |
| `--domain-knowledge` | Directory with onboarding outputs (default: `domain_knowledge/<repo>`). |
| `--output-dir` | Base directory for **URL** mode only; fetched issues go under `<output-dir>/<repo>/<number>/`. Default: `issues`. |

Implementation: `workflows/bug_analysis_workflow.py` (LangGraph: load/parse issue → optional GitHub fetch → hypothesis generator → hypothesis investigator).

---

## Architecture (reference)

### Workflows

- **Onboarding** — `workflows/onboarding_workflow.py`: file → business → contributor analysis; outputs under `domain_knowledge/<project_name>/`.
- **File analysis** — `workflows/file_analysis_workflow.py` → `file_analysis.md`.
- **Business analysis** — `workflows/business_analysis_workflow.py` → `business_analysis.md`.
- **Contributor analysis** — `workflows/contributor_analysis_workflow.py` → `contributor_analysis.md`.
- **Bug / issue analysis** — `workflows/bug_analysis_workflow.py`; agents in `agents/hypothesis_generator.py` and `agents/hypothesis_investigator.py`.

### Agents and utilities

- **File analyzer** — `agents/file_analyzer.py` (batched, parallel file-level LLM summaries).
- **utils/** — Filesystem helpers (`utils/files`), git contributor helpers (`utils/git`), shared config (`utils/config.py`).

### Typical end-to-end flow

1. Clone the upstream repo under `projects/<name>/`.
2. Run **`onboard`** on that path.
3. Run **`analyze`** with `--url` (or `--path` if `issue_details.json` is already present).
4. Inspect **`issues/<repo>/<number>/diagnosis.md`** for the combined diagnosis and investigation.
