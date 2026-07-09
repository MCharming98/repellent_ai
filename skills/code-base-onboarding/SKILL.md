---
name: code-base-onboarding
description: >-
  Onboard a codebase into Repellent AI domain knowledge by cloning the repo,
  estimating LLM token usage, and running the onboarding workflow. Use when the
  user asks to onboard a repository, build domain knowledge, run file/business/
  contributor analysis, or prepare a codebase for bug analysis.
---

# Code Base Onboarding

Self-contained skill under `skills/code-base-onboarding/`. All code, config, clones, and outputs live inside this directory — no imports or paths from the parent repository.

## Layout

```
skills/code-base-onboarding/
├── SKILL.md
├── config.yaml
├── requirements.txt
├── projects/                # Cloned repositories (created at runtime)
├── domain_knowledge/        # Generated docs (created at runtime)
└── scripts/
    ├── _paths.py            # Skill-root paths + sys.path setup
    ├── main.py              # Onboarding CLI (OnboardingWorkflow entry)
    ├── clone_repo.py        # Step 2: clone GitHub repo
    ├── estimate_tokens.py   # Step 3: token estimate
    ├── run_onboarding.py    # Step 4: wrapper around main._run_onboard
    ├── workflows/           # LangGraph onboarding pipelines
    ├── agents/              # LLM agents (file analyzer)
    ├── constants/           # Prompts and schemas
    └── utils/               # Config, files, git, text, langchain helpers
```

All Python modules under `scripts/` are imported via `scripts/_paths.py`:

```python
from _paths import setup_imports
setup_imports()  # adds scripts/ to sys.path
```

CLI scripts call `setup_imports()` before importing `utils`, `workflows`, or `agents`.

## Workflow

Copy this checklist and track progress:

```
Task Progress:
- [ ] Step 0: Install dependencies and configure
- [ ] Step 1: Repository intake
- [ ] Step 2: Fetch target repo with git
- [ ] Step 3: Estimate token usage and get user confirmation
- [ ] Step 4: Run onboarding workflow and store domain knowledge
```

### Step 0: Install dependencies and configure

From `skills/code-base-onboarding/`:

```bash
pip install -r requirements.txt
```

Edit `config.yaml`:

| Key | Required | Purpose |
|-----|----------|---------|
| `api_key` | Yes | LLM provider key (literal or `$VAR` / `${VAR}`) |
| `model_name` | No | Default `gemini-3-flash-preview` |
| `model_provider` | No | Default `google_genai` |
| `model_context_window` | Yes | Max input context for the model |
| `file_analysis_batch_size` | No | Default `10` |

### Step 1: Repository intake

If the user does not specify which codebase to onboard, ask the user which code base to onboard.

Accept any of:

- GitHub URL (`https://github.com/owner/repo`, `owner/repo`, `git@github.com:owner/repo.git`)
- Local path to an existing clone (absolute, or relative to this skill root)

Derive the project name from the repo basename (e.g. `projects/sqlfluff` → `sqlfluff`).

### Step 2: Fetch target repo with git

Use git tools to fetch the target repo.

**If the user gave a GitHub URL or `owner/repo`:**

1. Clone into `projects/<repo_name>/` under this skill directory.
2. Run `scripts/clone_repo.py` (uses `scripts/utils/git.py`):

```bash
cd skills/code-base-onboarding
python3 scripts/clone_repo.py OWNER/REPO
```

Or use shell git directly:

```bash
git clone https://github.com/OWNER/REPO.git projects/REPO
```

**If the user gave a local path:**

- Verify it is a git repository (`<path>/.git` exists).
- Use that path as `--repository` (absolute or relative to this skill root).
- Optionally run `git pull` in that directory to refresh.

**Do not proceed** if the clone fails or the path is missing.

### Step 3: Estimate token usage and confirm with user

Before running the LLM-heavy onboarding pipeline, estimate the read token usage to read and analyze the entire code base with LLM, and ask the user to confirm token usage.

1. Ensure `config.yaml` is readable and `api_key` resolves (literal or `$VAR` / `${VAR}`).
2. Run `scripts/estimate_tokens.py` (same logic as `FileAnalysisWorkflow.estimate_file_token_count`):

```bash
python3 scripts/estimate_tokens.py projects/<repo>
```

3. Sum per-file estimates into a total **input read estimate**. Note:
   - Files larger than `model_context_window` are skipped during onboarding.
   - Actual LLM usage will be higher (prompts, outputs, business/contributor stages). Use ~1.5–2× the read estimate as a rough total budget, or report read tokens separately and note downstream stages add more.
4. Present the script summary (or equivalent table) to the user:

| Metric | Value |
|--------|-------|
| Repository path | `projects/<repo>` |
| Source files | N |
| Estimated read tokens | X |
| Model | `<model_name>` (`<model_provider>`) |
| Context window | `<model_context_window>` |

**Stop and wait for explicit user confirmation** before Step 4. If the user declines, do not run onboarding.

### Step 4: Run onboarding workflow and store domain knowledge

Use `scripts/workflows/onboarding_workflow.py` to generate the domain knowledge documents and store them.

```bash
python3 scripts/run_onboarding.py --repository projects/<repo_name>
```

Equivalent direct entry:

```bash
python3 scripts/main.py --repository projects/<repo_name>
```

`run_onboarding.py` calls `main._run_onboard`, which builds and runs `OnboardingWorkflow`. Stages, in order:

1. **File analysis** — `workflows/file_analysis_workflow.py` → `domain_knowledge/<project_name>/file_analysis.md`
2. **Business analysis** — `workflows/business_analysis_workflow.py` → `domain_knowledge/<project_name>/business_analysis.md`
3. **Contributor analysis** — `workflows/contributor_analysis_workflow.py` → `domain_knowledge/<project_name>/contributor_analysis.md`

`<project_name>` is the basename of `--repository` (e.g. `projects/sqlfluff` → `domain_knowledge/sqlfluff/`).

After completion, verify the three output files exist and report:

- Output directory: `domain_knowledge/<project_name>/`
- Per-stage status and aggregated token usage from the workflow summary logs

## Prerequisites

- Run CLI commands from `skills/code-base-onboarding/` (paths like `projects/<repo>` are relative to the skill root).
- `git` CLI must be available (contributor analysis uses `git shortlog`).
- No `PYTHONPATH` setup needed — each CLI script calls `setup_imports()` from `_paths.py`.

## Outputs

| File | Purpose |
|------|---------|
| `domain_knowledge/<project>/file_analysis.md` | Per-file structure and responsibilities |
| `domain_knowledge/<project>/business_analysis.md` | Product context and user journeys |
| `domain_knowledge/<project>/contributor_analysis.md` | Git contributor rollup |

## Scripts and modules

| Path | Role |
|------|------|
| `scripts/_paths.py` | `SKILL_ROOT`, `SCRIPTS_DIR`, `setup_imports()` |
| `scripts/clone_repo.py` | Clone a GitHub repo into `projects/<repo_name>/` |
| `scripts/estimate_tokens.py` | Estimate read token usage before onboarding |
| `scripts/run_onboarding.py` | Run onboarding via `main._run_onboard` |
| `scripts/main.py` | Onboarding CLI; wires config → `OnboardingWorkflow` |
| `scripts/workflows/onboarding_workflow.py` | Orchestrates the three analysis stages |
| `scripts/workflows/file_analysis_workflow.py` | Scans source files, runs file-level agents |
| `scripts/workflows/business_analysis_workflow.py` | Business / CUJ analysis |
| `scripts/workflows/contributor_analysis_workflow.py` | Git contributor rollup |
| `scripts/agents/file_analyzer.py` | Per-batch file LLM agent |
| `scripts/utils/git.py` | `clone_github_repo`, `get_contributors`, URL parsing |
| `scripts/utils/text.py` | `estimate_token_count` |
| `scripts/utils/config.py` | Loads skill-root `config.yaml` |
| `scripts/utils/files.py` | `list_source_files_recursive`, `read_file`, `write_to_file` |
| `scripts/utils/langchain.py` | Token usage aggregation helpers |
