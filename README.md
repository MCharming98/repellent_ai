# Repellent AI

**Repellent AI** is an AI agent that learns any project on its own. It analyzes source code, contributors, and business logic to build a knowledge base. In the future, it will use this knowledge to triage and root cause issues.

## Entry Point

The main entry point is `src/main.py`. Configure `config.yaml` at the repository root (model settings and API key / env vars). Pass the repository path on the command line. Analysis outputs are always written under `domain_knowledge/<project_name>/` (relative to the current working directory).

```bash
PYTHONPATH=src python src/main.py onboard --repository projects/AntennaPod
```

With a `main` launcher in your `PATH` (or `python main.py onboard` if you name the entry script `main.py`):

```bash
python main onboard --repository projects/AntennaPod
```

## Architecture

The onboarding pipeline runs three workflows in sequence:

1. **File Analysis** → 2. **Business Analysis** → 3. **Contributor Analysis**

### Workflows

- **Onboarding Workflow** (`workflows/onboarding_workflow.py`)  
  Orchestrates the full onboarding pipeline: file analysis, business analysis, and contributor analysis. Outputs are written under `domain_knowledge/<project_name>/`.

- **File Analysis Workflow** (`workflows/file_analysis_workflow.py`)  
  Recursively discovers source files, creates parallel analysis agents, and produces a per-file analysis including responsibilities, contributors, and functions. Output: `file_analysis.md`.

- **Business Analysis Workflow** (`workflows/business_analysis_workflow.py`)  
  Reads the file analysis and generates a business and Critical User Journey (CUJ) overview: product summary, audience, use cases, features, and CUJ stages with linked source files. Output: `business_analysis.md`.

- **Contributor Analysis Workflow** (`workflows/contributor_analysis_workflow.py`)  
  Gathers contributor data per file and produces an analysis of all contributors: names, accounts, commit counts, and contribution summaries. Output: `contributor_analysis.md`.

### Agents

- **File Analyzer** (`agents/file_analyzer.py`)  
  LLM agent that inspects a batch of source files and produces structured summaries (file-level responsibilities, contributors, and functions). Multiple agents run in parallel to analyze the project.

### Supporting Modules

- **utils/** — Package with utilities for file/directory operations (`utils/files`), git contributor lookup (`utils/git`), used across workflows.

## Output

All outputs go under `domain_knowledge/<project_name>/` (not configurable):

- `file_analysis.md` — Per-file structural analysis
- `business_analysis.md` — Business logic and CUJ analysis
- `contributor_analysis.md` — Contributor analysis
