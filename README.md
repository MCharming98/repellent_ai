# Repellent AI

**Repellent AI** is an AI agent that learns any project on its own. It analyzes source code, contributors, and business logic to build a knowledge base. In the future, it will use this knowledge to triage and root cause issues.

## Entry Point

The main entry point is `src/main.py`:

```bash
python src/main.py --repository /path/to/source/repo --api-key YOUR_API_KEY
```

### Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--repository` | Yes | — | Path to the source repository |
| `--api-key` | Yes | — | API key for the LLM provider |
| `--workspace` | No | `agent_workspace/<project_name>` | Directory for analysis outputs |
| `--batch-size` | No | 10 | Batch size for structural analysis |
| `--model_name` | No | gemini-3-flash-preview | LLM model to use |
| `--model_provider` | No | google_genai | LLM provider (e.g. google_genai, anthropic) |

## Architecture

The onboarding pipeline runs three workflows in sequence:

1. **Structural Analysis** → 2. **Business Analysis** → 3. **Contributor Analysis**

### Workflows

- **Onboarding Workflow** (`onboarding_workflow.py`)  
  Orchestrates the full onboarding pipeline: structural analysis, business analysis, and contributor analysis. Outputs are written to the agent workspace.

- **Structural Analysis Workflow** (`structural_analysis_workflow.py`)  
  Recursively discovers source files, creates parallel analysis agents, and produces a per-file analysis including responsibilities, contributors, and functions. Output: `file_analysis.md`.

- **Business Analysis Workflow** (`business_analysis_workflow.py`)  
  Reads the structural analysis and generates a business and Critical User Journey (CUJ) overview: product summary, audience, use cases, features, and CUJ stages with linked source files. Output: `business_analysis.md`.

- **Contributor Analysis Workflow** (`contributor_analysis_workflow.py`)  
  Gathers contributor data per file and produces an analysis of all contributors: names, accounts, commit counts, and contribution summaries. Output: `contributor_analysis.md`.

### Agents

- **File Analyzer** (`file_analyzer.py`)  
  LLM agent that inspects a batch of source files and produces structured summaries (file-level responsibilities, contributors, and functions). Multiple agents run in parallel to analyze the project.

### Supporting Modules

- **utils/** — Package with utilities for file/directory operations (`utils/files`), git contributor lookup (`utils/git`), used across workflows.

## Output

All outputs go under the workspace directory (default: `agent_workspace/<project_name>/`):

- `file_analysis.md` — Per-file structural analysis
- `business_analysis.md` — Business logic and CUJ analysis
- `contributor_analysis.md` — Contributor analysis
