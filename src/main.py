import os
import re
import sys
from pathlib import Path

import yaml

from workflows.onboarding_workflow import OnboardingWorkflow


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.yaml"


# `api_key: $VAR_NAME` or `api_key: ${VAR_NAME}` reads from the process environment.
_ENV_VAR_REF = re.compile(
    r"^\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))$"
)


def _resolve_api_key_from_config(raw: object) -> str:
    """Return API key string: literal value, or env lookup for ``$VAR`` / ``${VAR}``."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        return str(raw).strip()
    s = raw.strip()
    if not s:
        return ""
    m = _ENV_VAR_REF.match(s)
    if m:
        name = m.group(1) or m.group(2)
        return os.environ.get(name, "")
    return s


def _load_config(path: Path) -> dict:
    if not path.is_file():
        print(f"Error: config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        print(f"Error: config must be a YAML mapping: {path}", file=sys.stderr)
        sys.exit(1)
    return data


def main() -> None:
    cfg = _load_config(_default_config_path())

    repository = (cfg.get("repository") or "").strip()
    if not repository:
        print(
            "Error: set `repository` in config.yaml (path to the source repository).",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = _resolve_api_key_from_config(cfg.get("api_key"))
    if not api_key:
        print(
            "Error: set `api_key` in config.yaml (literal or $VAR / ${VAR}), "
            "or LLM_API_KEY / GOOGLE_API_KEY in the environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    project_name = Path(repository).resolve().name
    domain_knowledge_dir = f"domain_knowledge/{project_name}"

    batch_size = int(cfg.get("file_analysis_batch_size", 10))
    model_name = cfg.get("model_name", "gemini-3-flash-preview")
    model_provider = cfg.get("model_provider", "google_genai")

    onboarding_workflow = OnboardingWorkflow(
        source_repository=repository,
        domain_knowledge=domain_knowledge_dir,
        file_analysis_batch_size=batch_size,
        model=model_name,
        model_provider=model_provider,
        api_key=api_key,
    )
    onboarding_workflow.build_workflow()
    onboarding_workflow.run()


if __name__ == "__main__":
    main()
