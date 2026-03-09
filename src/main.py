import argparse
from pathlib import Path

from onboarding_workflow import OnboardingWorkflow


def main():
    parser = argparse.ArgumentParser(description="Main entry point for Repellent AI")
    parser.add_argument("--repository", required=True, help="Source repository path")
    parser.add_argument("--workspace", default=None, help="Agent workspace directory (default: agent_workspace/<project_name>)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for structural analysis (default: 10)")
    parser.add_argument("--model_name", default="gemini-3-flash-preview", help="LLM model to use (default: gemini-3-flash-preview)")
    parser.add_argument("--model_provider", default="google_genai", help="LLM provider to use (default: google_genai)")
    parser.add_argument("--api-key", required=True, help="API key for the LLM provider")
    args = parser.parse_args()

    workspace = args.workspace
    if workspace is None:
        project_name = Path(args.repository).resolve().name
        workspace = f"./agent_workspace/{project_name}"

    onboarding_workflow = OnboardingWorkflow(
        source_repository=args.repository,
        agent_workspace=workspace,
        structural_analysis_batch_size=args.batch_size,
        model=args.model_name,
        model_provider=args.model_provider,
        api_key=args.api_key,
    )
    onboarding_workflow.build_workflow()
    onboarding_workflow.run()


if __name__ == "__main__":
    main()
