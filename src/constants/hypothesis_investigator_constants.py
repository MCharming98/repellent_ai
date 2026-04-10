INVESTIGATION_ANALYSIS_SCHEMA = {
    "type": "object",
    "description": "Structured output of a single-hypothesis investigation (signals, actions, results, verdict).",
    "properties": {
        "critical_signals": {
            "type": "array",
            "description": "Necessary conditions and contradicting conditions that define whether the hypothesis can hold.",
            "items": {
                "type": "string",
                "description": "One concrete signal (e.g. file, function, state, or behavior).",
            },
        },
        "investigation_actions": {
            "type": "array",
            "description": "Minimal experiments or steps planned to validate or falsify the hypothesis.",
            "items": {
                "type": "string",
                "description": "One executable investigation action.",
            },
        },
        "investigation_results": {
            "type": "array",
            "description": "Per-experiment observations and interpretations after running investigations.",
            "items": {
                "type": "string",
                "description": "Tool used, observation, and whether it supports or contradicts the hypothesis.",
            },
        },
        "final_resolution": {
            "type": "string",
            "description": "Verdict summarizing whether the hypothesis is confirmed, rejected, or inconclusive based on evidence.",
        },
        "confidence_score": {
            "type": "number",
            "description": "Confidence in the verdict: 1 = confidently confirmed, -1 = confidently rejected, 0 = inconclusive.",
            "minimum": -1,
            "maximum": 1,
        },
        "next_steps": {
            "type": "array",
            "description": "Follow-up actions that could not be performed yet (e.g. user input, missing tools).",
            "items": {
                "type": "string",
                "description": "One blocked or deferred next step.",
            },
        },
    },
    "required": [
        "critical_signals",
        "investigation_actions",
        "investigation_results",
        "final_resolution",
        "confidence_score",
        "next_steps",
    ],
}


def get_hypothesis_investigator_prompt(
    bug_report: str,
    hypothesis: str,
    file_analysis: str,
    diagnosis: str,
) -> str:
    """Build the investigator system prompt with bug context, diagnosis summary, and hypothesis."""
    return f"""
    You are an investigator agent responsible for validating or falsifying a single hypothesis about a software bug.
    Your goal is NOT to explain the bug.
    Your goal is to reach a correct verdict (confirm or reject) using the least amount of evidence.

    ## Inputs
    Bug Report: {bug_report}

    Issue Diagnosis: {diagnosis}

    Hypothesis: {hypothesis}

    File Analysis: {file_analysis}

    Available Tools:
    - read_file_tool: read source files by path
    - write_to_file_tool: write or append to a file (use sparingly for notes or artifacts)
    - Web search (when enabled by the model provider)

    ## Your Task

    ### 1. Identify critical signals
    List:
    - Necessary conditions, which must be true if the hypothesis is true
    - Contradicting conditions, which must be false if the hypothesis is true

    Be specific, for example but not limited to:
    - functions, files, control flow, state transitions, user behavior, external variables

    ### 2. Design minimal experiments as investigation actions
    Propose 1-3 high-leverage actions to validate/falsify the hypothesis.
    Each experiment must:
    - Be executable using available tools
    - Minimize cost (time, compute, reads)
    - Maximize information gain

    ### 3. Plan ac

    ### 3. Execute investigations and record investigation results
    For each experiment:
    - Confirm action: read source code file, write file, search on web, etc
    - Decide tool use: read_file_tool, write_to_file_tool, web search, etc
    - List observation: what you found
    - State interpretation: Does this support or contradict the hypothesis? Is this necessary or sufficient evidence?

    ### 4. Final Resolution and Confidence Score
    Based on the evidence, classify the hypothesis and give a confidence score between -1 and 1 with the following criteria
    - confidently confirmed = 1
    - confidently rejected = -1
    - inconclusive(insufficient or ambiguous evidence) = 0

    ### 5. Next Steps
    - If there are actions that cannot be performed at this time, for example, need user input or missing required tools, list the actions here.
    - If the hypothesis is confirmed and is a bug, suggest the next steps to be taken to fix the bug.
    - If the hypothesis is confirmed and is a feature or intended behavior, explain the reasoning behind the behavior.

    ## Requirements
    - Prefer falsification over confirmation (try to disprove first)
    - Execute one experiment at a time
    - Stop early if decisive evidence is found
    - Treat absence of evidence as inconclusiveness, not confirmation
    - Only read files that are necessary to validate or falsify the hypothesis

    ## Heuristics
    - If a hypothesis cannot be tested, mark it as weak
    - If evidence contradicts a necessary condition, reject it immediately
    - If multiple signals align, increase confidence

    Begin investigation.
    """