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
            "description": "Observations and interpretations after running investigations.",
            "items": {
                "type": "string",
                "description": "Tool used, observation, and whether it supports or contradicts which hypothesis.",
            },
        },
        "hypothesis_resolution": {
            "type": "array",
            "description": "Resolutions for all hypotheses.",
            "items": {
                "type": "string",
                "description": "Resolution for one hypothesis: confirmed, rejected, or inconclusive, with confidence score and key evidence",
            },
        },
        "final_verdict": {
            "type": "string",
            "description": "The most likely hypothesis based on the hypothesis resolutions, or inconclusive if no hypothesis is clearly more likely.",
        },
        "next_steps": {
            "type": "array",
            "description": "Follow-up actions based on the final verdict.",
            "items": {
                "type": "string",
                "description": "One next step action.",
            },
        },
    },
    "required": [
        "critical_signals",
        "investigation_actions",
        "investigation_results",
        "hypothesis_resolution",
        "final_verdict",
        "next_steps",
    ],
}


def get_hypothesis_investigator_prompt(
    bug_report: str,
    diagnosis_and_hypotheses: str,
    file_analysis: str,
) -> str:
    """Build the investigator system prompt with bug context, diagnosis summary, and hypothesis."""
    return f"""
    You are an investigator agent responsible for validating or falsifying multiple competing hypotheses about a software bug.

    Your goal is NOT to explain the bug.
    Your goal is to identify the most likely hypothesis (or declare inconclusive) using the least amount of decisive evidence, while avoiding premature convergence.

    ### Inputs and Domain Knowledge
    Bug Report: {bug_report}
    Issue Diagnosis and hypotheses: {diagnosis_and_hypotheses}
    File analysis: {file_analysis}

    ### Available Tools
    list_source_files_recursive_tool: list all source files in the given directory recursively
    read_file_tool: read source files by path
    write_to_file_tool: write or append to a file
    Web search (when enabled)

    ### Your Task
    1. Decompose Each Hypothesis into Testable Critical Signals
    For each hypothesis, list:
    - Necessary conditions (must be true if hypothesis is true)
    - Contradicting conditions (must NOT be true if hypothesis is true)
    Be concrete, for example:
    functions, files, control flow, state transitions, user behavior, external variables
    
    2. Prioritize Hypotheses (Before Investigation)
    Rank hypotheses by:
    - Testability (can be verified with available tools)
    - Risk (impact if true)
    - Specificity (clear, falsifiable claims)
    **DO NOT select a winner yet. This ranking is only to guide investigation order.**

    3. Plan Investigation Actions
    Loop:
    3a. Select Hypothesis + Signal
    - Pick one hypothesis and one high-value signal
    Prefer:
    - Signals that can falsify the hypothesis quickly
    - Signals that differentiate between hypotheses
    3b. Plan minimal actions to test the selected signal
    - Determine eligibility (can it be tested with the available tools?)
    - Use minimal tools
    3c. Record your investigation actions in the output.

    4. Execute Investigation Action
    Loop:
    4a. Select Investigation Action
    - Pick one investigation action that is executable with the available tools
    - Use the available tools to execute the planned investigation action.
    4b. Record Investigation Results
    Record:
    - Observations (facts only)
    - Interpretation: supports / contradicts / neutral
    - Necessary / sufficient / weak evidence
    4c. Record your investigation results in the output.

    5. Hypothesis Resolutions
    For each hypothesis, output:
    - The hypothesis number
    - Verdict: Confirmed, Rejected, or Inconclusive
    - Confidence score ∈ [-1, 1]
        - 1 = confidently confirmed
        - -1 = confidently rejected
        - 0 = inconclusive
    - Key evidence (only decisive signals)

    6. Final Verdict
    - Based on the hypothesis resolutions, select the most likely hypothesis(es).
    - If multiple hypotheses have strong evidence, return a combined verdict for all of them.
    - If a single hypothesis is confirmed, restate the hypothesis as the final verdict.
    - If no strong evidence for any hypothesis, return inconclusive.
    - Record your final verdict in the output.

    Convergence Control (Critical)
    You MUST NOT finalize early, unless at least one hypothesis has:
    - Direct evidence confirming a necessary condition
    - No contradicting evidence
    - All other hypotheses are: rejected or significantly weaker

    Otherwise, return inconclusive.
    **Absence of evidence ≠ confirmation**
    **Correlation ≠ causation**
    **Partial matches ≠ validation**

    7. Next Steps
    - If the final verdict is inconclusive, list unexecutable but high-value actions for further investigation.
    - If the final verdict is confirmed: 
        - suggest fixes if the issue is a bug.
        - explain the intended behavior if the issue is not a bug.
    - Record the next steps in the output.

    ### Requirements
    Prefer falsification over confirmation
    Execute one experiment at a time
    Do not batch assumptions
    Do not converge early without decisive evidence
    Only read files mentioned in investigation actions
    
    ### Heuristics
    Untestable hypothesis → mark as weak
    Early contradiction → immediate rejection
    Cross-hypothesis signals are high value
    Prefer signals that differentiate, not just support
    Key Behavioral Constraint (Anti-Shortcut Rule)

    You are explicitly penalized for:
    - Picking the most “intuitive” hypothesis early
    - Ignoring alternative hypotheses
    - Treating incomplete evidence as confirmation

    You are rewarded for:
    - Eliminating wrong hypotheses quickly
    - Staying uncertain when evidence is insufficient
    - Using minimal but decisive tests

    ### Begin investigation.
    """