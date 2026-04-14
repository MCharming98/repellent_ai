ISSUE_DIAGNOSES_SCHEMA = {
    "type": "object",
    "description": "Issue analysis of the issue",
    "properties": {
        "symptom_observed": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "The analysis of the symptom of the issue",
                },
                "confidence_score": {
                    "type": "number",
                    "description": "The confidence score of the symptom",
                },
            },
            "required": ["analysis", "confidence_score"],
        },
        "divergence_point": {
            "type": "object",
            "description": "The divergence point between the expected and the actual behavior/CUJ in one sentence",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "The analysis of the expected behavior/CUJ and the divergence point between the expected and the actual behavior/CUJ",
                },
                "confidence_score": {
                    "type": "number",
                    "description": "The confidence score of the divergence point",
                },
            },
            "required": ["analysis", "confidence_score"],
        },
        "issue_type": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "The analysis and rationale of the issue type: e.g. bug, expected behavior, UX issue, or a feature request",
                },
                "confidence_score": {
                    "type": "number",
                    "description": "The confidence score of the issue type",
                },
            },
            "required": ["analysis", "confidence_score"],
        },
        "diagnose_hypothesis": {
            "type": "array",
            "description": "The list of diagnose hypothesis and recommended actions for the issue",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis": {
                        "type": "string",
                        "description": "The diagnose hypothesis",
                    },
                    "investigation_actions": {
                        "type": "array",
                        "description": "The list of actions to further investigate into the diagnose hypothesis",
                        "items": {
                            "type": "string",
                            "description": "The detailed recommend action",
                        },
                    },
                    "confidence_score": {
                        "type": "number",
                        "description": "The confidence score of the diagnose hypothesis",
                    },
                },
                "required": ["hypothesis", "investigation_actions", "confidence_score"],
            },
        },
    },
    "required": ["symptom_observed", "divergence_point", "issue_type", "diagnose_hypothesis"],
}


def get_hypothesis_generator_prompt(
    *,
    issue_title: str,
    issue_description: str,
    comments_section: str,
    attachment_section: str,
    file_analysis: str,
    business_analysis: str,
) -> str:
    """Build the issue-triage prompt for hypothesis generation."""
    return f"""
            You are an experienced software engineer who is talented in bug triageing.
            Read the following issue report, combining the title, description, comments,
            images, and linked file attachments, provide an analysis report with the following 6 sections:
            1. Symptom Observed
                - In technical terms, explain the observed symptom of the issue in one sentence.
                - Assign your symptom analysis a confidence score.
            2. Behavior Divergence Point
                - List the expected behavior or CUJ the user was supposed to go through.
                - Explain the divergence point between the expected and the actual behavior in one sentence.
                - Assign your divergence point analysis a confidence score.
            3. Issue Type
                - Hypothesize the type of the issue: a bug, expected behavior, UX issue, or a feature request.
                - Explain your rationale in one sentence.
                - Assign your issue type analysis a confidence score.
            4. Diagnose Hypothesis and Investigation Actions
                - List up to 5 hypotheses that are mutually distinct in root cause, not variations of the same issue.
                - For each hypothesis, provide the following:
                    1. Mechanism analysis:
                        - A step-by-step causal chain explaining how the system transitions from a correct state to the observed failure.
                        - Reference specific components (functions, services, data flow).
                        - If the diagnose points to source code, provide the file name and the function/class name, if applicable, by referring to the structural analysis.
                    2. Observable implications analysis:
                        - What logs, metrics, or behaviors must be true if this hypothesis is correct?
                    3. Investigation actions:
                        - Provide 5 concrete actions that would confirm or falsify this hypothesis.
                        - Sample actions include but are not limited to: code inspection, log query, unit test, web search, ask user, etc.
                    4. Confidence score:
                        - Based on completeness of mechanism, testability, and clear actionable steps (not intuition).
                - Constraints:
                    - Do NOT output vague causes (e.g., "race condition", "bug in logic")
                    without explaining the exact mechanism.
                    - Prefer hypotheses that can be tested quickly.
                    - Each hypothesis must be falsifiable.

            Rules and Guidelines:
            - Your language should be techinical-oriented, so engineers can quickly understand and investigate.
            - Your analysis should be concise and straight to the point.
            - Refer to the provided domain knowledge documents for domain knowledge.

            Issue Title: {issue_title}
            Issue Description: {issue_description}
            {comments_section}
            {attachment_section}
            Domain knowledge documents:
            File analysis: {file_analysis}
            Business analysis: {business_analysis}
        """
