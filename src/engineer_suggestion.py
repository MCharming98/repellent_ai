SCHEMA = {
    "type": "array",
    "description": "The list of suggested engineers",
    "items": {
        "type": "object",
        "properties": {
            "engineer_name": {
                "type": "string",
                "description": "The name of the suggested engineer",
            },
            "rationale": {
                "type": "string",
                "description": "The rationale for the suggested engineer",
            },
            "confidence_score": {
                "type": "number",
                "description": "The confidence score of the suggested engineer",
                "minimum": 0,
                "maximum": 100,
            },
        },
        "required": ["engineer_name", "rationale", "confidence_score"],
    },
}

PROMPT = """
5. Suggested Engineers
    - Select 3 suggested engineers with the highest confidence score to further triage this issue.
    - In the section body:
        - State your rationale in one sentence
        - Assign each engineer suggestion action a confidence score.
        - Enclose each body in the <details> tag.
        - Put the engineer’s name and the confidence score in the <summary> as the title.
            - Format: Engineer 1/2/3: [engineer’s name] (Confidence [your confidence %])
"""