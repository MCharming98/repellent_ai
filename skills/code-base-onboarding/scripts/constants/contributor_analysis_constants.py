CONTRIBUTOR_ANALYSIS_SCHEMA = {
    "type": "object",
    "description": "Contributor analysis of the repository",
    "properties": {
        "text": {
            "type": "string",
            "description": "The full contributor analysis in markdown format",
        }
    },
    "required": ["text"],
}


def get_contributor_analysis_prompt_intro() -> str:
    """Static prompt prefix; the workflow appends per-file contributor lines."""
    return """
            Given the following files and the contributor list for each file, write an analysis containing the following:
                -  List all the engineers who has contributed to this repository, sort the engineers by number of commits descending.
                    - For each contributor, list their name, account(s), and a high-level summary of their contributions.
                    - If different accounts or names point to the same contributor, consolidate them.
            Input:
            """
