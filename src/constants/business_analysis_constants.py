BUSINESS_ANALYSIS_SCHEMA = {
    "type": "object",
    "description": "Business and Critical User Journey analysis of the product",
    "properties": {
        "text": {
            "type": "string",
            "description": "The full business and CUJ analysis in markdown format",
        }
    },
    "required": ["text"],
}


def get_business_analysis_prompt(file_analysis_input: str) -> str:
    """Full user prompt for business / CUJ analysis given ``file_analysis.md`` contents."""
    return f"""
            Read through the file analysis summary to understand the product's business logics, use cases and critical user journey(CUJ) for each use case.
            Your task is to compose a business and CUJ analysis of this product, containing:
                - High-level overview of the product
                - The intended audience of the product
                - The use cases and features of the product
                - The CUJ for each use case. Each CUJ stage should point to the source code files it interacts with

            File analysis input: {file_analysis_input}
            """
