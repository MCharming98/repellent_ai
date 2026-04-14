"""Text formatting helpers."""


def format_key_to_subheading(key: str) -> str:
    """Convert snake_case keys to title-cased markdown headings."""
    return key.replace("_", " ").strip().title()
