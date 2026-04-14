"""Generates structured issue analysis and hypotheses using project knowledge from the agent workspace."""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import os
import re
import time
from urllib.parse import urlparse

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.structured_output import ToolStrategy
from constants.hypothesis_generator_constants import (
    ISSUE_DIAGNOSES_SCHEMA,
    get_hypothesis_generator_prompt,
)
from utils import (
    read_file,
    extract_image_markdown,
    extract_github_user_attachment_links,
    fetch_image_as_data_url,
    fetch_url_bytes,
    guess_attachment_extension,
    is_image_bytes,
    is_text_bytes,
    format_key_to_subheading,
    write_to_file,
)


_SUMMARY_SECTION_KEYS = ("symptom_observed", "divergence_point", "issue_type")

_ATTACHMENT_BASENAME_SAFE = re.compile(r"[^\w.\-]+")


def _combined_issue_markdown(issue_details: dict) -> str:
    """Concatenate issue body and all comment bodies for link scanning.

    Used by attachment collection so we can discover markdown links in both the
    original issue description and follow-up discussion comments.
    """
    parts: list[str] = [issue_details.get("body") or ""]
    for c in issue_details.get("comments") or []:
        if isinstance(c, dict):
            parts.append(str(c.get("body") or ""))
    return "\n\n".join(parts)


def _format_issue_comments_for_prompt(issue_details: dict) -> str:
    """Format GitHub-style issue comments for the triage prompt (author + body per comment)."""
    comments = issue_details.get("comments") or []
    if not isinstance(comments, list) or not comments:
        return ""
    blocks: list[str] = []
    for i, c in enumerate(comments, start=1):
        if not isinstance(c, dict):
            continue
        user = (c.get("user") or {}).get("login") or "unknown"
        body = str(c.get("body") or "").strip()
        blocks.append(f"Comment {i} (@{user}):\n{body}")
    return "\n\n".join(blocks).strip()


def _sanitize_attachment_basename(name: str, fallback: str) -> str:
    """Sanitize a candidate filename to a safe basename.

    Strips directory components, replaces unsafe chars, guards against ``.``/``..``,
    and enforces a length cap to avoid path/FS issues.
    """
    base = (name or fallback).strip()
    base = os.path.basename(base.replace("\\", "/"))
    base = _ATTACHMENT_BASENAME_SAFE.sub("_", base)
    if not base or base in (".", ".."):
        base = fallback
    return base[:180]


def _unique_path_in_dir(directory: Path, filename: str) -> Path:
    """Return a non-colliding file path inside ``directory``.

    Keeps the original name when available, otherwise appends ``_2``, ``_3``, etc.
    before the extension.
    """
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = Path(filename).stem, Path(filename).suffix
    n = 2
    while True:
        alt = directory / f"{stem}_{n}{suffix}"
        if not alt.exists():
            return alt
        n += 1


def _build_saved_filename(
    label: str,
    url: str,
    data: bytes,
    content_type: str,
    *,
    image_link_index: int | None = None,
) -> str:
    """Basename with extension for a saved attachment."""
    if image_link_index is not None:
        ext = guess_attachment_extension(data, content_type)
        return f"image_{image_link_index}{ext}"
    label = (label or "").strip()
    if label and label.lower() != "image":
        safe = _sanitize_attachment_basename(label, "attachment")
        if Path(safe).suffix:
            return safe
    path_last = urlparse(url).path.rstrip("/").split("/")[-1]
    if path_last and "." in path_last:
        safe = _sanitize_attachment_basename(path_last, "attachment")
        if Path(safe).suffix:
            return safe
    ext = guess_attachment_extension(data, content_type)
    return f"attachment{ext}"


def _format_issue_diagnosis_markdown(data: dict) -> str:
    """Render full diagnosis.md: triage sections plus all diagnose hypotheses."""
    blocks: list[str] = []
    for key in _SUMMARY_SECTION_KEYS:
        if key not in data:
            continue
        val = data[key]
        blocks.append(f"## {format_key_to_subheading(key)}\n")
        if isinstance(val, dict):
            blocks.append(str(val.get("analysis", "")).strip())
            blocks.append("")
            blocks.append(f"Confidence score: {val.get('confidence_score', '')}")
        blocks.append("")

    hyps = data.get("diagnose_hypothesis") or data.get("diagnosis_hypothesis") or []
    if isinstance(hyps, list) and hyps:
        blocks.append("## diagnose_hypothesis\n")
        for i, item in enumerate(hyps):
            if not isinstance(item, dict):
                continue
            blocks.append(_format_hypothesis_item_markdown(item, i + 1))
            blocks.append("")

    return "\n".join(blocks).strip() + "\n"


def _format_hypothesis_item_markdown(item: dict, index_one_based: int) -> str:
    """Markdown for one hypothesis (subsection under diagnose_hypothesis)."""
    lines: list[str] = [
        f"### Hypothesis {index_one_based}\n",
        str(item.get("hypothesis", "")).strip(),
        "",
    ]
    actions = item.get("investigation_actions") or []
    if actions:
        lines.append("Investigation actions")
        for action in actions:
            lines.append(f"- {action}")
        lines.append("")
    lines.append(f"Confidence score: {item.get('confidence_score', '')}")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


class HypothesisGenerator:
    """Triages an issue and emits hypotheses with investigation actions using workspace knowledge."""

    class State(TypedDict):
        issue_dir: str
        agent_workspace: str
        issue_details: dict
        issue_images: list[str]
        issue_attachment_context: str
        file_analysis: str
        business_analysis: str
        contributor_analysis: str
        issue_diagnosis_json: dict
        issue_diagnosis: str
        write_status: bool

    def __init__(self, issue_dir: str, agent_workspace: str, model: str, model_provider: str, api_key: str):
        self.issue_dir = Path(issue_dir)
        self.agent_workspace = Path(agent_workspace)
        self.model = model
        self.model_provider = model_provider
        self.api_key = api_key
        model_kwargs = {"api_key": api_key}
        # Default to Google Developer API instead of Vertex AI
        if model_provider == "google_genai":
            model_kwargs["google_api_key"] = api_key
        self.agent = create_agent(
            model=init_chat_model(model, model_provider=model_provider, **model_kwargs),
            response_format=ToolStrategy(ISSUE_DIAGNOSES_SCHEMA)
        )

    def load_issue(self, state: State) -> dict:
        """Load issue details from JSON into state."""
        path = Path(state["issue_dir"], "issue_details.json")
        with open(path, "r", encoding="utf-8") as f:
            raw_issue_details = json.load(f)
        comments = raw_issue_details.get("comments")
        if not isinstance(comments, list):
            comments = []
        issue_details = {
            "title": raw_issue_details.get("title", ""),
            "body": raw_issue_details.get("body", ""),
            "comments": comments,
        }
        return {"issue_details": issue_details}

    def collect_issue_attachments(self, state: State) -> dict:
        """
        Download URLs from ``[label](user-attachments/...)`` and ``[Image](...)`` markdown.
        Saves only images and text under ``<issue_dir>/attachments/``; binary files (e.g. zip)
        are skipped (not stored, not inlined in the prompt).
        """
        issue_dir = Path(state["issue_dir"])
        att_dir = issue_dir / "attachments"
        att_dir.mkdir(parents=True, exist_ok=True)

        combined = _combined_issue_markdown(state["issue_details"])
        seen_urls: set[str] = set()
        context_blocks: list[str] = []
        issue_images: list[str] = []
        saved_count = 0

        pairs = extract_github_user_attachment_links(combined)
        for label, url in pairs:
            if url in seen_urls:
                continue
            got = fetch_url_bytes(url)
            if not got:
                print(f"Hypothesis generator: failed to download attachment {url!r}")
                continue
            data, ct = got

            if is_image_bytes(data, ct):
                seen_urls.add(url)
                fname = _build_saved_filename(label, url, data, ct, image_link_index=None)
                out_path = _unique_path_in_dir(att_dir, fname)
                out_path.write_bytes(data)
                saved_count += 1
                print(f"Hypothesis generator: saved user attachment -> attachments/{out_path.name}")
                issue_images.append(url)
            elif is_text_bytes(data, ct):
                seen_urls.add(url)
                fname = _build_saved_filename(label, url, data, ct, image_link_index=None)
                out_path = _unique_path_in_dir(att_dir, fname)
                out_path.write_bytes(data)
                saved_count += 1
                rel = out_path.name
                print(f"Hypothesis generator: saved user attachment -> attachments/{rel}")
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    text = data.decode("utf-8", errors="replace")
                context_blocks.append(
                    f"### {label}\nSource: {url}\nSaved as: attachments/{rel}\n\n"
                    f"{text}"
                )
            else:
                seen_urls.add(url)
                print(
                    f"Hypothesis generator: skipping binary attachment ({label!r}): {url!r}"
                )

        for i, url in enumerate(extract_image_markdown(combined)):
            if url in seen_urls:
                continue
            got = fetch_url_bytes(url)
            if not got:
                print(f"Hypothesis generator: failed to download image {url!r}")
                continue
            data, ct = got
            if not is_image_bytes(data, ct):
                seen_urls.add(url)
                print(
                    f"Hypothesis generator: skipping non-image [Image] URL (binary or unknown): {url!r}"
                )
                continue
            seen_urls.add(url)
            fname = _build_saved_filename("", url, data, ct, image_link_index=i)
            out_path = _unique_path_in_dir(att_dir, fname)
            out_path.write_bytes(data)
            saved_count += 1
            print(f"Hypothesis generator: saved image attachment -> attachments/{out_path.name}")
            issue_images.append(url)

        ctx = "\n\n---\n\n".join(context_blocks) if context_blocks else ""
        if ctx or issue_images or saved_count:
            print(
                f"Hypothesis generator: attachments: {saved_count} file(s) saved, "
                f"{len(issue_images)} image URL(s) for multimodal"
            )
        return {"issue_images": issue_images, "issue_attachment_context": ctx}

    def load_project_knowledge(self, state: State) -> dict:
        """Load project knowledge from agent workspace (file_analysis, business_analysis, contributor_analysis)."""
        workspace = Path(state["agent_workspace"])
        file_analysis = read_file(str(workspace / "file_analysis.md"))
        business_analysis = read_file(str(workspace / "business_analysis.md"))
        contributor_analysis = read_file(str(workspace / "contributor_analysis.md"))
        return {"file_analysis": file_analysis, "business_analysis": business_analysis, "contributor_analysis": contributor_analysis}

    def analyze_issue(self, state: State) -> dict:
        """Analyze issue using project knowledge."""
        print(f"Hypothesis generator: Analyzing {state['issue_dir']}")
        start_time = time.perf_counter()
        issue_details = state["issue_details"]
        issue_images = state.get("issue_images") or []
        file_analysis = state["file_analysis"]
        business_analysis = state["business_analysis"]
        contributor_analysis = state["contributor_analysis"]
        attachment_ctx = (state.get("issue_attachment_context") or "").strip()
        comments_block = _format_issue_comments_for_prompt(issue_details)
        comments_section = (
            f"\n            Issue comments (discussion thread):\n            {comments_block}\n"
            if comments_block
            else ""
        )
        attachment_section = (
            f"\n            User-uploaded file contents (from linked GitHub attachments):\n"
            f"            {attachment_ctx}\n"
            if attachment_ctx
            else ""
        )
        prompt = get_hypothesis_generator_prompt(
            issue_title=issue_details["title"],
            issue_description=issue_details["body"],
            comments_section=comments_section,
            attachment_section=attachment_section,
            file_analysis=file_analysis,
            business_analysis=business_analysis,
        )
        image_blocks = []
        for image_url in issue_images:
            data_url = fetch_image_as_data_url(image_url)
            if data_url:
                image_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
        message = {
            "role": "user",
            "content": [{"type": "text", "text": prompt}, *image_blocks],
        }
        result = self.agent.invoke(
            {"messages": [message]}
        )
        elapsed = time.perf_counter() - start_time
        print(f"Hypothesis generator: analysis completed in {elapsed:.2f}s")
        sr = result.get("structured_response")
        if not isinstance(sr, dict):
            raise ValueError(
                "Hypothesis generator: expected structured_response dict, got "
                f"{type(sr)}"
            )
        if "symptom_observed" not in sr and "text" in sr:
            try:
                sr = json.loads(sr["text"])
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(
                    f"Hypothesis generator: could not parse structured_response: {e}"
                ) from e
        return {"issue_diagnosis_json": sr}

    def format_issue_diagnosis_markdown(self, state: State) -> dict:
        md = _format_issue_diagnosis_markdown(state["issue_diagnosis_json"])
        return {"issue_diagnosis": md}

    def write_analysis_to_file(self, state: State) -> dict:
        issue_dir = Path(state["issue_dir"])
        diagnosis_path = issue_dir / "diagnosis.md"
        print(f"Hypothesis generator: Writing diagnosis to {diagnosis_path}")
        write_to_file(str(diagnosis_path), state["issue_diagnosis"], "w")
        return {"write_status": True}

    def build_workflow(self):
        self.workflow = StateGraph(self.State)
        self.workflow.add_node("load_issue", self.load_issue)
        self.workflow.add_node("collect_issue_attachments", self.collect_issue_attachments)
        self.workflow.add_node("load_project_knowledge", self.load_project_knowledge)
        self.workflow.add_node("analyze_issue", self.analyze_issue)
        self.workflow.add_node("format_issue_diagnosis_markdown", self.format_issue_diagnosis_markdown)
        self.workflow.add_node("write_analysis_to_file", self.write_analysis_to_file)

        self.workflow.add_edge(START, "load_issue")
        self.workflow.add_edge("load_issue", "collect_issue_attachments")
        self.workflow.add_edge("collect_issue_attachments", "load_project_knowledge")
        self.workflow.add_edge("load_project_knowledge", "analyze_issue")
        self.workflow.add_edge("analyze_issue", "format_issue_diagnosis_markdown")
        self.workflow.add_edge("format_issue_diagnosis_markdown", "write_analysis_to_file")
        self.workflow.add_edge("write_analysis_to_file", END)
        self.workflow = self.workflow.compile()

    async def run(self):
        final_state = await self.workflow.ainvoke({
            "issue_dir": str(self.issue_dir),
            "agent_workspace": str(self.agent_workspace),
        })
        return final_state


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a GitHub issue using project knowledge from an agent workspace"
    )
    parser.add_argument(
        "--issue-details",
        required=True,
        dest="issue_details",
        help="Path to issue_details.json",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Agent workspace dir (file_analysis.md, business_analysis.md, contributor_analysis.md)",
    )
    parser.add_argument(
        "--model_name",
        default="gemini-3-flash-preview",
        help="LLM model (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--model_provider",
        default="google_genai",
        help="LLM provider (default: google_genai)",
    )
    parser.add_argument("--api-key", required=True, help="API key for the LLM provider")
    args = parser.parse_args()

    agent = HypothesisGenerator(
        issue_dir=args.issue_details,
        agent_workspace=args.workspace,
        model=args.model_name,
        model_provider=args.model_provider,
        api_key=args.api_key,
    )
    agent.build_workflow()
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()