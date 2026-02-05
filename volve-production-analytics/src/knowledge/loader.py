"""Load and parse Markdown documents from the docs/ directory."""

from pathlib import Path
from typing import Dict, List

from src.config import DOCS_DIR


def load_all_docs() -> List[Dict]:
    """
    Load all .md files from DOCS_DIR.

    Returns list of dicts with keys: title, content, sections, filename.
    """
    docs = []
    if not DOCS_DIR.exists():
        return docs

    for md_file in sorted(DOCS_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        sections = _parse_sections(text)
        docs.append({
            "title": md_file.stem.replace("_", " ").title(),
            "content": text,
            "sections": sections,
            "filename": md_file.name,
        })
    return docs


def _parse_sections(text: str) -> List[Dict[str, str]]:
    """Split Markdown into sections by ## headings."""
    sections = []
    current_heading = "Introduction"
    current_body: List[str] = []

    for line in text.split("\n"):
        if line.startswith("## "):
            if current_body:
                sections.append({
                    "heading": current_heading,
                    "body": "\n".join(current_body).strip(),
                })
            current_heading = line.lstrip("# ").strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append({
            "heading": current_heading,
            "body": "\n".join(current_body).strip(),
        })

    return sections
