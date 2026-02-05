"""
Keyword-based document retrieval.

Simple token-overlap scoring -- no embeddings, no external dependencies.
"""

import re
from typing import Dict, List, Optional

from .loader import load_all_docs


def search_docs(
    query: str,
    top_k: int = 3,
    docs: Optional[List[Dict]] = None,
) -> List[Dict[str, str]]:
    """
    Search knowledge base docs by keyword overlap scoring.

    Returns ranked results, each with keys: title, heading, snippet, score.
    """
    if docs is None:
        docs = load_all_docs()

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored_sections: List[Dict] = []
    for doc in docs:
        for section in doc["sections"]:
            section_tokens = _tokenize(section["body"])
            score = _score_overlap(query_tokens, section_tokens)
            if score > 0:
                snippet = section["body"][:200].replace("\n", " ").strip()
                if len(section["body"]) > 200:
                    snippet += "..."
                scored_sections.append({
                    "title": doc["title"],
                    "heading": section["heading"],
                    "snippet": snippet,
                    "score": score,
                })

    scored_sections.sort(key=lambda x: x["score"], reverse=True)
    return scored_sections[:top_k]


def _tokenize(text: str) -> set:
    """Lowercase, strip punctuation, split into unique tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _score_overlap(query_tokens: set, doc_tokens: set) -> float:
    """Overlap: |intersection| / |query_tokens|."""
    if not query_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)
