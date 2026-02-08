"""Tests for the knowledge base module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.knowledge.retrieval import _tokenize, _score_overlap, search_docs
from src.knowledge.loader import load_all_docs, _parse_sections


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_lowercase(self):
        tokens = _tokenize("HELLO World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_strips_punctuation(self):
        tokens = _tokenize("What is WAPE?")
        assert "wape" in tokens
        assert "?" not in str(tokens)

    def test_returns_set(self):
        tokens = _tokenize("oil oil oil")
        assert isinstance(tokens, set)
        assert len(tokens) == 1

    def test_empty_string(self):
        assert _tokenize("") == set()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoreOverlap:
    def test_full_overlap(self):
        assert _score_overlap({"a", "b"}, {"a", "b", "c"}) == 1.0

    def test_partial_overlap(self):
        assert _score_overlap({"a", "b"}, {"b", "c"}) == 0.5

    def test_no_overlap(self):
        assert _score_overlap({"a"}, {"b", "c"}) == 0.0

    def test_empty_query(self):
        assert _score_overlap(set(), {"a", "b"}) == 0.0


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

class TestParseSections:
    def test_splits_on_h2(self):
        text = "Intro text\n## Section A\nBody A\n## Section B\nBody B"
        sections = _parse_sections(text)
        assert len(sections) == 3
        assert sections[0]["heading"] == "Introduction"
        assert sections[1]["heading"] == "Section A"
        assert sections[2]["heading"] == "Section B"

    def test_no_headings(self):
        text = "Just some text\nwithout headings"
        sections = _parse_sections(text)
        assert len(sections) == 1
        assert sections[0]["heading"] == "Introduction"


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

class TestLoadAllDocs:
    def test_loads_from_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test_doc.md").write_text(
                "# Test\n\n## Section One\nContent here.\n## Section Two\nMore content."
            )
            with patch("src.knowledge.loader.DOCS_DIR", Path(tmpdir)):
                docs = load_all_docs()
                assert len(docs) == 1
                assert docs[0]["title"] == "Test Doc"
                assert len(docs[0]["sections"]) == 3  # intro + 2 sections

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.knowledge.loader.DOCS_DIR", Path(tmpdir)):
                docs = load_all_docs()
                assert docs == []

    def test_nonexistent_dir(self):
        with patch("src.knowledge.loader.DOCS_DIR", Path("/nonexistent/path")):
            docs = load_all_docs()
            assert docs == []


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearchDocs:
    @pytest.fixture
    def sample_docs(self):
        return [
            {
                "title": "KPI Definitions",
                "content": "",
                "sections": [
                    {"heading": "WAPE", "body": "Weighted Absolute Percentage Error measures forecast accuracy"},
                    {"heading": "MAPE", "body": "Mean Absolute Percentage Error is a common forecast metric"},
                ],
                "filename": "kpi_definitions.md",
            },
            {
                "title": "Data Dictionary",
                "content": "",
                "sections": [
                    {"heading": "Columns", "body": "oil gas water wellbore date on_stream"},
                ],
                "filename": "data_dictionary.md",
            },
        ]

    def test_finds_relevant_section(self, sample_docs):
        results = search_docs("weighted absolute percentage error", docs=sample_docs)
        assert len(results) > 0
        assert results[0]["heading"] == "WAPE"

    def test_no_match_returns_empty(self, sample_docs):
        results = search_docs("xyzzy", docs=sample_docs)
        assert results == []

    def test_respects_top_k(self, sample_docs):
        results = search_docs("forecast metric", top_k=1, docs=sample_docs)
        assert len(results) <= 1

    def test_result_has_expected_keys(self, sample_docs):
        results = search_docs("WAPE", docs=sample_docs)
        if results:
            r = results[0]
            assert "title" in r
            assert "heading" in r
            assert "snippet" in r
            assert "score" in r
