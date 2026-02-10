"""Tests for the copilot module."""

import pytest
from unittest.mock import patch, MagicMock

from src.copilot import get_active_provider
from src.copilot.provider import CopilotProvider
from src.copilot.rule_engine import RuleBasedProvider, INTENT_PATTERNS
from src.copilot.placeholders import AzureOpenAIProvider, GoogleGeminiProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_context():
    """Realistic context dict matching dashboard runtime data."""
    return {
        "summary": {
            "report_month": "September 2015",
            "total_oil": 45000.0,
            "total_gas": 3200000.0,
            "total_water": 12000.0,
            "mom_change_pct": -5.2,
            "yoy_change_pct": -18.3,
        },
        "top_wellbores": [
            {"wellbore": "15/9-F-11", "oil": 25000, "gas": 1800000, "water": 5000},
            {"wellbore": "15/9-F-1 C", "oil": 12000, "gas": 900000, "water": 4000},
        ],
        "anomalies": [
            {"wellbore": "15/9-F-1 B", "current_oil": 800, "rolling_avg": 2000, "pct_change": -60.0},
        ],
        "zscore_anomalies": [
            {"date": "2015-09-01", "oil": 200, "zscore": -3.1, "is_anomaly": True},
        ],
        "forecast": {
            "model": "ets",
            "yhat_next": 42000.0,
            "values": [
                {"date": "2015-10", "yhat": 42000.0},
                {"date": "2015-11", "yhat": 40000.0},
            ],
        },
        "backtest_metrics": {
            "mape": 12.5,
            "wape": 10.8,
            "mae": 4500,
            "rmse": 5200,
            "n_observations": 12,
        },
        "comparison_metrics": {
            "mape": 15.0,
            "wape": 13.2,
            "mae": 5800,
            "rmse": 6500,
            "n_observations": 12,
        },
        "filters": {"mode": "Total Field", "wellbore": "All", "model": "ets"},
    }


@pytest.fixture
def provider():
    return RuleBasedProvider()


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    def test_defaults_to_rule_based(self):
        provider = get_active_provider()
        assert isinstance(provider, RuleBasedProvider)

    def test_azure_not_available(self):
        assert AzureOpenAIProvider().is_available is False

    def test_gemini_not_available_when_no_key(self):
        with patch("src.copilot.placeholders.GEMINI_API_KEY", ""):
            p = GoogleGeminiProvider()
            assert p.is_available is False

    def test_azure_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            AzureOpenAIProvider().answer("test", {})

    def test_gemini_name(self):
        assert GoogleGeminiProvider().name == "Google Gemini"


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

class TestGeminiProvider:
    def test_available_when_key_set(self):
        with patch("src.copilot.placeholders.GEMINI_API_KEY", "test-key-123"):
            p = GoogleGeminiProvider()
            assert p.is_available is True

    def test_not_available_when_no_key(self):
        with patch("src.copilot.placeholders.GEMINI_API_KEY", ""):
            p = GoogleGeminiProvider()
            assert p.is_available is False

    def test_answer_returns_expected_format(self, sample_context):
        mock_response = MagicMock()
        mock_response.text = "Oil production declined 5.2% month-over-month."

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        with patch("src.copilot.placeholders.GEMINI_API_KEY", "test-key-123"):
            p = GoogleGeminiProvider()
            p._model = mock_model

            result = p.answer("weekly summary", sample_context)
            assert "answer" in result
            assert "sources" in result
            assert "action" in result
            assert result["sources"][0] == "Google Gemini"
            assert result["action"] is None
            assert "declined" in result["answer"]

    def test_prompt_includes_context(self, sample_context):
        with patch("src.copilot.placeholders.GEMINI_API_KEY", "test-key-123"):
            p = GoogleGeminiProvider()
            prompt = p._build_prompt("show KPIs", sample_context)
            assert "45,000" in prompt
            assert "September 2015" in prompt
            assert "show KPIs" in prompt

    def test_prompt_includes_docs(self, sample_context):
        docs = [
            {"title": "KPI Def", "heading": "WAPE", "snippet": "Weighted error", "score": 0.8},
        ]
        with patch("src.copilot.placeholders.GEMINI_API_KEY", "test-key-123"):
            p = GoogleGeminiProvider()
            prompt = p._build_prompt("what is WAPE", sample_context, docs=docs)
            assert "Knowledge Base" in prompt
            assert "KPI Def" in prompt
            assert "Weighted error" in prompt

    def test_answer_includes_doc_sources(self, sample_context):
        mock_response = MagicMock()
        mock_response.text = "WAPE is Weighted Absolute Percentage Error."

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        docs = [
            {"title": "KPI Def", "heading": "WAPE", "snippet": "Weighted error", "score": 0.8},
        ]
        with patch("src.copilot.placeholders.GEMINI_API_KEY", "test-key-123"):
            p = GoogleGeminiProvider()
            p._model = mock_model

            result = p.answer("what is WAPE", sample_context, docs=docs)
            assert "Google Gemini" in result["sources"]
            assert "KPI Def" in result["sources"]

    def test_get_active_provider_returns_gemini_when_key_set(self):
        with patch("src.copilot.placeholders.GEMINI_API_KEY", "test-key-123"):
            provider = get_active_provider()
            assert isinstance(provider, GoogleGeminiProvider)

    def test_prompt_handles_empty_context(self):
        with patch("src.copilot.placeholders.GEMINI_API_KEY", "test-key-123"):
            p = GoogleGeminiProvider()
            prompt = p._build_prompt("hello", {})
            assert "hello" in prompt
            assert "oil & gas" in prompt


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

class TestIntentClassification:
    def test_weekly_summary(self, provider):
        assert provider._classify_intent("weekly ops summary") == "weekly_summary"
        assert provider._classify_intent("give me an overview") == "weekly_summary"
        assert provider._classify_intent("summarize production") == "weekly_summary"

    def test_forecast_drivers(self, provider):
        assert provider._classify_intent("why did the forecast change") == "forecast_drivers"
        assert provider._classify_intent("what drove the forecast") == "forecast_drivers"

    def test_draft_email(self, provider):
        assert provider._classify_intent("draft email summary") == "draft_email"
        assert provider._classify_intent("send report") == "draft_email"

    def test_anomalies(self, provider):
        assert provider._classify_intent("any anomalies?") == "anomalies"
        assert provider._classify_intent("show flagged points") == "anomalies"

    def test_kpis(self, provider):
        assert provider._classify_intent("show me the KPIs") == "kpis"
        assert provider._classify_intent("production totals") == "kpis"

    def test_forecast(self, provider):
        assert provider._classify_intent("what is the forecast") == "forecast"
        assert provider._classify_intent("predict next month") == "forecast"

    def test_backtest(self, provider):
        assert provider._classify_intent("model performance") == "backtest"
        assert provider._classify_intent("what is the WAPE") == "backtest"

    def test_top_wells(self, provider):
        assert provider._classify_intent("top wells") == "top_wells"
        assert provider._classify_intent("highest producing") == "top_wells"

    def test_unknown_fallback(self, provider):
        assert provider._classify_intent("hello there") == "unknown"


# ---------------------------------------------------------------------------
# Handler outputs
# ---------------------------------------------------------------------------

class TestRuleBasedHandlers:
    def test_weekly_summary_has_bullets(self, provider, sample_context):
        result = provider.answer("weekly ops summary", sample_context)
        assert result["action"] == "weekly_summary"
        # Should have at least 5 bullet points
        bullet_count = result["answer"].count("  - ")
        assert bullet_count >= 5

    def test_draft_email_has_subject(self, provider, sample_context):
        result = provider.answer("draft email summary", sample_context)
        assert result["action"] == "draft_email"
        assert "Subject:" in result["answer"]
        assert "September 2015" in result["answer"]

    def test_kpis_include_values(self, provider, sample_context):
        result = provider.answer("show KPIs", sample_context)
        assert "45,000" in result["answer"]
        assert "September 2015" in result["answer"]

    def test_forecast_includes_model(self, provider, sample_context):
        result = provider.answer("what is the forecast", sample_context)
        assert "ets" in result["answer"].lower() or "42,000" in result["answer"]

    def test_backtest_includes_metrics(self, provider, sample_context):
        result = provider.answer("model performance", sample_context)
        assert "WAPE" in result["answer"]
        assert "10.8" in result["answer"]

    def test_top_wells_lists_names(self, provider, sample_context):
        result = provider.answer("top wells", sample_context)
        assert "15/9-F-11" in result["answer"]

    def test_anomalies_shows_count(self, provider, sample_context):
        result = provider.answer("any anomalies", sample_context)
        assert "1" in result["answer"]

    def test_forecast_drivers_compares_models(self, provider, sample_context):
        result = provider.answer("why did forecast change", sample_context)
        assert "ETS" in result["answer"] or "baseline" in result["answer"].lower()

    def test_fallback_lists_capabilities(self, provider, sample_context):
        result = provider.answer("hello world", sample_context)
        assert "I can help with" in result["answer"]
        assert result["action"] is None

    def test_empty_context_no_crash(self, provider):
        result = provider.answer("weekly ops summary", {})
        assert "answer" in result
        assert "sources" in result

    def test_doc_citations_appended(self, provider, sample_context):
        docs = [
            {"title": "KPI Def", "heading": "WAPE", "snippet": "Weighted error", "score": 0.8},
        ]
        result = provider.answer("what is WAPE", sample_context, docs=docs)
        assert "KPI Def" in result["answer"]
        assert "Related documentation:" in result["answer"]
