"""
LLM Copilot Providers
======================

AzureOpenAIProvider -- placeholder (not yet implemented).
GoogleGeminiProvider -- Google Gemini-powered copilot for natural-language
    production Q&A. Requires GEMINI_API_KEY in environment.
"""

from typing import Dict, List, Optional

from .provider import CopilotProvider
from ..config import GEMINI_API_KEY


# ---------------------------------------------------------------------------
# Azure OpenAI -- placeholder
# ---------------------------------------------------------------------------


class AzureOpenAIProvider(CopilotProvider):
    """
    Placeholder for Azure OpenAI integration.

    TODO: Implement using azure-openai SDK.
    Requires: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT
    """

    @property
    def name(self) -> str:
        return "Azure OpenAI (not configured)"

    @property
    def is_available(self) -> bool:
        return False

    def answer(
        self,
        query: str,
        context: Dict,
        docs: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        raise NotImplementedError("Azure OpenAI provider is not yet implemented.")


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------


class GoogleGeminiProvider(CopilotProvider):
    """Google Gemini-powered copilot for natural-language production Q&A."""

    def __init__(self):
        self._model = None

    @property
    def name(self) -> str:
        return "Google Gemini"

    @property
    def is_available(self) -> bool:
        return bool(GEMINI_API_KEY)

    def _get_model(self):
        """Lazy-load the Gemini model (avoids import crash when SDK absent)."""
        if self._model is None:
            import google.generativeai as genai

            genai.configure(api_key=GEMINI_API_KEY)
            self._model = genai.GenerativeModel("gemini-1.5-flash")
        return self._model

    def _build_prompt(
        self,
        query: str,
        context: Dict,
        docs: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Assemble a structured prompt from dashboard context."""
        parts = [
            "You are an oil & gas operations analyst for the Volve field "
            "(North Sea, 2008-2016). Answer concisely using the data below. "
            "Cite numbers when available.\n",
        ]

        # KPI summary
        summary = context.get("summary", {})
        if summary:
            parts.append("--- KPI Summary ---")
            parts.append(f"Report month: {summary.get('report_month', 'N/A')}")
            parts.append(f"Total oil: {summary.get('total_oil', 0):,.0f} Sm3")
            parts.append(f"Total gas: {summary.get('total_gas', 0):,.0f} Sm3")
            parts.append(f"Total water: {summary.get('total_water', 0):,.0f} Sm3")
            parts.append(f"MoM change: {summary.get('mom_change_pct', 0):.1f}%")
            parts.append(f"YoY change: {summary.get('yoy_change_pct', 0):.1f}%")

        # Top wellbores
        top = context.get("top_wellbores", [])
        if top:
            parts.append("\n--- Top Wellbores ---")
            for w in top[:5]:
                parts.append(
                    f"  {w['wellbore']}: oil={w.get('oil', 0):,.0f}, "
                    f"gas={w.get('gas', 0):,.0f}, water={w.get('water', 0):,.0f}"
                )

        # Anomalies
        anomalies = context.get("anomalies", [])
        zscore = context.get("zscore_anomalies", [])
        parts.append(f"\n--- Anomalies ---")
        parts.append(f"Wellbore anomalies: {len(anomalies)}")
        parts.append(f"Z-score anomaly points: {len(zscore)}")

        # Forecast
        fc = context.get("forecast", {})
        if fc:
            parts.append(f"\n--- Forecast ---")
            parts.append(f"Model: {fc.get('model', 'N/A')}")
            parts.append(f"Next month forecast: {fc.get('yhat_next', 0):,.0f} Sm3")

        # Backtest metrics
        bt = context.get("backtest_metrics", {})
        if bt:
            parts.append(f"\n--- Backtest Metrics (ETS) ---")
            parts.append(f"MAPE: {bt.get('mape', 'N/A')}%")
            parts.append(f"WAPE: {bt.get('wape', 'N/A')}%")
            parts.append(f"MAE: {bt.get('mae', 'N/A')}")
            parts.append(f"RMSE: {bt.get('rmse', 'N/A')}")

        comp = context.get("comparison_metrics", {})
        if comp:
            parts.append(f"\n--- Baseline Metrics ---")
            parts.append(f"MAPE: {comp.get('mape', 'N/A')}%")
            parts.append(f"WAPE: {comp.get('wape', 'N/A')}%")

        # Knowledge base docs
        if docs:
            parts.append("\n--- Knowledge Base ---")
            for d in docs[:3]:
                parts.append(
                    f"[{d['title']} > {d['heading']}]: {d['snippet']}"
                )

        parts.append(f"\n--- User Question ---\n{query}")
        return "\n".join(parts)

    def answer(
        self,
        query: str,
        context: Dict,
        docs: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        model = self._get_model()
        prompt = self._build_prompt(query, context, docs)
        response = model.generate_content(prompt)
        sources = ["Google Gemini"]
        if docs:
            sources.extend(d["title"] for d in docs[:3])
        return {
            "answer": response.text,
            "sources": sources,
            "action": None,
        }
