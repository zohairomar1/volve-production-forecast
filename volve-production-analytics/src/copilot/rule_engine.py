"""
Rule-Based Copilot Engine
==========================

Deterministic copilot that answers questions using pattern matching
and direct data lookups. No external LLM required.
"""

import re
from typing import Dict, List, Optional

from .provider import CopilotProvider

# Intent patterns: (compiled regex, intent_key)
INTENT_PATTERNS = [
    (re.compile(r"weekly\s*(ops)?\s*summary|summarize|overview", re.I), "weekly_summary"),
    (re.compile(r"why.*(forecast|changed|change)|forecast\s*driver|what\s*drove", re.I), "forecast_drivers"),
    (re.compile(r"draft\s*email|email\s*summary|send\s*report", re.I), "draft_email"),
    (re.compile(r"anomal|unusual|outlier|flag", re.I), "anomalies"),
    (re.compile(r"kpi|metric|production\s*(total|number|volume)", re.I), "kpis"),
    (re.compile(r"forecast|predict|next\s*month|future", re.I), "forecast"),
    (re.compile(r"backtest|model\s*(performance|accuracy|validation)|mape|wape|mae|rmse", re.I), "backtest"),
    (re.compile(r"top\s*well|best\s*well|highest\s*produc", re.I), "top_wells"),
]


class RuleBasedProvider(CopilotProvider):
    """Rule-based copilot that generates answers from structured data."""

    @property
    def name(self) -> str:
        return "Rule-Based Engine"

    @property
    def is_available(self) -> bool:
        return True

    def answer(
        self,
        query: str,
        context: Dict,
        docs: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        intent = self._classify_intent(query)
        handler = {
            "weekly_summary": self._weekly_summary,
            "forecast_drivers": self._forecast_drivers,
            "draft_email": self._draft_email,
            "anomalies": self._anomalies,
            "kpis": self._kpis,
            "forecast": self._forecast,
            "backtest": self._backtest,
            "top_wells": self._top_wells,
        }.get(intent, self._fallback)

        result = handler(context)

        # Append doc citations if knowledge base returned results
        if docs:
            result["sources"].extend(
                f"{d['title']} > {d['heading']}" for d in docs[:3]
            )
            result["answer"] += "\n\nRelated documentation:\n"
            for d in docs[:3]:
                result["answer"] += f"  - {d['title']} > {d['heading']}: {d['snippet']}\n"

        return result

    def _classify_intent(self, query: str) -> str:
        for pattern, intent in INTENT_PATTERNS:
            if pattern.search(query):
                return intent
        return "unknown"

    # -----------------------------------------------------------------
    # Intent handlers
    # -----------------------------------------------------------------

    def _weekly_summary(self, ctx: Dict) -> Dict:
        s = ctx.get("summary", {})
        bullets = []
        bullets.append(f"Report period: {s.get('report_month', 'N/A')}")
        bullets.append(f"Total oil production: {s.get('total_oil', 0):,.0f} Sm3")

        mom = s.get("mom_change_pct")
        if mom is not None:
            direction = "up" if mom > 0 else "down"
            bullets.append(f"Oil MoM change: {mom:+.1f}% ({direction} from prior month)")

        yoy = s.get("yoy_change_pct")
        if yoy is not None:
            bullets.append(f"Oil YoY change: {yoy:+.1f}%")

        anomalies = ctx.get("anomalies", [])
        bullets.append(f"Anomalies detected (rolling-avg method): {len(anomalies)}")

        zs_anomalies = ctx.get("zscore_anomalies", [])
        bullets.append(f"Z-score anomalies flagged: {len(zs_anomalies)}")

        fc = ctx.get("forecast", {})
        if fc.get("yhat_next"):
            bullets.append(
                f"Next-month forecast ({fc.get('model', 'N/A')}): "
                f"{fc['yhat_next']:,.0f} Sm3"
            )

        metrics = ctx.get("backtest_metrics", {})
        wape = metrics.get("wape")
        if wape is not None:
            bullets.append(f"Model WAPE: {wape:.1f}%")

        return {
            "answer": "Weekly Operations Summary\n" + "\n".join(f"  - {b}" for b in bullets),
            "sources": ["reporting.get_last_month_summary", "reporting.detect_anomalies"],
            "action": "weekly_summary",
        }

    def _forecast_drivers(self, ctx: Dict) -> Dict:
        metrics = ctx.get("backtest_metrics", {})
        comp = ctx.get("comparison_metrics", {})
        s = ctx.get("summary", {})

        lines = ["Forecast Change Drivers:"]
        mom = s.get("mom_change_pct")
        if mom is not None:
            lines.append(f"  - Recent MoM trend: {mom:+.1f}% (influences ETS trend component)")
        yoy = s.get("yoy_change_pct")
        if yoy is not None:
            lines.append(f"  - YoY trajectory: {yoy:+.1f}% (influences seasonal component)")

        wape_ets = metrics.get("wape")
        wape_base = comp.get("wape") if comp else None
        if wape_ets is not None and wape_base is not None:
            if wape_ets < wape_base:
                lines.append(
                    f"  - ETS outperforms baseline (WAPE {wape_ets:.1f}% vs {wape_base:.1f}%), "
                    "suggesting trend/seasonality capture adds value"
                )
            else:
                lines.append(
                    f"  - Baseline competitive with ETS (WAPE {wape_base:.1f}% vs {wape_ets:.1f}%), "
                    "suggesting limited additional trend signal"
                )

        anomalies = ctx.get("anomalies", [])
        if anomalies:
            lines.append(
                f"  - {len(anomalies)} anomalous wellbore(s) detected, which may shift aggregate forecast"
            )

        return {
            "answer": "\n".join(lines),
            "sources": ["evaluation.compute_backtest_metrics", "reporting.get_last_month_summary"],
            "action": "forecast_drivers",
        }

    def _draft_email(self, ctx: Dict) -> Dict:
        s = ctx.get("summary", {})
        top = ctx.get("top_wellbores", [])
        fc = ctx.get("forecast", {})

        lines = [
            f"Subject: Volve Production Report - {s.get('report_month', 'N/A')}",
            "",
            f"Total oil production for {s.get('report_month', 'the latest period')}: "
            f"{s.get('total_oil', 0):,.0f} Sm3.",
        ]
        mom = s.get("mom_change_pct")
        if mom is not None:
            lines.append(f"Month-over-month change: {mom:+.1f}%.")
        if top:
            lines.append(
                f"Top producer: {top[0].get('wellbore', 'N/A')} "
                f"({top[0].get('oil', 0):,.0f} Sm3)."
            )
        if fc.get("yhat_next"):
            lines.append(f"Next-month forecast: {fc['yhat_next']:,.0f} Sm3.")
        lines.append("")
        lines.append("Full details available on the dashboard.")

        return {
            "answer": "\n".join(lines),
            "sources": ["reporting.generate_email_summary"],
            "action": "draft_email",
        }

    def _anomalies(self, ctx: Dict) -> Dict:
        anomalies = ctx.get("anomalies", [])
        zs = ctx.get("zscore_anomalies", [])
        if not anomalies and not zs:
            text = "No anomalies detected in the current view."
        else:
            lines = ["Anomaly Summary:"]
            if anomalies:
                lines.append(f"  - {len(anomalies)} wellbore(s) with >30% drop vs rolling average:")
                for a in anomalies[:5]:
                    lines.append(f"    - {a.get('wellbore')}: {a.get('pct_change', 0):.1f}% change")
            if zs:
                lines.append(f"  - {len(zs)} points flagged by z-score method")
            text = "\n".join(lines)

        return {"answer": text, "sources": ["reporting.detect_anomalies"], "action": None}

    def _kpis(self, ctx: Dict) -> Dict:
        s = ctx.get("summary", {})
        mom = s.get("mom_change_pct")
        yoy = s.get("yoy_change_pct")
        text = (
            f"KPIs for {s.get('report_month', 'N/A')}:\n"
            f"  - Oil: {s.get('total_oil', 0):,.0f} Sm3 "
            f"(MoM: {f'{mom:+.1f}' if mom is not None else 'N/A'}%)\n"
            f"  - Gas: {s.get('total_gas', 0):,.0f} Sm3\n"
            f"  - Water: {s.get('total_water', 0):,.0f} Sm3\n"
            f"  - YoY oil change: {f'{yoy:+.1f}' if yoy is not None else 'N/A'}%"
        )
        return {"answer": text, "sources": ["reporting.get_last_month_summary"], "action": None}

    def _forecast(self, ctx: Dict) -> Dict:
        fc = ctx.get("forecast", {})
        if not fc:
            return {"answer": "No forecast data available.", "sources": [], "action": None}
        text = f"Forecast ({fc.get('model', 'N/A')}):\n"
        if fc.get("yhat_next"):
            text += f"  - Next month: {fc['yhat_next']:,.0f} Sm3\n"
        values = fc.get("values", [])
        if values:
            text += "  - Horizon:\n"
            for v in values[:6]:
                text += f"    - {v.get('date', '')}: {v.get('yhat', 0):,.0f} Sm3\n"
        return {"answer": text, "sources": ["forecasting.forecast_series"], "action": None}

    def _backtest(self, ctx: Dict) -> Dict:
        m = ctx.get("backtest_metrics", {})
        if not m:
            return {"answer": "No backtest metrics available.", "sources": [], "action": None}

        def fmt(val):
            return f"{val:.1f}" if val is not None else "N/A"

        text = (
            f"Model Validation (rolling-origin backtest):\n"
            f"  - MAE: {m.get('mae', 0):,.0f} Sm3\n"
            f"  - RMSE: {m.get('rmse', 0):,.0f} Sm3\n"
            f"  - MAPE: {fmt(m.get('mape'))}%\n"
            f"  - WAPE: {fmt(m.get('wape'))}%\n"
            f"  - Observations: {m.get('n_observations', 0)}"
        )
        return {"answer": text, "sources": ["evaluation.compute_backtest_metrics"], "action": None}

    def _top_wells(self, ctx: Dict) -> Dict:
        top = ctx.get("top_wellbores", [])
        if not top:
            return {"answer": "No wellbore data available.", "sources": [], "action": None}
        lines = ["Top Producing Wellbores (latest month):"]
        for i, w in enumerate(top[:5], 1):
            lines.append(f"  {i}. {w.get('wellbore', 'N/A')}: {w.get('oil', 0):,.0f} Sm3 oil")
        return {"answer": "\n".join(lines), "sources": ["reporting.get_top_wellbores"], "action": None}

    def _fallback(self, ctx: Dict) -> Dict:
        return {
            "answer": (
                "I can help with:\n"
                "  - Weekly ops summary\n"
                "  - KPI overview\n"
                "  - Anomaly analysis\n"
                "  - Forecast details and drivers\n"
                "  - Model validation metrics\n"
                "  - Top wellbore rankings\n"
                "  - Draft email summary\n\n"
                "Try asking one of these, or rephrase your question."
            ),
            "sources": [],
            "action": None,
        }
