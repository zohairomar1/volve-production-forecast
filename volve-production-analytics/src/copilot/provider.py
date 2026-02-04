"""
Copilot Provider Interface
===========================

Abstract base class for copilot engines. The rule-based engine is the
default and only active provider. LLM providers are stubbed for future use.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class CopilotProvider(ABC):
    """
    Abstract base class for copilot response generation.

    All providers must implement the ``answer`` method, which receives
    a structured context dict and a user query string, and returns
    a structured response dict.
    """

    @abstractmethod
    def answer(
        self,
        query: str,
        context: Dict,
        docs: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        """
        Generate an answer to the user's query.

        Parameters
        ----------
        query : str
            Natural-language question or action request.
        context : Dict
            Runtime data with keys: summary, top_wellbores, anomalies,
            zscore_anomalies, forecast, backtest_metrics, comparison_metrics,
            filters.
        docs : list of dict, optional
            Retrieved doc snippets from knowledge base, each with
            keys ``title``, ``heading``, ``snippet``, ``score``.

        Returns
        -------
        Dict
            ``answer`` (str), ``sources`` (list[str]), ``action`` (str or None).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider is configured and ready to use."""
        ...
