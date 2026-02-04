"""Copilot module -- Operations Copilot for the Volve dashboard."""

from .provider import CopilotProvider
from .rule_engine import RuleBasedProvider
from .placeholders import AzureOpenAIProvider, GoogleVertexProvider


def get_active_provider() -> CopilotProvider:
    """
    Return the first available provider, falling back to rule-based.

    Check order: AzureOpenAI -> GoogleVertex -> RuleBasedEngine
    """
    for provider_cls in [AzureOpenAIProvider, GoogleVertexProvider]:
        provider = provider_cls()
        if provider.is_available:
            return provider
    return RuleBasedProvider()
