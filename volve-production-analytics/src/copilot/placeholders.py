"""
Placeholder LLM Providers
===========================

These providers are NOT functional. They exist to demonstrate the
vendor-agnostic interface design. Swap in real credentials and SDK
calls to enable.
"""

from typing import Dict, List, Optional

from .provider import CopilotProvider


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


class GoogleVertexProvider(CopilotProvider):
    """
    Placeholder for Google Vertex AI / Gemini integration.

    TODO: Implement using google-cloud-aiplatform SDK.
    Requires: GOOGLE_PROJECT_ID, GOOGLE_LOCATION, GOOGLE_MODEL_ID
    """

    @property
    def name(self) -> str:
        return "Google Vertex AI (not configured)"

    @property
    def is_available(self) -> bool:
        return False

    def answer(
        self,
        query: str,
        context: Dict,
        docs: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        raise NotImplementedError("Google Vertex AI provider is not yet implemented.")
