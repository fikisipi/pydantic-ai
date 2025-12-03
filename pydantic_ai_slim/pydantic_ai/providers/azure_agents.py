"""Provider for Azure AI Agents."""

from __future__ import annotations as _annotations

import os
from typing import Any

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers import Provider

try:
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install `azure-ai-projects` and `azure-identity` to use the Azure AI Agents provider, '
        'you can use the `azure-agents` optional group — `pip install "pydantic-ai-slim[azure-agents]"`'
    ) from _import_error


class AzureAgentsProvider(Provider[AIProjectClient]):
    """Provider for Azure AI Agents service.

    See <https://azure.microsoft.com/en-us/products/ai-foundry> for more information.
    """

    @property
    def name(self) -> str:
        return 'azure-agents'

    @property
    def base_url(self) -> str:
        """Return the project endpoint as the base URL."""
        return self._project_endpoint

    @property
    def client(self) -> AIProjectClient:
        return self._client

    def __init__(
        self,
        *,
        project_endpoint: str | None = None,
        credential: Any | None = None,
        client: AIProjectClient | None = None,
    ) -> None:
        """Create a new Azure AI Agents provider.

        Args:
            project_endpoint: The Azure AI Foundry project endpoint URL.
                If not provided, the `PROJECT_ENDPOINT` environment variable will be used.
            credential: Azure credential object. If not provided, DefaultAzureCredential() will be used.
            client: An existing `AIProjectClient` instance.
                If provided, `project_endpoint` and `credential` must be `None`.
        """
        if client is not None:
            if project_endpoint is not None:
                raise UserError('Cannot provide both `client` and `project_endpoint`')
            if credential is not None:
                raise UserError('Cannot provide both `client` and `credential`')
            self._client = client
            # Try to extract endpoint if available
            self._project_endpoint = getattr(client, '_endpoint', 'unknown')
        else:
            project_endpoint = project_endpoint or os.getenv('PROJECT_ENDPOINT')
            if not project_endpoint:
                raise UserError(
                    'Must provide one of the `project_endpoint` argument or the `PROJECT_ENDPOINT` environment variable'
                )

            credential = credential or DefaultAzureCredential()
            self._client = AIProjectClient(
                endpoint=project_endpoint,
                credential=credential,
            )
            self._project_endpoint = project_endpoint
