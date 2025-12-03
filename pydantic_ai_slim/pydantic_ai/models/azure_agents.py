"""Azure AI Agents model implementation.

This module provides integration with Azure AI Agents service (Azure AI Foundry Agents).
Azure AI Agents is a stateful agent orchestration service that uses threads and runs,
similar to OpenAI's Assistants API.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import anyio

from .. import usage
from .._utils import now_utc as _now_utc
from ..exceptions import UserError
from ..messages import (
    FinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..profiles import ModelProfileSpec
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    check_allow_model_requests,
)

try:
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import (
        AgentThread,
        ThreadMessage,
        ThreadRun,
    )
    from azure.identity import DefaultAzureCredential
except ImportError as _import_error:
    raise ImportError(
        'Please install `azure-ai-projects` and `azure-identity` to use Azure AI Agents, '
        'you can use the `azure-agents` optional group — `pip install "pydantic-ai-slim[azure-agents]"`'
    ) from _import_error

__all__ = (
    'AzureAgentsModel',
    'AzureAgentsModelSettings',
)


class AzureAgentsModelSettings(ModelSettings, total=False):
    """Settings used for Azure AI Agents model requests."""

    # ALL FIELDS MUST BE `azure_agents_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    azure_agents_temperature: float
    azure_agents_top_p: float
    azure_agents_max_tokens: int
    azure_agents_additional_instructions: str


@dataclass(init=False)
class AzureAgentsModel(Model):
    """A model that uses Azure AI Agents service.

    Azure AI Agents is a stateful agent orchestration service in Azure AI Foundry.
    It manages threads (conversations) and runs (executions) similar to OpenAI's Assistants API.

    Note: Each request creates a new thread since pydantic-ai's Model interface is stateless.
    For long-running conversations, consider using the Azure AI Agents SDK directly.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AIProjectClient = field(repr=False)
    agent_id: str | None = field(default=None, repr=False)
    _model_name: str = field(repr=False)
    _project_endpoint: str = field(repr=False)

    def __init__(
        self,
        model_name: str,
        *,
        project_endpoint: str | None = None,
        agent_id: str | None = None,
        credential: Any | None = None,
        client: AIProjectClient | None = None,
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an Azure AI Agents model.

        Args:
            model_name: The name/deployment name of the model to use in Azure AI Foundry.
            project_endpoint: The Azure AI Foundry project endpoint URL.
                If not provided, the `PROJECT_ENDPOINT` environment variable will be used.
            agent_id: Optional pre-existing agent ID to reuse. If not provided, a new agent
                will be created for each agent execution.
            credential: Azure credential object. If not provided, DefaultAzureCredential() will be used.
            client: An existing AIProjectClient instance. If provided, project_endpoint and credential
                must be None.
            profile: Optional model profile specification.
            settings: Optional model settings.
        """
        self._model_name = model_name
        self.agent_id = agent_id

        if client is not None:
            if project_endpoint is not None:
                raise UserError('Cannot provide both `client` and `project_endpoint`')
            if credential is not None:
                raise UserError('Cannot provide both `client` and `credential`')
            self.client = client
            # Try to get endpoint from client if available
            self._project_endpoint = getattr(client, '_endpoint', 'unknown')
        else:
            import os

            project_endpoint = project_endpoint or os.getenv('PROJECT_ENDPOINT')
            if not project_endpoint:
                raise UserError(
                    'Must provide one of the `project_endpoint` argument or the `PROJECT_ENDPOINT` environment variable'
                )

            credential = credential or DefaultAzureCredential()
            self.client = AIProjectClient(
                endpoint=project_endpoint,
                credential=credential,
            )
            self._project_endpoint = project_endpoint

        self._profile = profile
        self._settings = settings

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider system name."""
        return 'azure_agents'

    async def request(
        self,
        messages: list[ModelMessage],
        *,
        model_settings: ModelSettings | None = None,
        model_request_params: ModelRequestParameters | None = None,
    ) -> ModelResponse:
        """Make a request to Azure AI Agents."""
        check_allow_model_requests()

        # Merge settings
        settings = {**(self._settings or {}), **(model_settings or {})}

        # Extract system prompt and instructions
        instructions = self._extract_instructions(messages)

        # Create or reuse agent
        if self.agent_id:
            agent_id = self.agent_id
        else:
            # Create a temporary agent for this request
            agent = await self._create_agent(
                instructions=instructions,
                tools=model_request_params.tools if model_request_params else None,
            )
            agent_id = agent.id

        try:
            # Create thread
            thread = await self._create_thread()

            # Add user messages to thread
            await self._add_messages_to_thread(thread.id, messages)

            # Create and process run
            additional_instructions = settings.get('azure_agents_additional_instructions')
            run = await self._create_and_process_run(
                thread_id=thread.id,
                agent_id=agent_id,
                additional_instructions=additional_instructions,
            )

            # Get response messages
            response_messages = await self._get_thread_messages(thread.id)

            # Convert to ModelResponse
            return self._build_model_response(response_messages, run)

        finally:
            # Clean up temporary agent if we created one
            if not self.agent_id and agent_id:
                try:
                    await self._delete_agent(agent_id)
                except Exception:
                    # Best effort cleanup
                    pass

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        *,
        model_settings: ModelSettings | None = None,
        model_request_params: ModelRequestParameters | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Stream a request to Azure AI Agents.

        Note: Azure AI Agents doesn't support true streaming, so this simulates streaming
        by yielding the complete response at once.
        """
        check_allow_model_requests()

        # For now, we'll make a regular request and yield it all at once
        response = await self.request(
            messages, model_settings=model_settings, model_request_params=model_request_params
        )

        # Create a streamed response wrapper
        streamed = _AzureAgentsStreamedResponse(
            response=response,
            model_name=self._model_name,
            timestamp=response.timestamp,
        )

        yield streamed

    def _extract_instructions(self, messages: list[ModelMessage]) -> str:
        """Extract system instructions from messages."""
        instructions_parts = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        instructions_parts.append(part.content)

        return ' '.join(instructions_parts) if instructions_parts else 'You are a helpful assistant.'

    async def _create_agent(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None = None,
    ) -> Any:
        """Create a new agent in Azure AI Agents."""
        # TODO: Convert pydantic-ai tools to Azure AI Agents tool definitions
        # For now, create a basic agent without custom tools
        agent = await anyio.to_thread.run_sync(
            lambda: self.client.agents.create_agent(
                model=self._model_name,
                name='pydantic-ai-agent',
                instructions=instructions,
            )
        )
        return agent

    async def _create_thread(self) -> AgentThread:
        """Create a new thread for conversation."""
        return await anyio.to_thread.run_sync(lambda: self.client.agents.create_thread())

    async def _add_messages_to_thread(
        self,
        thread_id: str,
        messages: list[ModelMessage],
    ) -> None:
        """Add messages to the thread."""
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, UserPromptPart):
                        await anyio.to_thread.run_sync(
                            lambda: self.client.agents.messages.create(
                                thread_id=thread_id,
                                role='user',
                                content=part.content,
                            )
                        )
                    elif isinstance(part, ToolReturnPart):
                        # Tool returns are typically handled automatically
                        # by the agent service, but we can add them as user messages
                        await anyio.to_thread.run_sync(
                            lambda: self.client.agents.messages.create(
                                thread_id=thread_id,
                                role='user',
                                content=f'Tool {part.tool_name} returned: {part.content}',
                            )
                        )

    async def _create_and_process_run(
        self,
        thread_id: str,
        agent_id: str,
        additional_instructions: str | None = None,
    ) -> ThreadRun:
        """Create and wait for a run to complete."""
        run = await anyio.to_thread.run_sync(
            lambda: self.client.agents.create_and_process_run(
                thread_id=thread_id,
                agent_id=agent_id,
                additional_instructions=additional_instructions,
            )
        )
        return run

    async def _get_thread_messages(self, thread_id: str) -> list[ThreadMessage]:
        """Get messages from the thread."""
        messages = await anyio.to_thread.run_sync(lambda: self.client.agents.messages.list(thread_id=thread_id))
        # Return messages in chronological order (latest first from API, so reverse)
        return list(reversed(list(messages)))

    async def _delete_agent(self, agent_id: str) -> None:
        """Delete a temporary agent."""
        await anyio.to_thread.run_sync(lambda: self.client.agents.delete_agent(agent_id))

    def _build_model_response(
        self,
        messages: list[ThreadMessage],
        run: ThreadRun,
    ) -> ModelResponse:
        """Build a ModelResponse from Azure AI Agents thread messages."""
        # Get the last assistant message
        parts: list[ModelResponsePart] = []

        for message in messages:
            if message.role == 'assistant':
                # Add text content
                if hasattr(message, 'content') and message.content:
                    for content_item in message.content:
                        if hasattr(content_item, 'text'):
                            parts.append(TextPart(content=content_item.text.value))

        # If no parts found, add empty text
        if not parts:
            parts.append(TextPart(content=''))

        # Determine finish reason
        finish_reason: FinishReason
        if run.status == 'completed':
            finish_reason = 'stop'
        elif run.status == 'failed':
            finish_reason = 'error'
        else:
            finish_reason = 'stop'

        return ModelResponse(
            parts=parts,
            model_name=self._model_name,
            timestamp=_now_utc(),
            usage=usage.RequestUsage(),  # TODO: Extract usage from run if available
            provider_name='azure-agents',
            finish_reason=finish_reason,
        )


@dataclass
class _AzureAgentsStreamedResponse(StreamedResponse):
    """Streamed response wrapper for Azure AI Agents.

    Since Azure AI Agents doesn't support true streaming, this yields the complete response.
    """

    response: ModelResponse = field(repr=False)
    _model_name: str = field(repr=False)
    _timestamp: datetime = field(repr=False)

    def __init__(
        self,
        response: ModelResponse,
        model_name: str,
        timestamp: datetime,
    ):
        self.response = response
        self._model_name = model_name
        self._timestamp = timestamp

    async def _process_stream(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Yield the complete response as stream events."""
        # Yield all parts at once
        for part in self.response.parts:
            if isinstance(part, TextPart):
                # Yield as a complete text chunk
                from ..messages import PartStartEvent

                yield PartStartEvent(part=part, index=0)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str | None:
        return 'azure-agents'

    @property
    def timestamp(self) -> datetime:
        return self._timestamp
