"""Tests for Azure AI Agents model integration."""

from __future__ import annotations as _annotations

from unittest.mock import MagicMock, patch

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from azure.ai.projects import AIProjectClient

    from pydantic_ai.models.azure_agents import AzureAgentsModel

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='azure-ai-projects not installed'),
]


def test_init_with_endpoint():
    """Test initializing with project endpoint."""
    with patch('pydantic_ai.models.azure_agents.AIProjectClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        model = AzureAgentsModel(
            model_name='gpt-4o',
            project_endpoint='https://test.api.azureml.ms',
        )

        assert model._model_name == 'gpt-4o'
        assert model._project_endpoint == 'https://test.api.azureml.ms'
        assert model.client == mock_client


def test_init_with_client():
    """Test initializing with existing client."""
    mock_client = MagicMock(spec=AIProjectClient)
    mock_client._endpoint = 'https://test.api.azureml.ms'

    model = AzureAgentsModel(
        model_name='gpt-4o',
        client=mock_client,
    )

    assert model._model_name == 'gpt-4o'
    assert model.client == mock_client


def test_init_missing_endpoint():
    """Test error when neither endpoint nor client provided."""
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(Exception):  # Should raise UserError
            AzureAgentsModel(model_name='gpt-4o')


def test_init_with_agent_id():
    """Test initializing with pre-existing agent ID."""
    with patch('pydantic_ai.models.azure_agents.AIProjectClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        model = AzureAgentsModel(
            model_name='gpt-4o',
            project_endpoint='https://test.api.azureml.ms',
            agent_id='test-agent-123',
        )

        assert model.agent_id == 'test-agent-123'


def test_extract_instructions():
    """Test extracting instructions from messages."""
    from pydantic_ai.messages import ModelRequest, SystemPromptPart

    with patch('pydantic_ai.models.azure_agents.AIProjectClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        model = AzureAgentsModel(
            model_name='gpt-4o',
            project_endpoint='https://test.api.azureml.ms',
        )

        messages = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a helpful assistant.'),
                ]
            )
        ]

        instructions = model._extract_instructions(messages)
        assert instructions == 'You are a helpful assistant.'


def test_extract_instructions_default():
    """Test default instructions when none provided."""
    with patch('pydantic_ai.models.azure_agents.AIProjectClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        model = AzureAgentsModel(
            model_name='gpt-4o',
            project_endpoint='https://test.api.azureml.ms',
        )

        instructions = model._extract_instructions([])
        assert instructions == 'You are a helpful assistant.'
