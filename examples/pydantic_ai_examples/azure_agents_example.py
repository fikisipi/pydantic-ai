"""Example of using Azure AI Agents with pydantic-ai.

This example demonstrates how to use Azure AI Agents service with pydantic-ai.
Azure AI Agents is part of Azure AI Foundry and provides managed agent orchestration.

Prerequisites:
- Azure AI Foundry project with an endpoint
- Azure credentials configured (via Azure CLI or environment variables)
- Set PROJECT_ENDPOINT environment variable
- Set MODEL_DEPLOYMENT_NAME environment variable

To run this example:
1. Install azure-agents dependencies: pip install "pydantic-ai-slim[azure-agents]"
2. Set up Azure credentials: az login
3. Set environment variables:
   export PROJECT_ENDPOINT="https://your-project.api.azureml.ms"
   export MODEL_DEPLOYMENT_NAME="gpt-4o"
4. Run: python azure_agents_example.py
"""

import os

from pydantic_ai import Agent


def main():
    # Get configuration from environment
    project_endpoint = os.environ.get('PROJECT_ENDPOINT')
    model_deployment = os.environ.get('MODEL_DEPLOYMENT_NAME', 'gpt-4o')

    if not project_endpoint:
        print('Error: PROJECT_ENDPOINT environment variable not set')
        print('Please set it to your Azure AI Foundry project endpoint')
        return

    # Create an agent using Azure AI Agents
    # The model string format is "azure-agents:deployment-name"
    agent = Agent(
        f'azure-agents:{model_deployment}',
        instructions='You are a helpful AI assistant. Be concise and friendly.',
    )

    # Run a simple query
    print('Sending query to Azure AI Agents...')
    result = agent.run_sync('What is the capital of France?')

    print(f'\nResponse: {result.output}')


if __name__ == '__main__':
    main()
