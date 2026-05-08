"""
azure_connect.py — One-time connection test. Run to verify your credentials.
Usage: python azureml/azure_connect.py
"""

import os
from dotenv import load_dotenv
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient

# Loat environment variables from .env file
load_dotenv()

# Get Azure credentials and workspace details from environment variables
credential = InteractiveBrowserCredential()

# Azure subscription ID, resource group, and workspace name
subcription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

# Create MLClient to connect to Azure ML workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=subcription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)

workspace = ml_client.workspaces.get(workspace_name)
print(f"Connected to Azure ML Workspace: {workspace.name}")
print(f"Workspace Location: {workspace.location}")
print(f"Resource Group: {workspace.resource_group}")