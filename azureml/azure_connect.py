"""
azure_connect.py — One-time connection test. Run to verify your credentials.

Usage:
    python azureml/azure_connect.py

Requires a .env file in the repo root with:
    AZURE_SUBSCRIPTION_ID=<your-subscription-id>
    AZURE_RESOURCE_GROUP=<your-resource-group>
    AZURE_WORKSPACE_NAME=<your-workspace-name>

A browser window will open the first time you run this so you can sign in
with your Azure account (InteractiveBrowserCredential).
"""

import os
import sys

from dotenv import load_dotenv
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient

# Load environment variables from .env file in the repo root
load_dotenv()

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group  = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name  = os.getenv("AZURE_WORKSPACE_NAME")

if not all([subscription_id, resource_group, workspace_name]):
    print(
        "ERROR: One or more required environment variables are missing. "
        "Please set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, and "
        "AZURE_WORKSPACE_NAME in your .env file."
    )
    sys.exit(1)

# Browser-based interactive login (opens a browser window)
credential = InteractiveBrowserCredential()

# Create MLClient to connect to the Azure ML workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

workspace = ml_client.workspaces.get(workspace_name)
print(f"Connected to Azure ML Workspace: {workspace.name}")
print(f"Workspace Location:   {workspace.location}")
print(f"Resource Group:       {workspace.resource_group}")
