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

──────────────────────────────────────────────────────────────────────────────
EDUCATIONAL OVERVIEW — Azure ML core concepts touched by this script
──────────────────────────────────────────────────────────────────────────────

  Subscription   : Your Azure billing container. Everything on Azure lives
                   inside one subscription.
  Resource Group : A logical folder inside a subscription that groups related
                   Azure resources (storage, compute, ML workspaces, etc.).
  Workspace      : The top-level Azure Machine Learning resource. It owns
                   datasets ("data assets"), trained models, environments,
                   compute clusters, and experiment / job history.
  MLClient       : The Python SDK v2 entry point. Every interaction with the
                   workspace (submitting a job, registering a model, listing
                   data assets) goes through an MLClient instance.
  Credential     : An object from `azure.identity` that knows HOW to obtain
                   an OAuth token for Azure. We use InteractiveBrowserCredential
                   here, which pops a browser window for sign-in. Other common
                   choices are DefaultAzureCredential (CLI / env / managed
                   identity fallback chain) and ManagedIdentityCredential
                   (used inside Azure-hosted compute).
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys

# ─── Third-party imports ──────────────────────────────────────────────────────
# `python-dotenv` reads KEY=VALUE pairs from a .env file into os.environ so we
# never hard-code subscription IDs into source control.
from dotenv import load_dotenv

# `azure.identity` provides credential classes. InteractiveBrowserCredential
# triggers the OAuth "device-code / browser" flow — perfect for laptops where
# the developer is signed in to Azure interactively.
from azure.identity import InteractiveBrowserCredential

# `azure.ai.ml.MLClient` is the SDK v2 façade for the Azure ML workspace.
# Think of it as the "kubectl" of Azure ML — every workspace operation
# (jobs.create_or_update, data.create_or_update, models.get, …) hangs off it.
from azure.ai.ml import MLClient


# ─── Step 1: Load .env into the process environment ──────────────────────────
# After this call, os.getenv("AZURE_SUBSCRIPTION_ID") returns the value from
# the .env file, but ONLY if the variable was not already set in the shell.
# Real shell environment variables always win — this is good practice for CI.
load_dotenv()

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")  # 'xxxxxxxx-xxxx-…'
resource_group  = os.getenv("AZURE_RESOURCE_GROUP")   # e.g. 'rg-mlcourse'
workspace_name  = os.getenv("AZURE_WORKSPACE_NAME")   # e.g. 'mlw-elg5378'

# Guard rail: fail fast with a friendly message if the user forgot the .env.
if not all([subscription_id, resource_group, workspace_name]):
    print(
        "ERROR: One or more required environment variables are missing. "
        "Please set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, and "
        "AZURE_WORKSPACE_NAME in your .env file."
    )
    sys.exit(1)


# ─── Step 2: Acquire an Azure credential ─────────────────────────────────────
# InteractiveBrowserCredential() does NOT contact Azure yet — it's lazy. The
# OAuth flow runs the first time the credential is asked for a token, which
# happens implicitly the first time the MLClient calls the Azure ML REST API.
# After login, the token is cached on disk under `~/.azure/`, so subsequent
# script runs reuse it silently until it expires.
credential = InteractiveBrowserCredential()


# ─── Step 3: Build an MLClient bound to the target workspace ─────────────────
# MLClient identifies the workspace by the (subscription, resource_group,
# workspace_name) triple. The credential is what authenticates each REST call.
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)


# ─── Step 4: Round-trip a real API call to prove the connection works ────────
# `ml_client.workspaces.get(name)` issues an authenticated GET to the Azure
# ML control plane. If the credential, subscription, RG, or workspace name
# is wrong, this is where the failure surfaces with a clear error.
workspace = ml_client.workspaces.get(workspace_name)

print(f"Connected to Azure ML Workspace: {workspace.name}")
print(f"Workspace Location:   {workspace.location}")          # e.g. 'eastus2'
print(f"Resource Group:       {workspace.resource_group}")
