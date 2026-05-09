"""
azure_train.py — Submit an MCUCoder training job to Azure ML.

Usage:
    python azureml/azure_train.py

The job mounts the two pre-uploaded data assets and runs:
    python main.py --mode train

──────────────────────────────────────────────────────────────────────────────
EDUCATIONAL OVERVIEW — anatomy of an Azure ML "command job"
──────────────────────────────────────────────────────────────────────────────

A *command job* is the most common Azure ML job type. It packages four things
together and runs them on a compute cluster:

  1. CODE          — a directory uploaded to the workspace as a snapshot.
                     This becomes the working directory inside the container.
                     Frozen at submit time, so two re-runs are reproducible
                     even if you keep editing locally afterwards.

  2. ENVIRONMENT   — a Docker image + a conda spec describing Python deps.
                     Azure ML builds (or reuses a cached) image and runs the
                     job inside it. Defining the env in code beats "works on
                     my laptop" by a wide margin.

  3. COMPUTE       — a named compute cluster (here `edheazml`). Clusters
                     auto-scale: 0 nodes when idle, spin up to N when jobs
                     are queued, scale back to 0 after an idle timeout.
                     You only pay for active node-minutes.

  4. INPUTS / OUTPUTS — typed bindings. Each Input points at a registered
                     data asset (URI_FOLDER / URI_FILE / MLTABLE) and gets
                     mounted (or downloaded) into the container at a path
                     of Azure ML's choosing. The script reads the path from
                     the env var defined under `environment_variables=`.

The submission flow:

    SDK call           →  Control plane API  →  Compute cluster
    ─────────────         ───────────────────    ─────────────────
    code zipped + sent    job validated +        node provisioned,
    inputs resolved       persisted as a Run     image pulled, code
    env spec uploaded                            unzipped, command run

Logging
-------
Inside an Azure ML job, MLflow's tracking URI is set automatically to the
workspace's MLflow server. The training script (`src/train.py`) logs:
  - hyperparameters         → Job → Overview → Parameters
  - per-epoch metrics       → Job → Metrics
  - the best checkpoint     → Job → Outputs + logs (under "checkpoints/")

The run also writes the best checkpoint to ./outputs/checkpoints/mcucoder.pth
so Azure ML's automatic ./outputs/ uploader picks it up regardless of MLflow.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys

from dotenv import load_dotenv
from azure.identity import InteractiveBrowserCredential

# `command` is a factory function that builds a command-job spec.
# `Input` represents one typed input binding to a data asset.
from azure.ai.ml import MLClient, command, Input

# `Environment` is the SDK entity describing the (image + conda) job runtime.
from azure.ai.ml.entities import Environment

# Enum used by Input(type=...) — see upload_datasets.py for the list.
from azure.ai.ml.constants import AssetTypes


# ─── Step 1: Authenticate + connect ──────────────────────────────────────────
load_dotenv()

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group  = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name  = os.getenv("AZURE_WORKSPACE_NAME")

if not all([subscription_id, resource_group, workspace_name]):
    print(
        "ERROR: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, and "
        "AZURE_WORKSPACE_NAME must all be set in your .env file."
    )
    sys.exit(1)

# Browser-based interactive sign-in. The token is cached on disk.
ml_client = MLClient(
    InteractiveBrowserCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)


# ─── Step 2: Define the runtime environment ──────────────────────────────────
# Path to the conda spec next to this file. The conda file lists ALL Python
# packages the job needs (PyTorch, compressai, mlflow, azureml-mlflow, …).
conda_file = os.path.join(os.path.dirname(__file__), "conda.yml")

# An Environment combines:
#   • a base Docker image (`image=...`) — gives us OS, CUDA, OpenMPI, etc.
#   • a conda specification (`conda_file=...`) — overlays our Python deps.
# Azure ML will build a derived image the first time you submit a job using
# this environment, then cache it. Bumping `name` or the conda file content
# triggers a rebuild; otherwise the cached image is reused for fast starts.
env = Environment(
    name="mcucoder-env",
    description="PyTorch + compressai + MLflow environment for MCUCoder.",
    conda_file=conda_file,
    # `mcr.microsoft.com/azureml/...` images are Microsoft-curated bases
    # that already contain CUDA drivers, OpenMPI, and the AzureML runtime.
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
)


# ─── Step 3: Pick the compute target ─────────────────────────────────────────
# This must match the *name* of an existing compute cluster in the workspace.
# Create one ahead of time in Studio (Compute → Compute clusters → New) or
# via `az ml compute create`. The cluster's VM SKU (CPU/GPU type) is fixed
# at creation; the script doesn't choose it.
compute_cluster = "edheazml"


# ─── Step 4: Build the command-job spec ──────────────────────────────────────
# Conceptually: "Run this shell command, in this environment, on this compute,
#               with these inputs mounted, and these env vars set."
job = command(

    # The directory uploaded as the job's code snapshot. "./" is the repo
    # root because we run this script from the repo root. Everything outside
    # the .gitignore/.amlignore patterns gets uploaded.
    code="./",

    # Shell command executed inside the container after the env is activated.
    # `python -u` disables stdout/stderr buffering so log lines stream to the
    # Studio "Outputs + logs" tab in real time instead of arriving in chunks.
    command="python -u main.py --mode train",

    # Wire in the environment + compute we built above.
    environment=env,
    compute=compute_cluster,

    # ── Inputs ────────────────────────────────────────────────────────────
    # Each key here becomes a placeholder (`${{inputs.<key>}}`) usable in
    # the `command` string and in `environment_variables`. At runtime Azure
    # ML resolves each Input to a path inside the container and substitutes
    # the placeholder with that path.
    #
    # `path="azureml:imagenet-jamieSJS:1"` references version 1 of the data
    # asset we registered with upload_datasets.py. Use `@latest` to always
    # take the newest version, or pin a specific number for reproducibility.
    inputs={
        "imagenet": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:imagenet-jamieSJS:1",
        ),
        "kodak": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:kodak-dataset:1",
        ),
    },

    # ── Environment variables ─────────────────────────────────────────────
    # `${{inputs.imagenet}}` is replaced by Azure ML with the in-container
    # path to the mounted ImageNet folder. Our `src/config.py` reads
    # AZUREML_TRAIN_DIR / AZUREML_VAL_DIR to redirect dataset paths so the
    # training code is unaware that it's running on Azure ML — it just sees
    # a normal directory of images.
    environment_variables={
        "AZUREML_TRAIN_DIR": "${{inputs.imagenet}}",
        "AZUREML_VAL_DIR":   "${{inputs.kodak}}",

        # Belt-and-suspenders unbuffered IO (in addition to `python -u`).
        "PYTHONUNBUFFERED":  "1",
    },

    # ── Identity ──────────────────────────────────────────────────────────
    # `display_name`     — the human-friendly name shown in Studio.
    # `experiment_name`  — groups related runs. All training + eval runs for
    #                      this project go into "ELG5378-MCUCoder", which
    #                      makes side-by-side metric comparison trivial.
    # `description`      — optional free-text shown on the job page.
    display_name="mcucoder-train",
    experiment_name="ELG5378-MCUCoder",
    description="MCUCoder progressive image-compression training run.",
)


# ─── Step 5: Submit the job to the workspace ─────────────────────────────────
# `jobs.create_or_update` POSTs the job spec to the Azure ML control plane.
# The returned object includes:
#   • `name`        — auto-generated GUID-ish ID (e.g. "wise_ant_5x9q1tn3kv").
#                     Use this with register_checkpoint.py later.
#   • `studio_url`  — clickable URL that opens the run in Azure ML Studio.
returned_job = ml_client.jobs.create_or_update(job)

print(f"Submitted:   {returned_job.name}")
print(f"Studio URL:  {returned_job.studio_url}")
print()
print("Next: once the job completes, register the checkpoint with:")
print(f"  python azureml/register_checkpoint.py --job-name {returned_job.name}")
