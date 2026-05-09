"""
azure_train.py — Submit an MCUCoder training job to Azure ML.

Usage:
    python azureml/azure_train.py

The job mounts the two pre-uploaded data assets and runs:
    python main.py --mode train

Logging
-------
Inside an Azure ML job, MLflow's tracking URI is set automatically to the
workspace's MLflow server. The training script (`src/train.py`) logs:
  - hyperparameters         → Job → Overview → Parameters
  - per-epoch metrics       → Job → Metrics
  - the best checkpoint     → Job → Outputs + logs (under "checkpoints/")

The run also writes the best checkpoint to ./outputs/checkpoints/mcucoder.pth
so Azure ML's automatic ./outputs/ uploader picks it up regardless of MLflow.
"""

import os
import sys

from dotenv import load_dotenv
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.ai.ml.constants import AssetTypes


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

ml_client = MLClient(
    InteractiveBrowserCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

conda_file = os.path.join(os.path.dirname(__file__), "conda.yml")

env = Environment(
    name="mcucoder-env",
    description="PyTorch + compressai + MLflow environment for MCUCoder.",
    conda_file=conda_file,
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
)

compute_cluster = "edheazml"

# Define the training job.
job = command(
    code="./",
    command=(
        # `python -u` makes stdout/stderr unbuffered so logs stream in real time
        # to the Azure ML Studio "Outputs + logs" tab.
        "python -u main.py --mode train"
    ),
    environment=env,
    compute=compute_cluster,
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
    environment_variables={
        # Point src/config.py paths at the mounted Azure inputs.
        "AZUREML_TRAIN_DIR": "${{inputs.imagenet}}",
        "AZUREML_VAL_DIR":   "${{inputs.kodak}}",
        # Make Python output unbuffered for live log streaming.
        "PYTHONUNBUFFERED":  "1",
    },
    display_name="mcucoder-train",
    experiment_name="ELG5378-MCUCoder",
    description="MCUCoder progressive image-compression training run.",
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Submitted:   {returned_job.name}")
print(f"Studio URL:  {returned_job.studio_url}")
print()
print("Next: once the job completes, register the checkpoint with:")
print(f"  python azureml/register_checkpoint.py --job-name {returned_job.name}")
