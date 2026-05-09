"""
azure_evaluate.py — Submit an MCUCoder evaluation job to Azure ML.

Run AFTER the training job has completed and the checkpoint is registered.

Usage:
    python azureml/azure_evaluate.py

Outputs (downloadable from Azure ML Studio → job → Outputs + logs):
    outputs/results/eval_summary.json
    outputs/results/rd_curves.pdf
    outputs/results/model_recon_k*.png
    outputs/results/jpeg_q*.png

The same files are also logged as MLflow artifacts under "results/" and
"results/samples/".
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

job = command(
    code="./",
    command="python -u main.py --mode evaluate",
    environment=env,
    compute="edheazml",
    inputs={
        "kodak": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:kodak-dataset:1",
        ),
        "checkpoint": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:mcucoder-checkpoint@latest",  # latest registered checkpoint
        ),
    },
    environment_variables={
        "AZUREML_VAL_DIR":        "${{inputs.kodak}}",
        "AZUREML_CHECKPOINT_DIR": "${{inputs.checkpoint}}",
        "PYTHONUNBUFFERED":       "1",
    },
    display_name="mcucoder-evaluate",
    experiment_name="ELG5378-MCUCoder",
    description="MCUCoder rate-distortion evaluation vs JPEG baseline.",
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Submitted:   {returned_job.name}")
print(f"Studio URL:  {returned_job.studio_url}")
