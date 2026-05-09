"""
azure_evaluate.py — Submit an MCUCoder evaluation job to Azure ML.

Run AFTER the training job has completed and the checkpoint is available.

Usage:
    python azureml/azure_evaluate.py

Outputs (downloadable from Azure ML Studio → job → Outputs tab):
    outputs/results/eval_summary.json
    outputs/results/rd_curves.pdf
    outputs/results/model_recon_k*.png
    outputs/results/jpeg_q*.png
"""

import os
from dotenv import load_dotenv
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.ai.ml.constants import AssetTypes

load_dotenv()

ml_client = MLClient(
    InteractiveBrowserCredential(),
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
    workspace_name=os.getenv("AZURE_WORKSPACE_NAME"),
)

conda_file = os.path.join(os.path.dirname(__file__), "conda.yml")

env = Environment(
    name="mcucoder-env",
    conda_file=conda_file,
    image="mcr.microsoft.com/azureml/curated/pytorch-2.2-cuda11.8:latest",
)

job = command(
    code="./",
    command=(
        "pip install compressai --no-deps && "
        "python main.py --mode evaluate"
    ),
    environment=env,
    compute="edheazml",
    inputs={
        "kodak": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:kodak-dataset:1",
        ),
        "checkpoint": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:mcucoder-checkpoint:1",  # registered after training
        ),
    },
    environment_variables={
        "AZUREML_VAL_DIR":        "${{inputs.kodak}}",
        "AZUREML_CHECKPOINT_DIR": "${{inputs.checkpoint}}",
    },
    display_name="mcucoder-evaluate",
    experiment_name="ELG5378-MCUCoder",
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Submitted:  {returned_job.name}")
print(f"Studio URL: {returned_job.studio_url}")