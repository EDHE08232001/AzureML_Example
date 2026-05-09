"""
azure_train.py — Submit an MCUCoder training job to Azure ML.

Usage:
    python azureml/azure_train.py

The job mounts the two pre-uploaded data assets and runs:
    python main.py --mode train
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
    image="mcr.microsoft.com/azureml/curated/pytorch-2.2-cuda11.8:latest"
)
compute_cluster = "edheazml"

# Define the training job
job = command(
    code="./",
    command="python main.py --mode train",
    environment=env,
    compute=compute_cluster,
    inputs={
        "imagenet": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:imagenet-jamieSJS:1"
        ),
        "kodak": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:kodak-dataset:1",
        )
    },
    environment_variables={
        # Point src/config.py paths at the mounted Azure inputs
        "AZUREML_TRAIN_DIR": "${{inputs.imagenet}}",
        "AZUREML_VAL_DIR":   "${{inputs.kodak}}",
    },
    display_name="mcucoder-train",
    experiment_name="ELG5378-MCUCoder"
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Submitted:  {returned_job.name}")
print(f"Studio URL: {returned_job.studio_url}")