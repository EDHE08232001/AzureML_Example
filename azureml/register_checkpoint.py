"""
register_checkpoint.py — Register the trained checkpoint as an Azure ML data asset.

Run AFTER the training job completes, pointing at the job's output.

Usage:
    python azureml/register_checkpoint.py --job-name <job_name>

The job name is printed when you submit azure_train.py, e.g.:
    Submitted: brave_lamp_abc123
"""

import argparse
import os
from dotenv import load_dotenv
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

load_dotenv()

ml_client = MLClient(
    InteractiveBrowserCredential(),
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
    workspace_name=os.getenv("AZURE_WORKSPACE_NAME"),
)

parser = argparse.ArgumentParser()
parser.add_argument("--job-name", required=True, help="Name of the completed training job")
args = parser.parse_args()

# Build the path to the job's output directory in Azure ML
checkpoint_path = f"azureml://jobs/{args.job_name}/outputs/artifacts/paths/outputs/checkpoints/"

data_asset = Data(
    name="mcucoder-checkpoint",
    description="Trained MCUCoder checkpoint from training job",
    path=checkpoint_path,
    type=AssetTypes.URI_FOLDER,
)
ml_client.data.create_or_update(data_asset)
print(f"Registered checkpoint from job: {args.job_name}")
print("Asset name: mcucoder-checkpoint:1")