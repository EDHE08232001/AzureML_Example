"""
register_checkpoint.py — Register the trained checkpoint as an Azure ML data asset.

Run AFTER the training job completes, pointing at the job's output.

Usage:
    python azureml/register_checkpoint.py --job-name <job_name>

The job name is printed when you submit azure_train.py, e.g.:
    Submitted: brave_lamp_abc123

Each invocation creates a new version of the `mcucoder-checkpoint` asset, so
you can re-train and re-register safely. Use `mcucoder-checkpoint@latest` in
azure_evaluate.py to consume the most recent version.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
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

parser = argparse.ArgumentParser()
parser.add_argument("--job-name", required=True, help="Name of the completed training job")
args = parser.parse_args()

# Build the path to the job's output directory in Azure ML.
# The training script writes to ./outputs/checkpoints/, which Azure ML
# auto-uploads under the "outputs/" path of the job's artifact store.
checkpoint_path = (
    f"azureml://jobs/{args.job_name}/outputs/artifacts/paths/outputs/checkpoints/"
)

data_asset = Data(
    name="mcucoder-checkpoint",
    description=f"Trained MCUCoder checkpoint from job: {args.job_name}",
    path=checkpoint_path,
    type=AssetTypes.URI_FOLDER,
)
created = ml_client.data.create_or_update(data_asset)
print(f"Registered checkpoint from job: {args.job_name}")
print(f"Asset name:    {created.name}:{created.version}")
print(f"Asset latest:  mcucoder-checkpoint@latest")
