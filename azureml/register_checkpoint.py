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

──────────────────────────────────────────────────────────────────────────────
EDUCATIONAL OVERVIEW — promoting job outputs to first-class Data assets
──────────────────────────────────────────────────────────────────────────────

When an Azure ML job finishes, its outputs (everything written to ./outputs/
inside the container) live at a JOB-scoped URI:

    azureml://jobs/<job_name>/outputs/artifacts/paths/outputs/...

These URIs work, but they're awkward:
  • opaque — they refer to a specific job that future readers may not know.
  • brittle — deleting an old job breaks every downstream reference.
  • unversioned — you can't say "give me the third checkpoint we trained".

The fix is to *register* the checkpoint as its own named, versioned Data
asset. Behind the scenes Azure ML just records a pointer; nothing is copied.
But now:

  azureml:mcucoder-checkpoint:1            ← this exact run's output
  azureml:mcucoder-checkpoint:2            ← the next run's output
  azureml:mcucoder-checkpoint@latest       ← always the newest
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes


# ─── Authenticate + connect (same pattern as every other azureml/ script) ──
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


# ─── Take the training job name as a CLI argument ──────────────────────────
# We pass it on the command line (rather than reading from env / file) so the
# script is stateless and obvious to copy/paste from Studio or a CI pipeline.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--job-name",
    required=True,
    help="Name of the completed training job (printed by azure_train.py)",
)
args = parser.parse_args()


# ─── Build the path to the job's output directory ──────────────────────────
# The training script writes the best checkpoint to ./outputs/checkpoints/
# inside the container. Azure ML's auto-uploader copies the entire ./outputs/
# directory to the job's artifact store under this canonical URI scheme:
#
#   azureml://jobs/<job_name>/outputs/artifacts/paths/<relative_path>/
#
# We point the data asset at the "checkpoints" subdirectory specifically.
checkpoint_path = (
    f"azureml://jobs/{args.job_name}/outputs/artifacts/paths/outputs/checkpoints/"
)


# ─── Construct + register the new Data asset version ───────────────────────
# Calling `create_or_update` with a name that already exists adds a NEW
# version rather than overwriting the previous one. So you can re-run this
# script after every training round without losing history.
data_asset = Data(
    name="mcucoder-checkpoint",                       # logical name
    description=f"Trained MCUCoder checkpoint from job: {args.job_name}",
    path=checkpoint_path,                             # azureml:// URI
    type=AssetTypes.URI_FOLDER,                       # it's a directory
)

created = ml_client.data.create_or_update(data_asset)


# ─── Report the assigned version so the user knows what to consume ─────────
print(f"Registered checkpoint from job: {args.job_name}")
print(f"Asset name:    {created.name}:{created.version}")
print(f"Asset latest:  mcucoder-checkpoint@latest")
