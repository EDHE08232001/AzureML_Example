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

──────────────────────────────────────────────────────────────────────────────
EDUCATIONAL OVERVIEW — chaining jobs through registered assets
──────────────────────────────────────────────────────────────────────────────

This script demonstrates the canonical Azure ML pattern for multi-stage ML
pipelines that share artifacts:

      ┌────────────┐  outputs/checkpoints/   ┌─────────────────────────┐
      │  TRAIN job │ ──────────────────────► │ register_checkpoint.py  │
      └────────────┘     (auto-uploaded)     │  → Data asset:          │
                                             │     mcucoder-checkpoint │
                                             └────────────┬────────────┘
                                                          │ azureml:…@latest
                                                          ▼
                                                 ┌────────────────┐
                                                 │ EVALUATE job   │
                                                 │ (this script)  │
                                                 └────────────────┘

Why register the checkpoint as a Data asset instead of hard-coding the
training job's output URI?
  • Decoupling   — eval jobs don't break when you re-run training.
  • Versioning   — `@latest` always pulls the newest model; old versions
                   stay around for comparison or rollback.
  • Lineage      — Studio shows which training job produced which
                   checkpoint version.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys

from dotenv import load_dotenv
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.ai.ml.constants import AssetTypes


# ─── Authenticate + connect (same pattern as the training script) ───────────
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


# ─── Reuse the same Environment definition as training ──────────────────────
# Re-declaring the Environment with the SAME `name` and the SAME conda spec
# means Azure ML will reuse the cached Docker image built for the training
# job — eval starts in seconds rather than minutes.
conda_file = os.path.join(os.path.dirname(__file__), "conda.yml")
env = Environment(
    name="mcucoder-env",
    description="PyTorch + compressai + MLflow environment for MCUCoder.",
    conda_file=conda_file,
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
)


# ─── Build the evaluation command job ───────────────────────────────────────
job = command(

    # Same code snapshot as training; we just run a different `--mode`.
    code="./",
    command="python -u main.py --mode evaluate",

    environment=env,
    compute="edheazml",

    inputs={
        # Validation set is just the registered Kodak asset (version 1).
        "kodak": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:kodak-dataset:1",
        ),

        # ── Key concept: `@latest` label ──────────────────────────────────
        # `mcucoder-checkpoint@latest` resolves at submission time to
        # whatever version is currently the highest. So:
        #
        #    1. Run azure_train.py            → job name X
        #    2. Run register_checkpoint.py X  → mcucoder-checkpoint:1
        #    3. Run azure_evaluate.py         → consumes :1
        #    4. Re-train + register            → mcucoder-checkpoint:2
        #    5. Re-run azure_evaluate.py      → automatically picks :2
        #
        # No need to edit this file between training cycles.
        "checkpoint": Input(
            type=AssetTypes.URI_FOLDER,
            path="azureml:mcucoder-checkpoint@latest",
        ),
    },

    # The eval script's `src/config.py` reads these env vars to locate the
    # mounted Kodak directory and the directory containing mcucoder.pth.
    environment_variables={
        "AZUREML_VAL_DIR":        "${{inputs.kodak}}",
        "AZUREML_CHECKPOINT_DIR": "${{inputs.checkpoint}}",
        "PYTHONUNBUFFERED":       "1",
    },

    # Reuse the same `experiment_name` so train + eval runs sit side-by-side
    # in the same experiment view in Studio.
    display_name="mcucoder-evaluate",
    experiment_name="ELG5378-MCUCoder",
    description="MCUCoder rate-distortion evaluation vs JPEG baseline.",
)


# ─── Submit ─────────────────────────────────────────────────────────────────
returned_job = ml_client.jobs.create_or_update(job)
print(f"Submitted:   {returned_job.name}")
print(f"Studio URL:  {returned_job.studio_url}")
