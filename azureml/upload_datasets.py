"""
upload_datasets.py — Upload local datasets to Azure ML as named data assets.

Run once before submitting any training job:
    python azureml/upload_datasets.py

Assets created (or new versions added):
    imagenet-jamieSJS   — ~13k training images  (datasets/imagenet/train/)
    kodak-dataset       — 24 validation images   (datasets/kodak/)

──────────────────────────────────────────────────────────────────────────────
EDUCATIONAL OVERVIEW — what is an Azure ML "data asset"?
──────────────────────────────────────────────────────────────────────────────

A *data asset* is a named, versioned pointer to data that lives inside the
workspace's blob storage. Once registered, you reference it from any job by
name (e.g. `azureml:kodak-dataset:1` or `azureml:kodak-dataset@latest`)
instead of hard-coding storage paths. Benefits:

  • Reproducibility — every job records the EXACT data version it consumed.
  • Lineage         — the Studio UI shows which jobs produced / consumed
                      each version of an asset.
  • Mounting        — at job runtime Azure ML can mount the asset as a
                      read-only folder inside the container, so your code
                      just sees a normal filesystem path.

`AssetTypes`:
  • URI_FOLDER — a directory of files (what we use here).
  • URI_FILE   — a single file.
  • MLTABLE    — a tabular schema-aware dataset.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys

from dotenv import load_dotenv
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient

# `Data` is the SDK v2 entity class that represents a data asset. We populate
# its fields and hand the object to ml_client.data.create_or_update() to
# register / update the asset in the workspace.
from azure.ai.ml.entities import Data

# Enum of supported asset types. Using the enum (instead of the string
# "uri_folder") gives us auto-complete + type-checking and protects against
# typos that would only surface at submission time.
from azure.ai.ml.constants import AssetTypes


# ─── Authenticate and connect to the workspace ───────────────────────────────
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

# InteractiveBrowserCredential opens a browser tab on first use; subsequent
# runs reuse the cached token from ~/.azure until it expires.
ml_client = MLClient(
    InteractiveBrowserCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)


# ─── Describe the local datasets we want to register ─────────────────────────
# Each dict becomes one Data asset. The `name` is what jobs will reference,
# the `path` is the LOCAL directory the SDK will upload to workspace blob
# storage, and `description` shows up in the Studio UI.
datasets = [
    {
        "path":        "./datasets/imagenet/train",
        "name":        "imagenet-jamieSJS",
        "description": "JamieSJS ImageNet-10 subset, 13k images, test split",
    },
    {
        "path":        "./datasets/kodak",
        "name":        "kodak-dataset",
        "description": "Kodak 24-image lossless PNG evaluation set",
    },
]


# ─── Upload + register each dataset ──────────────────────────────────────────
for dataset in datasets:
    abs_path = os.path.abspath(dataset["path"])

    # Defensive check — better to skip with a clear warning than to make
    # `create_or_update` raise a confusing internal error later.
    if not os.path.isdir(abs_path):
        print(f"WARNING: '{abs_path}' does not exist — skipping '{dataset['name']}'.")
        continue

    print(f"Uploading dataset '{dataset['name']}' from path '{abs_path}'...")

    # Build the Data entity in memory.
    #   - `type=URI_FOLDER`  →  "this asset is a directory of files"
    #   - `path=<local>`     →  the SDK uploads this directory to workspace
    #                            blob storage on submit. After upload, `path`
    #                            on the registered asset becomes an
    #                            `azureml://...` URI, NOT this local path.
    #   - `name`             →  shared logical name across versions.
    #   - `version`          →  optional; if omitted Azure ML auto-increments,
    #                            so re-running this script bumps the version
    #                            number rather than overwriting.
    data_asset = Data(
        name=dataset["name"],
        description=dataset["description"],
        path=abs_path,
        type=AssetTypes.URI_FOLDER,
    )

    # `create_or_update` is the SDK v2 idiom for "upsert". Returns the
    # registered asset, including the version that was assigned.
    created = ml_client.data.create_or_update(data_asset)

    # Print the assigned version so the user can wire it into azure_train.py
    # (e.g. "azureml:kodak-dataset:3").
    print(f"  -> {created.name}:{created.version}")

print("All datasets processed.")
