"""
upload_datasets.py — Upload local datasets to Azure ML as named data assets.

Run once before submitting any training job:
    python azureml/upload_datasets.py

Assets created (or new versions added):
    imagenet-jamieSJS   — ~13k training images  (datasets/imagenet/train/)
    kodak-dataset       — 24 validation images   (datasets/kodak/)
"""

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

for dataset in datasets:
    abs_path = os.path.abspath(dataset["path"])
    if not os.path.isdir(abs_path):
        print(f"WARNING: '{abs_path}' does not exist — skipping '{dataset['name']}'.")
        continue

    print(f"Uploading dataset '{dataset['name']}' from path '{abs_path}'...")
    data_asset = Data(
        name=dataset["name"],
        description=dataset["description"],
        path=abs_path,
        type=AssetTypes.URI_FOLDER,
    )
    created = ml_client.data.create_or_update(data_asset)
    print(f"  -> {created.name}:{created.version}")

print("All datasets processed.")
