"""
upload_datasets.py — Upload local datasets to Azure ML as named data assets.

Run once before submitting any training job:
    python azureml/upload_datasets.py

Assets created:
    imagenet-jamieSJS   — ~13k training images  (datasets/imagenet/train/)
    kodak-dataset       — 24 validation images   (datasets/kodak/)
"""

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
    print(f"Uploading dataset '{dataset['name']}' from path '{dataset['path']}'...")
    data_asset = Data(
        name=dataset["name"],
        description=dataset["description"],
        path=dataset["path"],
        type=AssetTypes.URI_FOLDER,
    )
    ml_client.data.create_or_update(data_asset)
    print(f"Uploaded dataset '{dataset['name']}' to Azure ML.")

print("All datasets uploaded successfully.")