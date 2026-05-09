#!/bin/zsh
set -e

echo "Submitting Azure ML training job..."
python azureml/azure_train.py

echo "Done. Check the Studio URL above for job progress."