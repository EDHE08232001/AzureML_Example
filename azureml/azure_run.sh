#!/bin/zsh
set -e

echo "Submitting Azure ML training job..."
python azureml/azure_train.py

echo ""
echo "Next steps after training completes:"
echo "  1. python azureml/register_checkpoint.py --job-name <job_name>"
echo "     Remember to replace <job_name> with the ACTUAL name of your Azure ML training job"
echo "  2. python azureml/azure_evaluate.py"