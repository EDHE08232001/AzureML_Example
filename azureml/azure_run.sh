#!/bin/zsh
# Convenience helper: submit the training job and print the next-step commands.
# Run from the repo root:  ./azureml/azure_run.sh
set -euo pipefail

echo "Submitting Azure ML training job..."
python azureml/azure_train.py

cat <<'EOF'

Next steps after training completes:
  1. Find the job name in the output above (or in Azure ML Studio).
  2. Register the checkpoint as a data asset:
       python azureml/register_checkpoint.py --job-name <job_name>
  3. Submit the evaluation job (uses mcucoder-checkpoint@latest):
       python azureml/azure_evaluate.py

Live logs and metrics are visible at the printed Studio URL.
EOF
