#!/bin/zsh
# azure_run.sh — Convenience wrapper around the Azure ML training submission.
#
# ─────────────────────────────────────────────────────────────────────────────
# EDUCATIONAL OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
# This script is intentionally minimal. The Azure ML workflow has THREE stages
# that must run in order, and only the first one is fully automatable today
# because steps 2-3 depend on knowing the training job's name (printed at
# step 1's end). We submit step 1 here and remind the user about steps 2-3.
#
#     1. Submit training job          ── this script ──
#     2. Register trained checkpoint   ── manual: pass --job-name
#     3. Submit evaluation job         ── consumes mcucoder-checkpoint@latest
# ─────────────────────────────────────────────────────────────────────────────
#
# Run from the repo root:
#     ./azureml/azure_run.sh

# Strict shell mode:
#   -e   exit on first error
#   -u   error on unset variables
#   -o pipefail  exit if any command in a pipeline fails
set -euo pipefail

echo "Submitting Azure ML training job..."
python azureml/azure_train.py

# A heredoc keeps the multi-line message readable. The 'EOF' is single-quoted
# so $variables inside aren't expanded by the shell.
cat <<'EOF'

Next steps after training completes:
  1. Find the job name in the output above (or in Azure ML Studio).
  2. Register the checkpoint as a data asset:
       python azureml/register_checkpoint.py --job-name <job_name>
  3. Submit the evaluation job (uses mcucoder-checkpoint@latest):
       python azureml/azure_evaluate.py

Live logs and metrics are visible at the printed Studio URL.
EOF
