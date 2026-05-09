# Azure ML Workflow

This folder contains everything needed to run **MCUCoder** on Azure Machine
Learning end-to-end — dataset upload, training, checkpoint registration, and
evaluation — with full **MLflow logging** of metrics, parameters, and artifacts.

All scripts authenticate using `azure.identity.InteractiveBrowserCredential`
(a browser window opens the first time you run them; subsequent runs reuse the
cached token).

---

## 0. Prerequisites

1. An Azure ML workspace and a compute cluster named **`edheazml`**
   (or edit the `compute=` field in `azure_train.py` / `azure_evaluate.py`).
2. The local datasets at:
   - `datasets/imagenet/train/`  (ImageNet subset — see repo root README)
   - `datasets/kodak/`           (24 Kodak PNGs)
3. A `.env` file in the **repo root** (copy `.env.example`) containing:
   ```
   AZURE_SUBSCRIPTION_ID=...
   AZURE_RESOURCE_GROUP=...
   AZURE_WORKSPACE_NAME=...
   ```
4. Python deps for the **submitter** machine:
   ```bash
   pip install -r azureml/req_azureml.txt
   ```

---

## 1. Verify the connection

```bash
python azureml/azure_connect.py
```

A browser window opens for sign-in. On success it prints the workspace name,
location, and resource group.

## 2. Upload the datasets (one-time)

```bash
python azureml/upload_datasets.py
```

Creates / versions the `imagenet-jamieSJS` and `kodak-dataset` URI-folder assets.

## 3. Submit the training job

```bash
python azureml/azure_train.py
```

The output prints the **job name** and the **Studio URL**. Open the URL to
watch live logs and metrics. Inside the running job:

- Hyperparameters are logged via **`mlflow.log_params`** → visible in the job's
  **Overview → Parameters** panel.
- Per-epoch `train_loss`, `train_psnr`, `lr`, and validation metrics
  (`val_loss`, `val_psnr_{2,6,12}ch`, `val_msssim_{2,6,12}ch`,
  `best_val_loss`) are logged via **`mlflow.log_metrics`** → visible in the
  job's **Metrics** tab as live charts.
- The best checkpoint is logged as an MLflow artifact (`checkpoints/`) **and**
  written to `./outputs/checkpoints/mcucoder.pth`, which Azure ML auto-uploads
  to the job's **Outputs + logs** tab.

## 4. Register the trained checkpoint

Once training succeeds, copy the job name from step 3 and run:

```bash
python azureml/register_checkpoint.py --job-name <job_name>
```

This creates a new version of the `mcucoder-checkpoint` data asset.

## 5. Submit the evaluation job

```bash
python azureml/azure_evaluate.py
```

This consumes `mcucoder-checkpoint@latest` so it always picks up the most
recent registration. The evaluation job logs:

- `model_bpp`, `model_psnr`, `model_msssim_db` per active-channel count
  (`step=k` from 1–12).
- `jpeg_bpp`, `jpeg_psnr`, `jpeg_msssim_db` per JPEG quality level
  (`step=quality`).
- Artifacts: `eval_summary.json`, `rd_curves.pdf`, and sample reconstruction
  PNGs (under `results/` and `results/samples/`).

---

## Where the logs live

| Source            | Where it appears in Azure ML Studio                                  |
|-------------------|----------------------------------------------------------------------|
| `mlflow.log_params`   | Job → **Overview** → Parameters table                            |
| `mlflow.log_metrics`  | Job → **Metrics** (live charts, with `step` as the X-axis)       |
| `mlflow.log_artifact` | Job → **Outputs + logs** → `mlflow-artifacts/` and `outputs/`    |
| `print()` / `logger.info()` | Job → **Outputs + logs** → `user_logs/std_log.txt`         |
| `./outputs/...` files | Job → **Outputs + logs** → `outputs/` (auto-uploaded)            |

You can also query runs programmatically via the MLflow client — get the
tracking URI from the workspace and pass it to `mlflow.set_tracking_uri()`.

---

## Convenience script

```bash
./azureml/azure_run.sh
```

Submits training and prints the next-step commands.
