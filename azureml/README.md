# Azure ML Tutorial — Learn by Running MCUCoder

This document is both **a tutorial on Azure Machine Learning (SDK v2)** and the
**operating manual** for this repository. By the end of it you will:

- understand the core Azure ML object model (workspace, compute, environment, data asset, command job),
- know how to authenticate with `InteractiveBrowserCredential`,
- know how to submit and chain jobs that share data through registered assets,
- understand how MLflow plugs into Azure ML for tracking parameters, metrics, and artifacts,
- have a working end-to-end example you can copy and adapt to your own ML projects.

The example workload is **MCUCoder**, a progressive learned image-compression
model trained on an ImageNet subset and evaluated on the Kodak dataset. The ML
specifics don't matter for the tutorial — focus on the Azure ML scaffolding.

---

## Table of Contents

1. [Mental Model — what is Azure ML?](#1-mental-model)
2. [The Azure ML Object Hierarchy](#2-the-azure-ml-object-hierarchy)
3. [Authentication and `MLClient`](#3-authentication-and-mlclient)
4. [Compute Clusters](#4-compute-clusters)
5. [Environments — Docker + Conda](#5-environments-docker-conda)
6. [Data Assets and Versioning](#6-data-assets-and-versioning)
7. [Command Jobs — the Universal Unit of Work](#7-command-jobs-the-universal-unit-of-work)
8. [Inputs, Outputs, and the `${{inputs.x}}` Placeholder](#8-inputs-outputs-and-the-inputsx-placeholder)
9. [The `./outputs/` Auto-Upload Convention](#9-the-outputs-auto-upload-convention)
10. [MLflow Integration](#10-mlflow-integration)
11. [Hands-On Walkthrough — running this repo end-to-end](#11-hands-on-walkthrough)
12. [Where Each Kind of Log Appears in Studio](#12-where-each-kind-of-log-appears-in-studio)
13. [Common Pitfalls and Debugging Tips](#13-common-pitfalls)
14. [Going Further](#14-going-further)
15. [Glossary](#15-glossary)

---

## 1. Mental Model

Azure Machine Learning is a managed platform for running, tracking, and
governing ML workloads on Azure. The mental shift coming from "I run
`python train.py` on my laptop" is this:

> Instead of running code on a particular machine, you describe **what** to
> run (a command), **where** to run it (a compute cluster), **inside what
> environment** (a Docker image + conda spec), and **with which data**
> (registered data assets). Azure ML provisions the machine, sets up the
> environment, mounts the data, executes the command, captures everything
> the run produced, and tears the machine down.

Everything in this tutorial follows from that single shift.

```
                    ┌──────────────────────────────────────────────┐
                    │           Azure ML Workspace                 │
                    │  (control plane + storage + experiment log)  │
                    └─────────────┬────────────────────────────────┘
                                  │
       ┌──────────────────────────┼──────────────────────────────┐
       ▼                          ▼                              ▼
 Compute clusters         Registered assets              Job history
 (auto-scaling VMs)       (data, models, envs)           (runs, metrics, logs)
```

---

## 2. The Azure ML Object Hierarchy

| Layer | Object | What it is |
|---|---|---|
| Azure | **Subscription** | Your billing container. |
| Azure | **Resource Group** | A folder that groups related Azure resources. |
| Azure ML | **Workspace** | The top-level Azure ML resource. Owns everything below. |
| Workspace | **Compute** | Named CPU/GPU clusters that run jobs. |
| Workspace | **Environment** | Versioned Docker image + conda spec. |
| Workspace | **Data asset** | Versioned, named pointer to data in blob storage. |
| Workspace | **Model** | Versioned, named pointer to a trained model artifact. |
| Workspace | **Job** (run) | One execution: command + env + compute + I/O. Stored forever. |
| Workspace | **Experiment** | A label that groups related jobs. |

A useful rule of thumb: **everything you create in a workspace is named and
versioned**. Re-registering an asset with the same name produces version 2,
not an overwrite. This is what makes Azure ML reproducible.

---

## 3. Authentication and `MLClient`

Every interaction with the workspace goes through one Python object:

```python
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient

ml_client = MLClient(
    credential=InteractiveBrowserCredential(),
    subscription_id="<sub>",
    resource_group_name="<rg>",
    workspace_name="<ws>",
)
```

`MLClient` is a façade over a set of operation collections:

| Attribute | What it manages |
|---|---|
| `ml_client.workspaces` | Workspace metadata. |
| `ml_client.compute` | Compute clusters / instances. |
| `ml_client.environments` | Registered environments. |
| `ml_client.data` | Data assets. |
| `ml_client.models` | Registered models. |
| `ml_client.jobs` | Submit / inspect / cancel jobs. |

### Credentials, briefly

`azure.identity` provides several `*Credential` classes. They all expose the
same `get_token()` interface so you can swap them without changing call sites:

| Class | Use when |
|---|---|
| `InteractiveBrowserCredential` | Developer laptop. Pops a browser tab on first use; caches the token. **What this repo uses.** |
| `DefaultAzureCredential` | Generic fallback chain (env vars → managed identity → CLI → browser). Good in CI/CD. |
| `ManagedIdentityCredential` | Code running on an Azure VM / App Service / AKS pod with an attached identity. Zero secrets. |
| `ClientSecretCredential` | Service principal with a client secret. Useful for headless automation. |

For learning, `InteractiveBrowserCredential` is the easiest because it just
asks you to log in.

---

## 4. Compute Clusters

A **compute cluster** (`AmlCompute`) is a named pool of VMs with a fixed SKU
(e.g. `STANDARD_NC6s_v3` for one V100 GPU). It auto-scales between
`min_instances` and `max_instances`, charging only for active node-minutes.

You don't create the cluster from this script — you create it once in the
Studio UI (Compute → Compute clusters → New) or via the Azure CLI:

```bash
az ml compute create \
  --name edheazml --type amlcompute \
  --size STANDARD_NC6s_v3 --min-instances 0 --max-instances 1
```

Then in `azure_train.py` we just pass the name:

```python
job = command(..., compute="edheazml", ...)
```

When the job is submitted, Azure ML places it in the cluster's queue. The
cluster scales up, the job runs, the cluster scales back down to 0 after an
idle timeout (5 min default) so you stop paying. Cold-start adds 1–3 min to
the first run.

---

## 5. Environments — Docker + Conda

An **Environment** describes the container the job runs in. It's a tuple of:

- a base Docker image (gives you OS + CUDA + drivers + Azure ML runtime), and
- a conda spec (overlays the Python packages you need on top of the image).

In this repo, defined in [`azure_train.py`](./azure_train.py):

```python
env = Environment(
    name="mcucoder-env",
    conda_file="azureml/conda.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
)
```

### How the build cache works

The first job that uses a given (image, conda content) pair triggers an image
build (~5 min). Azure ML hashes the inputs and caches the resulting image. As
long as nothing in `conda.yml` changes, every subsequent job pulls the cached
image and starts in seconds.

### Curated images

`mcr.microsoft.com/azureml/...` images are Microsoft-curated. They already
contain CUDA, OpenMPI, and the Azure ML runtime hooks (which is what makes
the auto-upload of `./outputs/` and the MLflow tracking-URI redirection
work). Use them as your base unless you have a strong reason to roll your
own.

---

## 6. Data Assets and Versioning

A **data asset** is a named, versioned pointer to data living in workspace
blob storage.

| `AssetTypes` value | What it represents |
|---|---|
| `URI_FOLDER` | A directory of files. (What this repo uses.) |
| `URI_FILE` | A single file. |
| `MLTABLE` | A schema-aware tabular dataset. |

### Registering a folder

In [`upload_datasets.py`](./upload_datasets.py):

```python
data_asset = Data(
    name="kodak-dataset",
    description="Kodak 24-image lossless PNG evaluation set",
    path="./datasets/kodak",          # local folder
    type=AssetTypes.URI_FOLDER,
)
created = ml_client.data.create_or_update(data_asset)
print(f"{created.name}:{created.version}")   # "kodak-dataset:1"
```

The SDK uploads the local directory to workspace blob storage and registers
a Data entity that points at the resulting `azureml://...` URI. From now on
any job can reference it by name.

### Referencing in a job

```python
Input(type=AssetTypes.URI_FOLDER, path="azureml:kodak-dataset:1")
Input(type=AssetTypes.URI_FOLDER, path="azureml:kodak-dataset@latest")
```

Pin a version (`:1`) for reproducibility, or use `@latest` to always grab the
newest one. **Re-registering** with the same name creates **version 2** — the
old version stays addressable forever.

This is exactly how this repo chains training output into evaluation input:

```
azure_train.py  ──►  best checkpoint at azureml://jobs/<run>/.../checkpoints/
                                      │
                                      ▼
register_checkpoint.py  ──►  Data asset:  mcucoder-checkpoint:1
                                                                │
                                                                ▼
azure_evaluate.py  ──►  Input(path="azureml:mcucoder-checkpoint@latest")
```

---

## 7. Command Jobs — the Universal Unit of Work

A **command job** packages four things and submits them to a compute cluster:

```python
job = command(
    code="./",                                # 1. CODE   (uploaded snapshot)
    command="python -u main.py --mode train", # 2. COMMAND (what to run)
    environment=env,                          # 3. ENV    (image + conda)
    compute="edheazml",                       # 4. COMPUTE (cluster name)
    inputs={"kodak": Input(...)},             # typed I/O bindings
    environment_variables={...},
    display_name="mcucoder-train",
    experiment_name="ELG5378-MCUCoder",
)
returned = ml_client.jobs.create_or_update(job)
print(returned.name, returned.studio_url)
```

What happens on submission:

1. The SDK zips the `code` directory (respecting `.gitignore` / `.amlignore`)
   and uploads it as the job's **code snapshot**. The snapshot is **frozen**
   at this moment — editing local files afterwards has no effect on the run.
2. The SDK validates the spec and POSTs it to the workspace control plane.
3. The control plane creates a Job entity, queues it on the named compute.
4. The compute pulls/builds the environment image, mounts the inputs, sets
   the env vars, runs `command`, and uploads outputs + logs back.
5. `returned.name` is the run ID; `returned.studio_url` opens it in Studio.

### The `python -u` detail

`python -u` (and `PYTHONUNBUFFERED=1`) disable stdout/stderr buffering so log
lines stream to the Studio "Outputs + logs" tab in real time. Without this,
`tqdm` progress bars and `print` statements arrive in 4 KB chunks — confusing
when you're trying to debug a hang.

---

## 8. Inputs, Outputs, and the `${{inputs.x}}` Placeholder

`Input(...)` declares a typed binding to a data asset. At runtime Azure ML:

- mounts (or downloads) the data into the container at a path it chooses, and
- substitutes `${{inputs.<key>}}` in your `command` and `environment_variables`
  with that path.

This decouples your training code from any specific cloud path. In this repo:

```python
inputs={
    "imagenet": Input(type=URI_FOLDER, path="azureml:imagenet-jamieSJS:1"),
    "kodak":    Input(type=URI_FOLDER, path="azureml:kodak-dataset:1"),
},
environment_variables={
    "AZUREML_TRAIN_DIR": "${{inputs.imagenet}}",   # → "/mnt/azureml/cr/.../imagenet"
    "AZUREML_VAL_DIR":   "${{inputs.kodak}}",      # → "/mnt/azureml/cr/.../kodak"
},
```

`src/config.py` then reads those env vars to pick dataset paths. The training
code doesn't know it's running on Azure ML — it just sees normal directories.

### Outputs

You can declare typed `outputs={...}` similarly, but the simpler convention
this repo uses is **the auto-upload of `./outputs/`** — see the next section.

---

## 9. The `./outputs/` Auto-Upload Convention

Anything written to `./outputs/` (relative to the code snapshot, which is the
container's working directory) is **automatically uploaded** to the job's
artifact store when the job ends. You see it under
**Job → Outputs + logs → `outputs/`** in Studio.

`src/train.py` writes the best checkpoint to `./outputs/checkpoints/mcucoder.pth`,
so it lands at:

```
azureml://jobs/<run_name>/outputs/artifacts/paths/outputs/checkpoints/mcucoder.pth
```

That URI is what `register_checkpoint.py` packages into a `mcucoder-checkpoint`
Data asset.

---

## 10. MLflow Integration

Azure ML embeds an MLflow tracking server in every workspace. When a job runs
in Azure ML, the `azureml-mlflow` plugin (declared in `conda.yml`) does two
things automatically:

1. Sets `mlflow.set_tracking_uri()` to the workspace's MLflow server.
2. Activates an MLflow run that mirrors the Azure ML job.

That means **the same `mlflow.log_metrics(...)` call you'd write locally
streams metrics into Azure ML Studio's Metrics tab without any Azure-specific
code in your training script**.

In this repo (`src/train.py`):

```python
import mlflow

active = mlflow.active_run()
ctx = mlflow.start_run(nested=True) if active else mlflow.start_run()

with ctx:
    mlflow.log_params({"lr": 1e-4, "epochs": 10, ...})
    for epoch in range(num_epochs):
        ...
        mlflow.log_metrics({"train_loss": ..., "val_psnr_12ch": ...}, step=epoch)
    mlflow.log_artifact("outputs/checkpoints/mcucoder.pth", "checkpoints")
```

This works **identically locally and on Azure ML**. Locally MLflow falls back
to writing into a `./mlruns/` directory.

### What gets logged

| MLflow call | What it captures | Where it shows in Studio |
|---|---|---|
| `log_params({...})` | Hyperparameters (strings) | Job → **Overview** → Parameters |
| `log_metrics({...}, step=k)` | Numeric time series | Job → **Metrics** (live charts) |
| `log_artifact(path, ap=...)` | Files | Job → **Outputs + logs** → `mlflow-artifacts/` |

---

## 11. Hands-On Walkthrough

### 0. Prerequisites

1. An Azure ML workspace.
2. A compute cluster named **`edheazml`** in that workspace
   (or edit the `compute=` field in `azure_train.py` and `azure_evaluate.py`).
3. The local datasets at:
   - `datasets/imagenet/train/`  (ImageNet subset — see repo root README)
   - `datasets/kodak/`           (24 Kodak PNGs)
4. A `.env` file in the **repo root** (copy `.env.example`):
   ```
   AZURE_SUBSCRIPTION_ID=...
   AZURE_RESOURCE_GROUP=...
   AZURE_WORKSPACE_NAME=...
   ```
5. Submitter-side Python deps:
   ```bash
   pip install -r azureml/req_azureml.txt
   ```

### Step 1 — Verify the connection

```bash
python azureml/azure_connect.py
```

A browser window opens for sign-in (only on first run; the token caches under
`~/.azure/`). On success the script prints the workspace name, location, and
resource group. If this doesn't work, nothing else will — fix it first.

> **What you just exercised:** `InteractiveBrowserCredential` token flow,
> `MLClient` construction, and a real `workspaces.get()` round-trip.

### Step 2 — Upload the datasets (one-time)

```bash
python azureml/upload_datasets.py
```

This uploads `datasets/imagenet/train` and `datasets/kodak` to workspace
blob storage and registers two **`URI_FOLDER` data assets** named
`imagenet-jamieSJS` and `kodak-dataset`. Re-running creates **version 2**, not
an overwrite.

> **What you just exercised:** `Data` entity, `AssetTypes.URI_FOLDER`,
> `ml_client.data.create_or_update`.

### Step 3 — Submit the training job

```bash
python azureml/azure_train.py
```

The script prints the **job name** (e.g. `wise_ant_5x9q1tn3kv`) and the
**Studio URL**. Open the URL — you'll see live logs, a parameters panel
populated by `mlflow.log_params`, and metric charts that update each epoch.

What's happening behind the scenes:

1. The repo root is zipped and uploaded as the code snapshot.
2. Azure ML builds (or reuses) the `mcucoder-env` Docker image from `conda.yml`.
3. The cluster `edheazml` provisions a node, pulls the image, mounts both
   data assets, sets `AZUREML_TRAIN_DIR` / `AZUREML_VAL_DIR`.
4. `python -u main.py --mode train` runs. `src/train.py` calls
   `mlflow.start_run()` (which reuses Azure ML's auto-active run), logs
   params + per-epoch metrics, and saves the best checkpoint to
   `./outputs/checkpoints/mcucoder.pth`.
5. Azure ML uploads `./outputs/` to the job's artifact store on completion.

> **What you just exercised:** `command()` job spec, `Environment`, `Input`
> placeholders, `${{inputs.x}}` resolution, the `./outputs/` auto-upload,
> and end-to-end MLflow logging.

### Step 4 — Register the trained checkpoint

Copy the job name from step 3, then:

```bash
python azureml/register_checkpoint.py --job-name <job_name>
```

This wraps the job's `outputs/checkpoints/` URI in a new version of the
`mcucoder-checkpoint` data asset.

> **What you just exercised:** the `azureml://jobs/<name>/outputs/artifacts/paths/...`
> URI scheme and the train→register→evaluate pattern.

### Step 5 — Submit the evaluation job

```bash
python azureml/azure_evaluate.py
```

This consumes `mcucoder-checkpoint@latest`, so you don't need to edit the
file between training cycles. Outputs (PDF rate-distortion plot, JSON
summary, sample reconstructions) land both in `./outputs/results/` and as
MLflow artifacts under `results/` and `results/samples/`.

> **What you just exercised:** the `@latest` label, environment image cache
> reuse (Step 5 starts much faster than Step 3 because the image was already
> built), and downstream consumption of a registered model artifact.

---

## 12. Where Each Kind of Log Appears in Studio

| Source            | Where it appears in Azure ML Studio                                  |
|-------------------|----------------------------------------------------------------------|
| `mlflow.log_params`   | Job → **Overview** → Parameters table                            |
| `mlflow.log_metrics`  | Job → **Metrics** (live charts, with `step` as the X-axis)       |
| `mlflow.log_artifact` | Job → **Outputs + logs** → `mlflow-artifacts/`                   |
| `print()` / `logger.info()` | Job → **Outputs + logs** → `user_logs/std_log.txt`         |
| `./outputs/...` files | Job → **Outputs + logs** → `outputs/` (auto-uploaded)            |

You can also pull runs programmatically:

```python
import mlflow
mlflow.set_tracking_uri(ml_client.workspaces.get(ws_name).mlflow_tracking_uri)
client = mlflow.tracking.MlflowClient()
run = client.get_run("<job_name>")
print(run.data.metrics)
```

---

## 13. Common Pitfalls

**"My job was queued for ages."**
The compute cluster scaled to 0 and is cold-starting. The first node takes
1–3 min. Set `min_instances=1` if you need warm latency (and don't mind
paying for an idle node).

**"Pip install fails inside the job."**
Add the package to `azureml/conda.yml`, not `requirements.txt`. The job
container's environment is built **only** from `conda.yml`. `requirements.txt`
is for local development.

**"My logs only appear after the job finishes."**
Buffering. Use `python -u` in the `command` and add `PYTHONUNBUFFERED=1` to
`environment_variables`. (This repo already does both.)

**"`azureml:foo:1` not found."**
The asset isn't registered yet (run `upload_datasets.py`) or you have a
version mismatch. Use `@latest` while iterating, pin the version in
production.

**"My checkpoint isn't in `outputs/`."**
You probably wrote it to an absolute path or to a directory other than
`./outputs/`. Only `./outputs/` (relative to the code snapshot) is
auto-uploaded.

**"My code edits aren't taking effect."**
The code snapshot is frozen at submission time. Re-submit to pick up changes.

**"InteractiveBrowserCredential is hanging in CI."**
You can't open a browser there. Switch to `DefaultAzureCredential` and
authenticate via env vars or a service principal.

---

## 14. Going Further

When you're ready to graduate from single command jobs:

- **Pipeline jobs** — `from azure.ai.ml.dsl import pipeline`. Wire multiple
  command jobs into a DAG that shares typed outputs/inputs. The natural next
  step from this train→register→evaluate chain.
- **Sweep jobs** — hyperparameter search (`from azure.ai.ml.sweep import ...`).
  Same `command()` spec wrapped with `.sweep(...)` over a search space.
- **Distributed training** — set `distribution=PyTorchDistribution(process_count_per_instance=N)`
  on the `command` and `instance_count=K` on the compute, plus `torchrun` in
  your `command` string.
- **Online endpoints** — deploy a registered model behind a managed REST endpoint.
- **Automated MLOps** — wire all of the above into GitHub Actions / Azure
  DevOps, swap `InteractiveBrowserCredential` for a service principal, and
  pin asset versions in production manifests.

The mental model from §1 carries forward to all of these: **describe what
you want, let Azure ML provision and orchestrate**.

---

## 15. Glossary

- **Asset** — Anything named and versioned in the workspace (data, model, environment, component).
- **Code snapshot** — The zipped copy of your `code=` directory, frozen at job-submit time.
- **Command job** — The simplest job type: one shell command in a container with declared I/O.
- **Compute target** / **compute cluster** — The Azure VMs that run jobs. Auto-scaling pool.
- **Credential** — An `azure.identity` object that knows how to obtain an OAuth token.
- **Data asset** — Versioned pointer to data in workspace blob storage. Types: `URI_FOLDER`, `URI_FILE`, `MLTABLE`.
- **Environment** — Versioned (Docker image + conda spec) pair used to build the job container.
- **Experiment** — A label that groups related jobs in Studio.
- **Job** / **run** — One execution. Stored forever along with its logs, metrics, and outputs.
- **`MLClient`** — The SDK v2 façade. Every workspace operation goes through it.
- **MLflow** — Open-source experiment-tracking library. Azure ML embeds an MLflow server per workspace.
- **`@latest`** — Asset label that resolves at submission time to the highest version of an asset.
- **Studio** — The Azure ML web UI at `https://ml.azure.com`.
- **URI_FOLDER** — A data asset that is a directory of files (the type used throughout this repo).
- **Workspace** — The top-level Azure ML resource. Owns compute, assets, environments, and job history.
