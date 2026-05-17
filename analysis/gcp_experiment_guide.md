# TinyYOLO GCP Experiment Execution Guide

This guide provides a comprehensive, production-grade, step-by-step walkthrough for deploying and running the entire `tinyYOLO` experimental pipeline on **Google Cloud Platform (GCP)**. 

By leveraging GCP's **Deep Learning VM Image** and **Spot (Preemptible) Instances**, you can train your models in a fully optimized environment for a fraction of the cost (~$0.11 to ~$0.35 per hour).

---

## Technical Architecture & Resource Planning

Before launching a VM, we plan the resources based on `tinyYOLO` requirements:
* **GPU**: A single **NVIDIA Tesla T4** (16 GB GDDR6) is highly cost-effective and perfectly matches your existing benchmarks. With standard FP16 mixed-precision enabled, batch size `128` fits comfortably at resolution `416`.
* **CPU / Memory**: `n1-standard-4` (4 vCPUs, 15 GB RAM). This provides sufficient CPU workers (`--workers 4`) to prevent the image-loading and data augmentation pipeline (including mosaic composition) from becoming a GPU bottleneck.
* **Storage**: **100 GB Balanced Persistent Disk (pd-balanced)**. This provides ample space for the operating system, dependencies, the Pascal VOC dataset (~2 GB), COCO dataset (~20 GB), and all checkpoint/metrics outputs.

---

## Step 1: GCP Account & GPU Quota Setup

Google Cloud limits GPU usage by default to prevent accidental billing. You **must** request a GPU quota increase before you can launch an instance.

### 1.1 Requesting GPU Quota
1. Log in to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a Project (e.g., `tinyyolo-experiments`). Note your **Project ID**.
3. Search for **"Quotas"** in the top search bar, or navigate to **IAM & Admin > Quotas**.
4. Click on **"Quotas & System Limits"** at the top.
5. In the **Filter** box, type `nvidia-t4` or `gpus` and hit Enter.
6. Look for **"Compute Engine API - GPUs (all regions)"** or **"NVIDIA T4 GPUs"** in your target region (e.g., `us-central1`).
7. Check the box next to it, click **Edit Quotas** (or **Request Increase**), set the limit to **1**, and submit.
   > [!IMPORTANT]
   > Approval is automated and typically takes between 2 to 10 minutes. You will receive a confirmation email once approved.

---

## Step 2: Provisioning the Deep Learning VM

The easiest, most robust way to run PyTorch experiments is to use GCP's pre-configured **Deep Learning VM Image**. This image comes pre-packaged with PyTorch, CUDA, and the required NVIDIA drivers, saving you hours of manual setup.

### Option A: Launching via GCP Command Line (`gcloud`)
If you have the [Google Cloud SDK](https://cloud.google.com/sdk) installed locally, run this single command to create your GPU instance. We leverage **Spot Instances** here, which offer up to a **90% discount** compared to standard VMs!

```bash
gcloud compute instances create tinyyolo-gpu-vm \
    --project="YOUR_PROJECT_ID" \
    --zone="us-central1-a" \
    --machine-type="n1-standard-4" \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --maintenance-policy="TERMINATE" \
    --image-family="pytorch-latest-gpu" \
    --image-project="deeplearning-platform-release" \
    --boot-disk-size="100GB" \
    --boot-disk-type="pd-balanced" \
    --metadata="install-nvidia-driver=true" \
    --provisioning-model="SPOT" \
    --instance-termination-action="STOP"
```

### Option B: Launching via the GCP Web Console
1. Navigate to **Compute Engine > VM Instances** and click **Create Instance**.
2. **Name**: `tinyyolo-gpu-vm`
3. **Region / Zone**: Choose a region near you that supports T4 GPUs (e.g., `us-central1-a`).
4. **Machine Configuration**:
   * Select **GPU** tab.
   * GPU type: **NVIDIA Tesla T4** (Qty: 1).
   * Machine type: **n1-standard-4** (4 vCPUs, 15 GB memory).
5. **VM Provisioning Model**: Expand the **Advanced Options > VM product details** section and select **Spot** (under provisioning model) for major cost savings.
6. **Boot Disk**:
   * Click **Change** under Boot Disk.
   * Under **Operating System**, select **Deep Learning on Linux**.
   * Under **Version**, choose the latest **PyTorch (with CUDA)** version.
   * Size: **100 GB** (Balanced Persistent Disk).
7. **Identity and API Access**: Ensure **Allow full API access to all Google Cloud APIs** is selected (useful for potential Cloud Storage integration).
8. Click **Create**.

---

## Step 3: Connecting to the Instance

Once the VM's status displays a green checkmark, connect to it using SSH.

```bash
# Connect using the Google Cloud CLI:
gcloud compute ssh tinyyolo-gpu-vm --zone="us-central1-a"
```

*Alternatively, click the blue **SSH** button next to the instance name in the GCP console.*

---

## Step 4: Environment Verification & Repository Cloning

Inside the VM terminal, verify that the GPU drivers and PyTorch are correctly mapped, then deploy your repository.

### 4.1 Verify GPU and CUDA Status
Run these commands to confirm that PyTorch is utilizing the Tesla T4 GPU:

```bash
# 1. Check hardware and drivers
nvidia-smi

# 2. Check PyTorch CUDA availability
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)}')"
```
*Expected Output:* `CUDA Available: True | GPU: Tesla T4`

### 4.2 Clone and Install the Project
```bash
# Clone the repository
git clone https://github.com/ShMazumder/tinyYOLO.git
cd tinyYOLO

# Install the package in editable mode with dependencies
pip install -e .

# Install experiment helper packages
pip install tqdm timm ultralytics
```

### 4.3 Validate the Installation
Execute a quick forward pass to ensure the architecture registry is functional:
```bash
python3 -c "
import torch
from tinyYOLO.models import build_model
model, info = build_model(task='det', variant='standard')
x = torch.randn(1, 3, 320, 320)
out = model(x)
print(f'✓ Setup Verified! Model has {info[\"total_params_M\"]}M parameters.')
"
```

---

## Step 5: Handling Disconnections with tmux (Crucial!)

Training all seeds and ablations will take **12 to 20 hours**. If your local computer goes to sleep or you lose internet connection, standard SSH terminals will terminate, aborting your training mid-run. 

We use **`tmux`** (Terminal Multiplexer) to keep our sessions running persistently on the cloud VM.

```bash
# 1. Start a new named persistent session
tmux new -s tinyyolo-run

# 2. Inside the tmux window, navigate to the project directory
cd ~/tinyYOLO
```

### How to manage tmux:
* **Detach** (Leave training running in the background): Press `Ctrl + B` then release and press `D`. You can now safely close your terminal or turn off your computer.
* **Reattach** (Check progress later):
  ```bash
  # SSH back into the VM, then run:
  tmux attach -t tinyyolo-run
  ```
* **Scroll inside tmux**: Press `Ctrl + B` then `[` (use arrow keys or PgUp/PgDn to navigate, press `Q` to exit scroll mode).

---

## Step 6: Running the Experiments Hands-Free

Here are the precise commands to execute your experimental workload. Run these commands inside your active `tmux` session.

### 6.1 Phase 1: Pascal VOC Training — 5 Seeds (Standard vs Quantized)
We write a short bash shell script directly in the terminal to execute all 5 seeds sequentially. This ensures that when one seed finishes, the next starts immediately without manual intervention.

#### Run 5 Seeds of the Quantized Variant (Primary Benchmark):
```bash
# Pre-download the VOC dataset to avoid concurrent download warnings
python3 -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('VOC.yaml')"

# Execute the 5-seed training loop sequentially
for SEED in 42 123 256 512 1024; do
  echo "========== STARTING QUANTIZED SEED $SEED =========="
  python3 scripts/train.py --task det --variant quantized --data voc.yaml \
    --imgsz 416 --epochs 300 --seed $SEED --warmup 3 --batch 128 \
    --name voc-q-416-seed${SEED}
done
```

#### Run 5 Seeds of the Standard Variant:
```bash
for SEED in 42 123 256 512 1024; do
  echo "========== STARTING STANDARD SEED $SEED =========="
  python3 scripts/train.py --task det --variant standard --data voc.yaml \
    --imgsz 416 --epochs 300 --seed $SEED --warmup 3 --batch 128 \
    --name voc-std-416-seed${SEED}
done
```

---

### 6.2 Phase 2: COCO val2017 Training (Secondary Benchmark)
We train the quantized and standard models on the COCO val2017 dataset.

> [!TIP]
> If your disk space or time is highly constrained, use `coco-val.yaml` (5K images) instead of `coco.yaml` (118K images). `coco-val.yaml` takes ~1–2 hours per run on a T4 GPU, while the full COCO training takes ~6–10 hours.

```bash
# Train Quantized variant on COCO
for SEED in 42 123 256; do
  python3 scripts/train.py --task det --variant quantized --data coco-val.yaml \
    --imgsz 416 --epochs 200 --seed $SEED --warmup 3 --batch 128 \
    --name cocoval-q-416-seed${SEED}
done

# Train Standard variant on COCO
python3 scripts/train.py --task det --variant standard --data coco-val.yaml \
  --imgsz 416 --epochs 200 --seed 42 --warmup 3 --batch 128 \
  --name cocoval-std-416-seed42
```

---

### 6.3 Phase 3: Ablation & Multi-Task Validation
Run these sequentially to validate specific claims in the paper:

```bash
# A6: Resolution Ablation (quantized variant, sweep 224 to 640)
python3 scripts/train.py --task det --variant quantized --data voc.yaml \
  --imgsz 224,320,416,640 --sweep --epochs 100 --seed 42 --warmup 3 --batch 128

# Multi-Task: Instance Segmentation (COCO-Seg)
python3 scripts/train.py --task seg --variant quantized --data coco-val.yaml \
  --imgsz 416 --epochs 200 --seed 42 --warmup 3 --batch 128 --name seg-q-416-seed42

# Multi-Task: Pose Estimation (COCO-Pose)
python3 scripts/train.py --task pose --variant quantized --data coco8-pose.yaml \
  --imgsz 416 --epochs 200 --seed 42 --warmup 3 --batch 128 --name pose-q-416-seed42
```

---

### 6.4 Phase 4: Quantization Pipeline (PTQ & QAT)
After training converges on FP32 models, apply Post-Training Quantization (PTQ) and Quantization-Aware Fine-Tuning (QAT) to generate native INT8 models:

```bash
# Post-Training Quantization (PTQ) calibration
python3 scripts/quantize.py --mode ptq \
  --weights experiments/results/voc-q-416-seed42/best.pt \
  --task det --variant quantized --data voc.yaml \
  --imgsz 416 --n-calib 500 --backend qnnpack

# Quantization-Aware Training (QAT) fine-tuning
python3 scripts/quantize.py --mode qat \
  --weights experiments/results/voc-q-416-seed42/best.pt \
  --task det --variant quantized --data voc.yaml \
  --imgsz 416 --epochs 10 --lr 1e-4 --backend qnnpack
```

---

## Step 7: Downloading Results & Cleaning Up

Once all experiments have run to completion, compile the results and pull them to your local environment.

### 7.1 Compile and Display Results Summaries on the VM
Execute the reporting script on the GCP VM to aggregate the mAP, Precision, Recall, and file size distributions:
```bash
python3 -c "
import json, glob, numpy as np
from pathlib import Path

print('='*80)
print('  GCP EXPERIMENTAL PIPELINE — COMPLETE SUMMARY')
print('='*80)

# VOC Analysis
for variant in ['q', 'std']:
    label = 'Quantized' if variant == 'q' else 'Standard'
    maps50 = []
    for f in sorted(glob.glob(f'experiments/results/voc-{variant}-416-seed*/config.json')):
        with open(f) as fh:
            cfg = json.load(fh)
        maps50.append(cfg.get('final_metrics', {}).get('mAP50', 0))
    if maps50:
        print(f'  {label:10s}: VOC mAP@50 = {np.mean(maps50)*100:.2f}% ± {np.std(maps50)*100:.2f}% (n={len(maps50)})')
"
```

### 7.2 Download the Experiments Folder to Your Local Machine
Open a **new terminal tab on your local machine** (not inside the SSH VM session) and run the `gcloud compute scp` command to recursively download the generated results:

```bash
# Run this on your LOCAL machine to copy results from the GCP VM
gcloud compute scp --recurse \
    tinyyolo-gpu-vm:~/tinyYOLO/experiments/results/ \
    /Applications/XAMPP/xamppfiles/htdocs/tinyYOLO/experiments/ \
    --zone="us-central1-a"
```

---

## Step 8: Destroying/Stopping the VM (Crucial to Avoid Extra Charges)

To prevent GCP from continuing to bill you after training is complete:

### Option A: Stop the Instance (Preserves VM state and files)
Use this if you plan to do more training later. You will only be billed a minor charge for the storage disk (~$5/month for 100 GB balanced disk), but **nothing** for the GPU or compute.
```bash
gcloud compute instances stop tinyyolo-gpu-vm --zone="us-central1-a"
```

### Option B: Delete the Instance (Completely removes everything)
Use this once you have successfully transferred all results to your local machine and no longer need the VM. **This stops all billing entirely.**
```bash
gcloud compute instances delete tinyyolo-gpu-vm --zone="us-central1-a" --quiet
```

---

## 💡 Pro-Tips for Peak GCP Performance

1. **Avoid CUDA Out-Of-Memory**: The `--batch 128` setting fits perfectly on the Tesla T4 (16 GB) at `416×416` resolution. If you choose to run at `640×640` resolution, reduce the batch size:
   ```bash
   # For 640x640 resolution, batch 64 or 32 is recommended:
   python3 scripts/train.py --task det --variant quantized --data voc.yaml --imgsz 640 --batch 64
   ```
2. **Accelerate with compiled operators**: Ensure PyTorch 2.0+ kernel compilation is active by passing the `--compile` flag to double training throughput!
3. **Automate VM Auto-Shutdown**: If you want the VM to shut down automatically as soon as the training loop finishes, add `sudo poweroff` at the end of your sequential command string:
   ```bash
   # Sequential runs + auto shutdown
   (for SEED in 42 123 256; do python3 scripts/train.py --task det --variant quantized --data voc.yaml --imgsz 416 --epochs 300 --seed $SEED --batch 128; done) && sudo poweroff
   ```
   *Note: Spot instances will shut down and stop charging immediately upon execution completion.*
