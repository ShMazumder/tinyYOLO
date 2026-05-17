"""
TinyYOLO Notebook Patcher
=========================
Programmatically update the Google Colab custom dataset notebook
to use the new robust ONNX export and ONNX Runtime quantization workflow,
along with robust validation dataset search logic.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / 'experiments' / '05_custom_dataset.ipynb'

PTQ_SOURCE = [
    "# %% Post-Training Quantization (PTQ) Calibration & ONNX Deployment\n",
    "# 1. Run PTQ calibration to create the INT8 PyTorch checkpoint (.pt)\n",
    "best_weights = str(run_folders[-1] / \"best.pt\") if run_folders else \"best.pt\"\n",
    "\n",
    "!python3 scripts/quantize.py \\\n",
    "    --mode ptq \\\n",
    "    --weights {best_weights} \\\n",
    "    --task det \\\n",
    "    --variant {VARIANT} \\\n",
    "    --data {data_yaml} \\\n",
    "    --imgsz {IMGSZ} \\\n",
    "    --n-calib 100 \\\n",
    "    --backend qnnpack\n",
    "\n",
    "# 2. Export the Float32 model to ONNX using self-healing variant loader\n",
    "!python3 scripts/export.py \\\n",
    "    --weights {best_weights} \\\n",
    "    --imgsz {IMGSZ} \\\n",
    "    --nc 2 \\\n",
    "    --variant {VARIANT}\n",
    "\n",
    "# 3. Quantize the ONNX model to INT8 using ONNX Runtime\n",
    "from pathlib import Path\n",
    "run_dir = run_folders[-1] if run_folders else Path(\"experiments/results/crime-detection-yolo-run\")\n",
    "float_onnx = str(run_dir / \"exports\" / \"best.onnx\")\n",
    "int8_onnx = str(run_dir / \"quantized\" / \"model_int8.onnx\")\n",
    "\n",
    "!pip install -q onnxruntime\n",
    "!python3 scripts/quantize_onnx.py \\\n",
    "    --input {float_onnx} \\\n",
    "    --output {int8_onnx} \\\n",
    "    --mode dynamic"
]

QAT_SOURCE = [
    "# %% Quantization-Aware Fine-Tuning (QAT) & ONNX Deployment\n",
    "# 1. Run QAT training to create the INT8 PyTorch checkpoint (.pt)\n",
    "best_weights = str(run_folders[-1] / \"best.pt\") if run_folders else \"best.pt\"\n",
    "\n",
    "!python3 scripts/quantize.py \\\n",
    "    --mode qat \\\n",
    "    --weights {best_weights} \\\n",
    "    --task det \\\n",
    "    --variant {VARIANT} \\\n",
    "    --data {data_yaml} \\\n",
    "    --imgsz {IMGSZ} \\\n",
    "    --epochs 5 \\\n",
    "    --lr 1e-4 \\\n",
    "    --backend qnnpack\n",
    "\n",
    "# 2. Export the Float32 model to ONNX using self-healing variant loader\n",
    "!python3 scripts/export.py \\\n",
    "    --weights {best_weights} \\\n",
    "    --imgsz {IMGSZ} \\\n",
    "    --nc 2 \\\n",
    "    --variant {VARIANT}\n",
    "\n",
    "# 3. Quantize the ONNX model to INT8 using ONNX Runtime\n",
    "from pathlib import Path\n",
    "run_dir = run_folders[-1] if run_folders else Path(\"experiments/results/crime-detection-yolo-run\")\n",
    "float_onnx = str(run_dir / \"exports\" / \"best.onnx\")\n",
    "int8_onnx = str(run_dir / \"quantized\" / \"model_int8_qat.onnx\")\n",
    "\n",
    "!pip install -q onnxruntime\n",
    "!python3 scripts/quantize_onnx.py \\\n",
    "    --input {float_onnx} \\\n",
    "    --output {int8_onnx} \\\n",
    "    --mode dynamic"
]

INFERENCE_SOURCE = [
    "# %% Execute inference on validation sample\n",
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "# 1. Dynamically resolve dataset folder if session restarted\n",
    "if 'dataset_dir' not in locals() or dataset_dir is None:\n",
    "    dataset_candidates = sorted(list(Path(\".\").glob(\"crime-detection-*\")))\n",
    "    if not dataset_candidates:\n",
    "        dataset_candidates = sorted(list(Path(\"datasets\").glob(\"*\")))\n",
    "    dataset_dir = dataset_candidates[0] if dataset_candidates else Path(\"datasets\")\n",
    "\n",
    "# 2. Dynamically resolve variables if session restarted\n",
    "if 'IMGSZ' not in locals():\n",
    "    IMGSZ = 416\n",
    "if 'VARIANT' not in locals():\n",
    "    VARIANT = \"quantized\"\n",
    "\n",
    "runs_dir = Path(\"experiments/results\")\n",
    "run_folders = sorted(runs_dir.glob(\"*crime-detection-yolo-run*\"))\n",
    "\n",
    "# ----------------- SELECT TARGET MODEL WEIGHTS -----------------\n",
    "# Option 1: Standard FP32 Weights\n",
    "TARGET_WEIGHTS = str(run_folders[-1] / \"best.pt\") if run_folders else \"experiments/results/crime-detection-yolo-run/best.pt\"\n",
    "\n",
    "# Option 2: INT8 PTQ Quantized Weights (uncomment to test!)\n",
    "# TARGET_WEIGHTS = str(run_folders[-1] / \"quantized\" / \"model_int8_ptq.pt\") if run_folders else \"experiments/results/crime-detection-yolo-run/quantized/model_int8_ptq.pt\"\n",
    "\n",
    "# Option 3: INT8 QAT Fine-Tuned Weights (uncomment to test!)\n",
    "# TARGET_WEIGHTS = str(run_folders[-1] / \"quantized\" / \"model_int8_qat.pt\") if run_folders else \"experiments/results/crime-detection-yolo-run/quantized/model_int8_qat.pt\"\n",
    "# ---------------------------------------------------------------\n",
    "\n",
    "current_nc = nc if 'nc' in locals() else 2\n",
    "current_names = names if 'names' in locals() else {0: \"weapon\", 1: \"violence\"}\n",
    "\n",
    "weights_path = Path(TARGET_WEIGHTS)\n",
    "\n",
    "# Dynamically find the first image in the validation folder (supporting multiple formats)\n",
    "val_images = sorted(list(dataset_dir.glob(\"val/images/*\")))\n",
    "if not val_images:\n",
    "    val_images = sorted(list(dataset_dir.glob(\"images/val/*\")))\n",
    "if not val_images:\n",
    "    val_images = sorted(list(dataset_dir.glob(\"valid/images/*\")))\n",
    "if not val_images:\n",
    "    val_images = sorted(list(dataset_dir.glob(\"images/valid/*\")))\n",
    "\n",
    "if weights_path.exists() and val_images:\n",
    "    print(f\"\u2713 Loading target weights: {weights_path}\")\n",
    "    print(f\"\u2713 Found test image: {val_images[0]}\")\n",
    "    run_custom_inference(\n",
    "        image_path=val_images[0],\n",
    "        weights_path=str(weights_path),\n",
    "        nc=current_nc,\n",
    "        class_names=current_names,\n",
    "        imgsz=IMGSZ\n",
    "    )\n",
    "else:\n",
    "    print(f\"\u26a0\ufe0f Error: target weights ({weights_path}) or validation images folder is empty. Please verify paths!\")\n"
]


def patch_notebook():
    if not NOTEBOOK_PATH.exists():
        print(f"  [ERROR] Notebook not found at {NOTEBOOK_PATH}")
        return

    with open(NOTEBOOK_PATH, 'r') as f:
        nb_data = json.load(f)

    ptq_patched = False
    qat_patched = False
    infer_patched = False

    for cell in nb_data.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell_id = cell.get('metadata', {}).get('id')
            if cell_id == 'ptq_calib':
                cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in PTQ_SOURCE]
                ptq_patched = True
                print("  [SUCCESS] Patched PTQ Calibration Cell!")
            elif cell_id == 'qat_fine_tuning':
                cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in QAT_SOURCE]
                qat_patched = True
                print("  [SUCCESS] Patched QAT Fine-Tuning Cell!")
            elif cell_id == 'run_inference_code':
                cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in INFERENCE_SOURCE]
                infer_patched = True
                print("  [SUCCESS] Patched Inference Execution Cell!")

    if ptq_patched or qat_patched or infer_patched:
        with open(NOTEBOOK_PATH, 'w') as f:
            json.dump(nb_data, f, indent=1)
        print("  [SUCCESS] Notebook saved successfully!")
    else:
        print("  [WARN] Target cells not found. No patches applied.")


if __name__ == '__main__':
    patch_notebook()
