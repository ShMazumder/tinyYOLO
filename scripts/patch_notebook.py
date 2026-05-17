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

INFER_SOURCE = [
    "# %% Bounding box inference helper function\n",
    "# Note: Function is also defined inline below in the execution cell for absolute safety!\n",
    "pass"
]

INFERENCE_SOURCE = [
    "# %% Execute inference on validation sample\n",
    "import cv2\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import matplotlib.pyplot as plt\n",
    "from pathlib import Path\n",
    "from tinyYOLO.models import build_model\n",
    "from tinyYOLO.utils.postprocess import postprocess_detections\n",
    "\n",
    "def run_custom_inference(image_path, weights_path, nc, class_names, imgsz=416):\n",
    "    device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "    \n",
    "    # Build Float32 model\n",
    "    model, _ = build_model(task='det', variant=VARIANT, nc=nc)\n",
    "    checkpoint = torch.load(weights_path, map_location='cpu')\n",
    "    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint\n",
    "    \n",
    "    # Check if checkpoint is quantized (static INT8 checkpoint)\n",
    "    is_quantized = any('scale' in k or 'zero_point' in k for k in state_dict.keys())\n",
    "    \n",
    "    if is_quantized:\n",
    "        print(\"✓ Detected quantized PyTorch checkpoint! Preparing and converting model for INT8 execution...\")\n",
    "        device = 'cpu'  # PyTorch eager-mode quantized operations run on CPU only\n",
    "        import torch.ao.quantization as ao_quant\n",
    "        \n",
    "        # Define wrapper classes inline\n",
    "        class DequantizedHeadWrapper(nn.Module):\n",
    "            def __init__(self, head):\n",
    "                super().__init__()\n",
    "                self.dequant = ao_quant.DeQuantStub()\n",
    "                self.head = head\n",
    "            def forward(self, x):\n",
    "                if isinstance(x, (list, tuple)):\n",
    "                    return self.head([self.dequant(t) for t in x])\n",
    "                return self.head(self.dequant(x))\n",
    "                \n",
    "        class QuantizedWrapper(nn.Module):\n",
    "            def __init__(self, model):\n",
    "                super().__init__()\n",
    "                self.quant = ao_quant.QuantStub()\n",
    "                if hasattr(model, 'head') and not isinstance(model.head, DequantizedHeadWrapper):\n",
    "                    model.head = DequantizedHeadWrapper(model.head)\n",
    "                self.model = model\n",
    "            def forward(self, x):\n",
    "                x = self.quant(x)\n",
    "                return self.model(x)\n",
    "                \n",
    "        # Wrap Float32 model\n",
    "        wrapped_model = QuantizedWrapper(model)\n",
    "        \n",
    "        # Configure engine & qconfig\n",
    "        backend = 'qnnpack'\n",
    "        torch.backends.quantized.engine = backend\n",
    "        wrapped_model.qconfig = ao_quant.get_default_qconfig(backend)\n",
    "        if hasattr(wrapped_model.model, 'head'):\n",
    "            if isinstance(wrapped_model.model.head, DequantizedHeadWrapper):\n",
    "                wrapped_model.model.head.head.qconfig = None\n",
    "            else:\n",
    "                wrapped_model.model.head.qconfig = None\n",
    "                \n",
    "        # Prepare & convert model structure to quantized\n",
    "        wrapped_model.eval()\n",
    "        ao_quant.prepare(wrapped_model, inplace=True)\n",
    "        model = ao_quant.convert(wrapped_model, inplace=False)\n",
    "        \n",
    "        # Load state dict directly (no model. stripping needed as wrapper keys match perfectly)\n",
    "        model.load_state_dict(state_dict)\n",
    "    else:\n",
    "        # Strip compile/DDP prefixes recursively for Float32 checkpoints\n",
    "        cleaned_state_dict = {}\n",
    "        for k, v in state_dict.items():\n",
    "            if k.endswith(('total_ops', 'total_params')):\n",
    "                continue\n",
    "            k_clean = k\n",
    "            while k_clean.startswith('_orig_mod.') or k_clean.startswith('module.'):\n",
    "                if k_clean.startswith('_orig_mod.'):\n",
    "                    k_clean = k_clean[len('_orig_mod.'):]\n",
    "                if k_clean.startswith('module.'):\n",
    "                    k_clean = k_clean[len('module.'):]\n",
    "            if k_clean.startswith('model.'):\n",
    "                k_clean = k_clean[len('model.'):]\n",
    "            cleaned_state_dict[k_clean] = v\n",
    "            \n",
    "        model.load_state_dict(cleaned_state_dict)\n",
    "        \n",
    "    model.to(device).eval()\n",
    "    \n",
    "    # Preprocess image\n",
    "    img0 = cv2.imread(str(image_path))\n",
    "    h0, w0 = img0.shape[:2]\n",
    "    img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)\n",
    "    img_resized = cv2.resize(img_rgb, (imgsz, imgsz))\n",
    "    \n",
    "    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0\n",
    "    img_tensor = img_tensor.unsqueeze(0).to(device)\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        preds = model(img_tensor)\n",
    "        \n",
    "    # Decode predictions & apply NMS\n",
    "    detections = postprocess_detections(preds, conf_thres=0.25, iou_thres=0.45)[0]\n",
    "    \n",
    "    fig, ax = plt.subplots(figsize=(10, 8))\n",
    "    ax.imshow(img_rgb)\n",
    "    \n",
    "    if len(detections) > 0:\n",
    "        print(f\"\u2713 Detected {len(detections)} objects:\")\n",
    "        for det in detections:\n",
    "            x1, y1, x2, y2 = int(det[0]*w0), int(det[1]*h0), int(det[2]*w0), int(det[3]*h0)\n",
    "            score = det[4]\n",
    "            cls_id = int(det[5])\n",
    "            \n",
    "            name = class_names[cls_id] if class_names and cls_id in class_names else f\"cls_{cls_id}\"\n",
    "            print(f\"  - {name}: {score*100:.1f}%\")\n",
    "            \n",
    "            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='#ff0000', linewidth=3)\n",
    "            ax.add_patch(rect)\n",
    "            ax.text(x1, y1-10, f\"{name} {score:.2f}\", color='white', fontsize=12,\n",
    "                    bbox=dict(facecolor='#ff0000', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))\n",
    "    else:\n",
    "        print(\"No objects detected.\")\n",
    "    \n",
    "    plt.axis('off')\n",
    "    plt.show()\n",
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
    "# TARGET_WEIGHTS = str(run_folders[-1] / \"best.pt\") if run_folders else \"experiments/results/crime-detection-yolo-run/best.pt\"\n",
    "\n",
    "# Option 2: INT8 PTQ Quantized Weights\n",
    "TARGET_WEIGHTS = str(run_folders[-1] / \"quantized\" / \"model_int8_ptq.pt\") if run_folders else \"experiments/results/crime-detection-yolo-run/quantized/model_int8_ptq.pt\"\n",
    "\n",
    "# Option 3: INT8 QAT Fine-Tuned Weights\n",
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
    infer_code_patched = False

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
            elif cell_id == 'infer':
                cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in INFER_SOURCE]
                infer_patched = True
                print("  [SUCCESS] Patched Inference Helper Function Cell!")
            elif cell_id == 'run_inference_code':
                cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in INFERENCE_SOURCE]
                infer_code_patched = True
                print("  [SUCCESS] Patched Inference Execution Cell!")

    if ptq_patched or qat_patched or infer_patched or infer_code_patched:
        with open(NOTEBOOK_PATH, 'w') as f:
            json.dump(nb_data, f, indent=1)
        print("  [SUCCESS] Notebook saved successfully!")
    else:
        print("  [WARN] Target cells not found. No patches applied.")


if __name__ == '__main__':
    patch_notebook()
