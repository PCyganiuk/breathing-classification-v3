# ONNX export troubleshooting guide

## Problem

If export fails with an error like:

- `SymbolicValueError`
- `STFT does not currently support complex types`

it usually means the old ONNX exporter is hitting Torch's symbolic handling for the mel-spectrogram preprocessing path.

## Root cause

This project uses `torchaudio.transforms.MelSpectrogram` inside the preprocessing wrapper. In the affected environment, the legacy exporter path could not handle that graph correctly during ONNX export.

## Fix that worked

1. Activate the project virtual environment.
2. Install the exporter dependencies.
3. Switch the export script to the newer Dynamo-based exporter.
4. Resolve config/model paths relative to the script file.
5. Re-run the exporter.

## Step-by-step

### 1) Activate the environment

```bash
cd /home/pyza/Projects/breathing-classification-v2
source breathing_model/venv/bin/activate
```

### 2) Install required packages

```bash
pip install onnx==1.17.0 onnxscript
```

### 3) Update the exporter script

Use the following pattern in the export script:

```python
from pathlib import Path
import torch

config_path = Path(__file__).resolve().parent / "config.yaml"
model_path = Path(__file__).resolve().parent / "best_models" / "best_model_epoch_31.pth"
onnx_path = Path(__file__).resolve().parent / "best_models" / "best_model_epoch_31.onnx"

torch.onnx.export(
    full_model,
    dummy_input,
    str(onnx_path),
    export_params=True,
    do_constant_folding=True,
    input_names=["audio_input"],
    output_names=["logits"],
    dynamo=True,
    verbose=False,
)
```

### 4) Run the exporter

```bash
python breathing_model/model/transformer/export_to_onnx.py
```

## Verification

If the export succeeds, you should see output similar to:

```text
Breath classifier model exported and verified: .../best_model_epoch_31.onnx
ONNX export complete.
```

You can also verify that the file exists:

```bash
ls -lh breathing_model/model/transformer/best_models/best_model_epoch_31.onnx
```

## If it still fails

If the problem persists, the next fallback is to replace the mel-spectrogram preprocessing with a custom ONNX-friendly implementation or to export from a different PyTorch/torchvision/torchaudio combination.

## Notes

This repository-specific fix was validated with:

- Python 3.12
- PyTorch 2.7.1
- torchaudio 2.7.1
- ONNX 1.17.0
- onnxscript
