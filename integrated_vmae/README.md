# VMAE Unified Pipeline

This repository now has a unified command-line entrypoint to run the full workflow in one codebase:

1. missing-data simulation (preprocess)
2. training
3. fine-tuning
4. inference
5. experiment analysis + plots/reports

## Entry Point

Use:

```bash
python vmae_pipeline.py --help
```

Subcommands:

- `preprocess`
- `train`
- `finetune`
- `infer`
- `analyze`

## Integrated Code Layout

- Core unified pipeline: `vmae_pipeline.py`
- Main VMAE model/runtime: `MAE_LaMa.py`
- Experiment runtime + plots: `experiments.py`, `visualizer.py`
- Integrated MALA core (code-only): `integrated_mala_core/`
- Integrated analysis scripts: `analysis_tools/`
- Integration record: `INTEGRATION_NOTES.md`

## Data Path Strategy (No Re-upload Required)

The code is compatible with `E:/...` style paths to avoid re-uploading datasets.

The in-repo dataset tree under `E:/lama/...` is preserved.

Path resolution order:

1. direct path exists (Windows or local relative `E:/...` folder)
2. repository-relative `E:/...` path
3. optional mapping via env var `VMAE_DATA_ROOT`

Optional env vars:

- `VMAE_DATA_ROOT`: maps `E:/...` to another root
- `VMAE_MASK_DIR`: override predefined mask directory
- `VMAE_LAMA_INIT_DIR`: override LaMa init directory

Example:

```bash
export VMAE_DATA_ROOT=/mnt/e
```

Then `E:/lama/jet_S2_Daily_Mosaic/` is resolved as `/mnt/e/lama/jet_S2_Daily_Mosaic/`.

## Quick Commands

### 1) Preprocess (simulate missing data)

```bash
python vmae_pipeline.py preprocess \
  --data-dir "E:/lama/jet_S2_Daily_Mosaic/" \
  --mask-types "thin_cloud,cloud,strip,mixed" \
  --mask-ratios "0.1,0.3,0.5" \
  --num-sequences 2 \
  --output-dir preprocessed_missing_data
```

### 2) Train

```bash
python vmae_pipeline.py train \
  --data-dir "E:/lama/jet_S2_Daily_Mosaic/" \
  --ocean-mask-path "E:/lama/S2_Daily_Mosaic_Masked/mask.png" \
  --mask-type cloud \
  --mask-ratio 0.3 \
  --epochs 20 \
  --output-model checkpoints/vmae_trained.pth
```

### 3) Fine-tune

```bash
python vmae_pipeline.py finetune \
  --data-dir "E:/lama/jet_S2_Daily_Mosaic/" \
  --ocean-mask-path "E:/lama/S2_Daily_Mosaic_Masked/mask.png" \
  --pretrained-path checkpoints/vmae_trained.pth \
  --epochs 20 \
  --output-model checkpoints/vmae_finetuned.pth
```

### 4) Inference

```bash
python vmae_pipeline.py infer \
  --data-dir "E:/lama/jet_S2_Daily_Mosaic/" \
  --ocean-mask-path "E:/lama/S2_Daily_Mosaic_Masked/mask.png" \
  --model-path checkpoints/vmae_finetuned.pth
```

Outputs include:

- `inference_visualization.png`
- `inpainted_VMAE/reconstructed_frame_*.png`

### 5) Experiment Analysis

```bash
python vmae_pipeline.py analyze \
  --data-dir "E:/lama/jet_S2_Daily_Mosaic/" \
  --ocean-mask-path "E:/lama/S2_Daily_Mosaic_Masked/mask.png" \
  --model-path checkpoints/vmae_finetuned.pth \
  --ratios "10,30,50,60" \
  --mask-types "thin_cloud,strip,mixed"
```

Outputs include:

- `experiment_results/exp*/.../metrics.json`
- `experiment_results/figures/*.png`
- `experiment_results/comprehensive_analysis_report.txt`

## Notes

- For this machine, use `python` (Conda env) to ensure required packages are available.
- `MAE_LaMa.py` was updated for stronger path compatibility, safer filename parsing, and small-dataset stability.
