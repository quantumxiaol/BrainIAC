# BrainIAC: A foundation model for generalized Brain MRI analysis

<p align="center">
  <img src="pngs/brainiac.jpeg" width="800" alt="BrainIAC_V2 Logo"/>
</p>

## Overview

BrainIAC (Brain Imaging Adaptive Core) is vision based foundation model for generalized structural Brain MRI analysis. This repository provides the BrainIAC and downstream model checkpoints, with training/inference pipeline across all downstream tasks. Checkout the [Paper]([https://pmc.ncbi.nlm.nih.gov/articles/PMC11643205/](https://www.nature.com/articles/s41593-026-02202-6))

## Installation

### Prerequisites
- Python 3.9 - 3.12
- CUDA 11.0+ (if running on GPU)

### Option A (Recommended): uv

```bash
# Clone
git clone https://github.com/YourUsername/BrainIAC_V2.git
cd BrainIAC

# (Optional) Use TUNA mirror in mainland China
uv lock --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
uv sync --extra test --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

# Or without mirror
# uv sync --extra test

source .venv/bin/activate
```

### Option B: conda + pip

```bash
git clone https://github.com/YourUsername/BrainIAC_V2.git
cd BrainIAC

conda create -n brainiac python=3.10
conda activate brainiac

# Use pinned runtime deps from requirements.txt (important for MONAI compatibility)
pip install -r requirements.txt

# Install this repo in editable mode without upgrading pinned deps
pip install -e . --no-deps

# Test dependency
pip install pytest
```

### Verify Runtime Versions

```bash
python - <<'PY'
import monai, torch, pytorch_lightning
print("monai", monai.__version__)  # expected: 1.3.2
print("torch", torch.__version__)
print("pl", pytorch_lightning.__version__)
PY
```

## Model Checkpoints

Download and extract checkpoints into `./src/checkpoints/`:

```bash
python download_checkpoints.py
```

If your runtime environment cannot access the internet, download on another machine and then:

```bash
python download_checkpoints.py --archive /path/to/checkpoints.zip --output-dir ./src/checkpoints
```

Some inference scripts use these default names:

```bash
cd src/checkpoints
ln -sf vit_mci.ckpt mci.ckpt
ln -sf os_model.pt os.ckpt
ln -sf sequence_classifcation.ckpt multiclass.ckpt
```

## MCP Service (Single-Patient Infarct Segmentation)

An MCP server is available at `src/mcp-service/server.py` with one tool:
- `segment_infarct_single_patient(patient_image_path)`

Input:
- `patient_image_path`: absolute path to one `.nii`/`.nii.gz` patient image file

Output:
- absolute path to the generated segmentation mask

Default checkpoints/output:
- segmentation checkpoint: `src/checkpoints/segmentation.ckpt`
- BrainIAC backbone checkpoint: `src/checkpoints/BrainIAC.ckpt`
- output dir: `inference/mcp_outputs`
- HTTP host/port (when using `--http`): `127.0.0.1:8001`

The server auto-loads project-root `.env` (if present), so server-side deployment is easiest with:

```bash
cp .env.example .env
# edit .env
```

You can also override by exporting env vars:

```bash
export BRAINIAC_SEGMENTATION_CKPT=/abs/path/to/segmentation.ckpt
export BRAINIAC_SIMCLR_CKPT=/abs/path/to/BrainIAC.ckpt
export BRAINIAC_OUTPUT_DIR=/abs/path/to/output_dir
export BRAINIAC_GPU_DEVICE=0
export BRAINIAC_MCP_HOST=127.0.0.1
export BRAINIAC_MCP_PORT=8001
```

Start MCP server (STDIO, for local agent process):

```bash
python src/mcp-service/server.py
```

Start MCP server (HTTP + SSE + streamable-http):

```bash
python src/mcp-service/server.py --http --host 127.0.0.1 --port 8001
```

Run MCP server in background with `tmux`:

```bash
# 1) Create detached tmux session
tmux new-session -d -s brainiac-mcp

# 2) Start MCP service in that session (and write logs)
tmux send-keys -t brainiac-mcp 'cd ~/BrainIAC && source .venv/bin/activate && mkdir -p logs && python src/mcp-service/server.py --http --host 127.0.0.1 --port 8001 |& tee logs/mcp_server.log' C-m

# 3) Check running sessions
tmux ls

# 4) Attach to session for live logs
tmux attach -t brainiac-mcp
# Detach without stopping: Ctrl-b then d

# 5) Stop MCP service/session
tmux kill-session -t brainiac-mcp
```

## Testing

Basic tests (fast, no heavy model inference):

```bash
pytest -q -m "not integration"
```

Integration tests:

```bash
pytest -q -m integration
```

GPU integration tests only:

```bash
pytest -q -m "integration and gpu"
```

Notes:
- Tests are under `./test` and configured by `pytest.ini`.
- Integration tests auto-skip when required checkpoints/sample data/GPU are missing.

## ISLES-2022 Training

The training launcher `scripts/train_isles2022_segmentation.sh` automatically reads project-root `.env` if present.

```bash
cp .env.example .env
# edit .env as needed (dataset path, gpu id, output dir, log dir, etc.)
```

Start training:

```bash
bash scripts/train_isles2022_segmentation.sh
```

Stability note:
- Default `.env.example` uses `PRECISION=32`, `FREEZE_BACKBONE=no`, `ENCODER_LR=5e-5`, `DECODER_LR=1e-4`.
- If stable, you can try `PRECISION=16-mixed` for speed.
- Default `.env.example` uses `IMAGE_MODALITY=dwi_adc` (early-fusion 2-channel) and `REQUIRE_ALIGNED=yes` so image/mask pairs are kept in the same voxel space.

Command-line arguments still override `.env`:

```bash
bash scripts/train_isles2022_segmentation.sh /data/datasets/ISLES-2022 0
```

Segmentation evaluation (`src/test_segmentation.py`) reports:
- voxel metrics: Dice / IoU / Precision / Recall
- lesion metrics: lesion-wise F1 / lesion count diff
- volume metric: AVD (absolute volume difference, %)
- post-processing controls: `--threshold` (default `0.5`), `--min_lesion_voxels` (remove tiny connected components)
- threshold sweep: `--sweep_thresholds` + `--sweep_csv` (auto pick best threshold on validation CSV)

Example (fixed threshold + small-lesion filtering):

```bash
python src/test_segmentation.py \
  --config scripts/isles2022_segmentation.yml \
  --test_csv src/data/csvs/isles2022/isles2022_test.csv \
  --checkpoint_path /path/to/best.ckpt \
  --experiment_name isles2022_dwi_adc_post \
  --threshold 0.45 \
  --min_lesion_voxels 20
```

Example (auto threshold selection on validation CSV):

```bash
python src/test_segmentation.py \
  --config scripts/isles2022_segmentation.yml \
  --test_csv src/data/csvs/isles2022/isles2022_test.csv \
  --checkpoint_path /path/to/best.ckpt \
  --experiment_name isles2022_dwi_adc_sweep \
  --sweep_csv src/data/csvs/isles2022/isles2022_val.csv \
  --sweep_thresholds 0.30,0.35,0.40,0.45,0.50 \
  --sweep_metric lesion_f1 \
  --min_lesion_voxels 20
```





## Quick Start

See [quickstart.ipynb](./src/quickstart.ipynb) to get started on how to preprocess data, load BrainIAC to extract features, generate and visualize saliency maps. We provide data samples from publicly available [UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/) [License](https://creativecommons.org/licenses/by/4.0/) (with no modifications to the provided preprocessed images) and the [Pixar](https://openneuro.org/datasets/ds000228/versions/1.1.1)  [License](https://creativecommons.org/public-domain/cc0/) dataset in the [sample_data](src/data/sample/processed/) directory. 


## Train and Infer Downstream Models

- [Brain Age Prediction](./docs/downstream_tasks/brain_age_prediction.md)
- [IDH Mutation Classification](./docs/downstream_tasks/idh_mutation_classification.md)
- [Mild Cognitive Impairment Classification](./docs/downstream_tasks/mild_cognitive_impairment_classification.md)
- [Diffuse Glioma Overall Survival Prediction](./docs/downstream_tasks/diffuse_glioma_overall_survival.md)
- [MR Sequence Classification](./docs/downstream_tasks/MR_sequence_classification.md)
- [Time to Stroke Prediction](./docs/downstream_tasks/timetostroke_prediction.md)
- [Tumor Segmentation](./docs/downstream_tasks/tumor_segmentation.md)


## Brainiac Platform

BrainIAC and all the downstream models are hosted at [**Brainiac Platform**](https://www.brainiac-platform.com/) for inference. The platform provides an interface for uploading the structural brain MRI data and running inference on the models including BrainIAC, brain age, MCI classification, time since stroke prediction etc.



## Citation

```bibtex
@article{tak2026generalizable,
  title={A generalizable foundation model for analysis of human brain MRI},
  author={Tak, Divyanshu and Gormosa, B.A. and Zapaishchykova, A. and others},
  journal={Nature Neuroscience},
  year={2026},
  publisher={Springer Nature},
  doi={10.1038/s41593-026-02202-6},
  url={https://doi.org/10.1038/s41593-026-02202-6}
}
```


## License

This project is licensed for non-commercial academic research use only.
Commercial use is not permitted without a separate license from
Mass General Brigham.

For commercial licensing inquiries, please contact the
Mass General Brigham Office of Technology Development. See [LICENSE](LICENSE) for details.
