#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  # Load project env vars (e.g., ISLES_ROOT, GPU_DEVICE, WANDB_MODE).
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

ISLES_ROOT="${1:-${ISLES_ROOT:-/data/datasets/ISLES-2022}}"
GPU_DEVICE="${2:-${GPU_DEVICE:-0}}"

CSV_DIR="${CSV_DIR:-${REPO_ROOT}/src/data/csvs/isles2022}"
CFG_PATH="${CFG_PATH:-${REPO_ROOT}/scripts/isles2022_segmentation.yml}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/isles2022_segmentation}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
SIMCLR_CKPT="${SIMCLR_CKPT:-${REPO_ROOT}/src/checkpoints/BrainIAC.ckpt}"

TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-42}"

BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SW_BATCH_SIZE="${SW_BATCH_SIZE:-2}"
PRECISION="${PRECISION:-32}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-yes}"
RUN_NAME="${RUN_NAME:-isles2022_segmentation}"
PROJECT_NAME="${PROJECT_NAME:-brainiac_isles2022_segmentation}"

echo "[1/3] Build ISLES split CSVs from: ${ISLES_ROOT}"
python "${REPO_ROOT}/scripts/prepare_isles2022_segmentation_csv.py" \
  --isles-root "${ISLES_ROOT}" \
  --output-dir "${CSV_DIR}" \
  --train-ratio "${TRAIN_RATIO}" \
  --val-ratio "${VAL_RATIO}" \
  --seed "${SPLIT_SEED}"

echo "[2/3] Generate segmentation config: ${CFG_PATH}"
python "${REPO_ROOT}/scripts/make_isles2022_segmentation_config.py" \
  --train-csv "${CSV_DIR}/isles2022_train.csv" \
  --val-csv "${CSV_DIR}/isles2022_val.csv" \
  --simclr-ckpt "${SIMCLR_CKPT}" \
  --output-config "${CFG_PATH}" \
  --output-dir "${OUT_DIR}" \
  --log-dir "${LOG_DIR}" \
  --gpu-device "${GPU_DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-epochs "${MAX_EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --sw-batch-size "${SW_BATCH_SIZE}" \
  --precision "${PRECISION}" \
  --accumulate-grad-batches "${ACCUMULATE_GRAD_BATCHES}" \
  --gradient-clip-val "${GRADIENT_CLIP_VAL}" \
  --matmul-precision "${MATMUL_PRECISION}" \
  --freeze-backbone "${FREEZE_BACKBONE}" \
  --run-name "${RUN_NAME}" \
  --project-name "${PROJECT_NAME}"

echo "[3/3] Start training"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
if [[ -n "${WANDB_MODE:-}" ]]; then
  export WANDB_MODE
fi
python "${REPO_ROOT}/src/train_lightning_segmentation.py" --config "${CFG_PATH}"
