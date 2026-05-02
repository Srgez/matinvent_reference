#!/usr/bin/env bash
# Smoke run for GRPO-lite (Phase 1).
# Identical to smoke_rlaunch.sh but adds pipeline.grpo.mode=lite.
# Checks:
#   - advantage stats logged (mean≈0, std≈1 on full batch)
#   - ft loss and KL are finite
#   - checkpoint saves successfully
set -euo pipefail

# ========== Job Configuration ==========
JOB_NAME="${JOB_NAME:-matinvent-smoke-lite}"
GPU_NUM="${GPU_NUM:-2}"
MEMORY="${MEMORY:-200000}"
CPU_NUM="${CPU_NUM:-16}"
GROUP_NAME="${GROUP_NAME:-ai4sdata_gpu}"
MOUNT_PATH="${MOUNT_PATH:-gpfs://gpfs1/zhangsizhe:/mnt/shared-storage-user/zhangsizhe}"
IMAGE_NAME="${IMAGE_NAME:-registry.h.pjlab.org.cn/ailab-ai4sdata/zhangsizhe-workspace:20251221230837}"

# ========== Paths ==========
WORK_DIR="${WORK_DIR:-/mnt/shared-storage-user/zhangsizhe/matinvent_reference}"
MATTERGEN_CHECKPOINT_ROOT="${MATTERGEN_CHECKPOINT_ROOT:-/mnt/shared-storage-user/zhangsizhe/mattergen_steering_2/checkpoints}"
MATTERGEN_MODEL_PATH="${MATTERGEN_MODEL_PATH:-${MATTERGEN_CHECKPOINT_ROOT}/mattergen_base}"

# ========== GRPO-lite smoke settings ==========
EXP_NAME="${EXP_NAME:-smoke_grpo_lite}"
REWARD_NAME="ehull"
RESULTS_DIR="${RESULTS_DIR:-exp_res}"
LOGGER_NAME="${LOGGER_NAME:-csv}"
RL_EPOCH="${RL_EPOCH:-1}"
EVAL_SIZE="${EVAL_SIZE:-8}"               # enough samples for meaningful std
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-8}"
SAMPLE_NUM_BATCHES="${SAMPLE_NUM_BATCHES:-1}"
FT_BATCH_SIZE="${FT_BATCH_SIZE:-4}"
FT_TIMESTEPS="${FT_TIMESTEPS:-50}"
FT_EPOCHS="${FT_EPOCHS:-1}"
ACCUM_STEPS="${ACCUM_STEPS:-10}"
SIGMA="${SIGMA:-0.025}"
TOPK_RATIO="${TOPK_RATIO:-0.5}"         # top-50% for training; rest used for baseline
MATTERSIM_POTENTIAL_PATH="${MATTERSIM_POTENTIAL_PATH:-/mnt/shared-storage-user/zhangsizhe/mattergen_steering_2/checkpoints/mattersim/mattersim-v1.0.0-1M.pth}"
REFERENCE_DATASET_PATH="${REFERENCE_DATASET_PATH:-/mnt/shared-storage-user/zhangsizhe/mattergen/data-release/alex-mp/reference_MP2020correction.gz}"
GEMNET_SCALE_FILE="${GEMNET_SCALE_FILE:-/mnt/shared-storage-user/zhangsizhe/mattergen_steering_2/mattergen/common/gemnet/gemnet-dT.json}"
SAMPLING_CONFIG_PATH="${SAMPLING_CONFIG_PATH:-/mnt/shared-storage-user/zhangsizhe/mattergen_steering_2/sampling_conf}"

MATTERGEN_MODEL_NAME="${MATTERGEN_MODEL_NAME:-mattergen_base}"

# ========== GRPO hyperparameters ==========
GRPO_MODE="${GRPO_MODE:-lite}"
GRPO_ADV_EPS="${GRPO_ADV_EPS:-1e-8}"
GRPO_ADV_CLIP="${GRPO_ADV_CLIP:-5.0}"
GRPO_KL_WEIGHT_MODE="${GRPO_KL_WEIGHT_MODE:-fixed}"

# ========== Submit Job ==========
rlaunch \
  --gpu="${GPU_NUM}" \
  --memory="${MEMORY}" \
  --cpu="${CPU_NUM}" \
  --charged-group="${GROUP_NAME}" \
  --private-machine=group \
  --mount="${MOUNT_PATH}" \
  --image="${IMAGE_NAME}" \
  -e DISTRIBUTED_JOB=true \
  -- bash -c "
set -euo pipefail
printf '>>> Job started on: %s\n' \"\$(hostname)\"
echo '>>> GRPO-lite smoke run'
echo '>>> MatterGen: model=${MATTERGEN_MODEL_NAME}'
echo '>>> Experiment: ${EXP_NAME}'
echo '>>> GRPO mode: ${GRPO_MODE}, adv_clip=${GRPO_ADV_CLIP}, kl_weight_mode=${GRPO_KL_WEIGHT_MODE}'

cd \"${WORK_DIR}\"
source .venv/bin/activate
export MPLCONFIGDIR=\"${WORK_DIR}/.mplconfig\"
mkdir -p \"\${MPLCONFIGDIR}\"

MODELS_PROJECT_ROOT_RUNTIME=\"\$(python - <<'PY'
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
print(MODELS_PROJECT_ROOT)
PY
)\"
mkdir -p \"\${MODELS_PROJECT_ROOT_RUNTIME}/common/gemnet\"
ln -sf \"${GEMNET_SCALE_FILE}\" \"\${MODELS_PROJECT_ROOT_RUNTIME}/common/gemnet/gemnet-dT.json\"

python -u main.py \
  expname=${EXP_NAME} \
  results_dir=${RESULTS_DIR} \
  pipeline=mat_invent \
  model=mattergen \
  model.model_name=${MATTERGEN_MODEL_NAME} \
  +model.model_path=${MATTERGEN_MODEL_PATH} \
  reward=${REWARD_NAME} \
  logger=${LOGGER_NAME} \
  rl_epoch=${RL_EPOCH} \
  eval_size=${EVAL_SIZE} \
  sample_cfg.num_batches=${SAMPLE_NUM_BATCHES} \
  sample_cfg.max_num=${EVAL_SIZE} \
  ~sample_cfg.filter \
  reward.mattersim_potential_path=${MATTERSIM_POTENTIAL_PATH} \
  reward.mattersim_reference_dataset_path=${REFERENCE_DATASET_PATH} \
  model.sample_cfg.sampling_config_path=${SAMPLING_CONFIG_PATH} \
  model.sample_cfg.batch_size=${SAMPLE_BATCH_SIZE} \
  model.sample_cfg.num_batches=${SAMPLE_NUM_BATCHES} \
  model.finetune_cfg.batch_size=${FT_BATCH_SIZE} \
  model.finetune_cfg.timesteps=${FT_TIMESTEPS} \
  pipeline.finetune_cfg.batch_size=${FT_BATCH_SIZE} \
  pipeline.finetune_cfg.accum_steps=${ACCUM_STEPS} \
  pipeline.finetune_cfg.epochs=${FT_EPOCHS} \
  pipeline.finetune_cfg.sigma=${SIGMA} \
  pipeline.topk_ratio=${TOPK_RATIO} \
  pipeline.replay=False \
  pipeline.div_filter=False \
  pipeline.save_freq=1 \
  pipeline.grpo.mode=${GRPO_MODE} \
  pipeline.grpo.adv_eps=${GRPO_ADV_EPS} \
  pipeline.grpo.adv_clip=${GRPO_ADV_CLIP} \
  pipeline.grpo.kl_weight_mode=${GRPO_KL_WEIGHT_MODE}

echo '>>> GRPO-lite smoke run finished successfully'
"
