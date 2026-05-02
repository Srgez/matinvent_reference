#!/usr/bin/env bash
# ============================================================
# Resume GRPO-lite training from a saved checkpoint.
#
# Context:
#   Previous run (grpo_lite_ehull_v1_my) was interrupted after
#   completing Loop 27 (step counter 0–27).  The last saved
#   checkpoint is loop_0019/last.ckpt (saved after Loop 19).
#   Loops 20–27 ran but their model updates were not saved.
#
# Resume strategy:
#   - Load model from loop_0019/last.ckpt
#   - Set pipeline.start_loop=20  → loop counter starts at 20,
#     so sample/reward file names continue from step_0020_*
#     (no collision with existing step_0000–0019 files)
#   - rl_epoch=120 (unchanged); effective remaining = 120 − 20 = 100
#   - Write outputs to the same EXP directory so all results are
#     in one place; wandb appends to a new offline run in the
#     same wandb/ folder
#
# Notes on what is NOT resumed:
#   - Replay buffer: starts empty; will refill within ~20 loops
#     given reward_cutoff=0.65 and buffer_reward_mean was ~0.999
#   - Long-term memory (diversity tracker): starts fresh
#   - WandB: new offline run (sync both runs to compare)
# ============================================================
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Job configuration
# ──────────────────────────────────────────────────────────────
JOB_NAME="${JOB_NAME:-matinvent-lite-ehull-resume}"
GPU_NUM="${GPU_NUM:-4}"
MEMORY="${MEMORY:-400000}"
CPU_NUM="${CPU_NUM:-32}"
GROUP_NAME="${GROUP_NAME:-ai4sdata_gpu}"
MOUNT_PATH="${MOUNT_PATH:-gpfs://gpfs1/zhangsizhe:/mnt/shared-storage-user/zhangsizhe}"
IMAGE_NAME="${IMAGE_NAME:-registry.h.pjlab.org.cn/ailab-ai4sdata/zhangsizhe-workspace:20251221230837}"

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
WORK_DIR="${WORK_DIR:-/mnt/shared-storage-user/zhangsizhe/matinvent_reference}"
MATTERGEN_CHECKPOINT_ROOT="${MATTERGEN_CHECKPOINT_ROOT:-/mnt/shared-storage-user/zhangsizhe/mattergen_steering_2/checkpoints}"
MATTERSIM_POTENTIAL_PATH="${MATTERSIM_POTENTIAL_PATH:-${MATTERGEN_CHECKPOINT_ROOT}/mattersim/mattersim-v1.0.0-1M.pth}"
REFERENCE_DATASET_PATH="${REFERENCE_DATASET_PATH:-/mnt/shared-storage-user/zhangsizhe/mattergen/data-release/alex-mp/reference_MP2020correction.gz}"
GEMNET_SCALE_FILE="${GEMNET_SCALE_FILE:-/mnt/shared-storage-user/zhangsizhe/mattergen_steering_2/mattergen/common/gemnet/gemnet-dT.json}"
SAMPLING_CONFIG_PATH="${SAMPLING_CONFIG_PATH:-/mnt/shared-storage-user/zhangsizhe/mattergen_steering_2/sampling_conf}"

# ──────────────────────────────────────────────────────────────
# Resume configuration — EDIT THESE if resuming from a different checkpoint
# ──────────────────────────────────────────────────────────────
# Directory of the experiment to resume
RESUME_EXP_NAME="${RESUME_EXP_NAME:-grpo_lite_ehull_v1_my}"
RESULTS_DIR="${RESULTS_DIR:-exp_res}"

# Checkpoint directory (contains last.ckpt and config.yaml)
RESUME_CKPT_DIR="${RESUME_CKPT_DIR:-${WORK_DIR}/${RESULTS_DIR}/${RESUME_EXP_NAME}/models/loop_0019}"

# Which loop to resume FROM (loops 0..START_LOOP-1 are skipped)
# Set to the loop number AFTER the last saved checkpoint.
# Checkpoint saved at loop_0019 means training completed loops 0–19, so resume from 20.
START_LOOP="${START_LOOP:-20}"

# ──────────────────────────────────────────────────────────────
# Experiment identity — write outputs to the SAME experiment dir
# ──────────────────────────────────────────────────────────────
EXP_NAME="${EXP_NAME:-${RESUME_EXP_NAME}}"
LOGGER_NAME="${LOGGER_NAME:-wandb}"
MATTERGEN_MODEL_NAME="${MATTERGEN_MODEL_NAME:-my_train_mattergen_base}"

# ──────────────────────────────────────────────────────────────
# RL loop — keep at original total; start_loop handles the offset
# ──────────────────────────────────────────────────────────────
RL_EPOCH="${RL_EPOCH:-120}"

# ──────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────
EVAL_SIZE="${EVAL_SIZE:-64}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-32}"
SAMPLE_NUM_BATCHES="${SAMPLE_NUM_BATCHES:-3}"

# ──────────────────────────────────────────────────────────────
# Finetune
# ──────────────────────────────────────────────────────────────
FT_BATCH_SIZE="${FT_BATCH_SIZE:-32}"
FT_TIMESTEPS="${FT_TIMESTEPS:-300}"
FT_EPOCHS="${FT_EPOCHS:-3}"
ACCUM_STEPS="${ACCUM_STEPS:-50}"
LR="${LR:-5e-6}"
SIGMA="${SIGMA:-0.025}"
TOPK_RATIO="${TOPK_RATIO:-0.5}"
SAVE_FREQ="${SAVE_FREQ:-10}"

# ──────────────────────────────────────────────────────────────
# Experience replay
# ──────────────────────────────────────────────────────────────
REPLAY="${REPLAY:-True}"
BUFFER_SIZE="${BUFFER_SIZE:-300}"
SAMPLE_SIZE="${SAMPLE_SIZE:-16}"
REWARD_CUTOFF="${REWARD_CUTOFF:-0.65}"

# ──────────────────────────────────────────────────────────────
# Diversity filter
# ──────────────────────────────────────────────────────────────
DIV_FILTER="${DIV_FILTER:-True}"
DF_TOL="${DF_TOL:-3}"
DF_BUFF="${DF_BUFF:-6}"

# ──────────────────────────────────────────────────────────────
# GRPO-lite hyperparameters
# ──────────────────────────────────────────────────────────────
GRPO_MODE="${GRPO_MODE:-lite}"
GRPO_ADV_EPS="${GRPO_ADV_EPS:-1e-8}"
GRPO_ADV_CLIP="${GRPO_ADV_CLIP:-3.0}"
GRPO_KL_WEIGHT_MODE="${GRPO_KL_WEIGHT_MODE:-fixed}"

# ──────────────────────────────────────────────────────────────
# Reward
# ──────────────────────────────────────────────────────────────
REWARD_NAME="ehull"

# ──────────────────────────────────────────────────────────────
# Submit
# ──────────────────────────────────────────────────────────────
rjob submit \
  --name="${JOB_NAME}" \
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

OUT_DIR=\"${WORK_DIR}/${RESULTS_DIR}/${EXP_NAME}\"
mkdir -p \"\${OUT_DIR}\"

# Copy this resume script for reproducibility
cp \"${WORK_DIR}/run_on_h/resume_grpo_lite_ehull.sh\" \"\${OUT_DIR}/resume_launch.sh\"

LOG_FILE=\"\${OUT_DIR}/train.log\"
exec > >(tee -a \"\${LOG_FILE}\") 2>&1

echo '==================================================='
echo 'GRPO-lite RESUME — ehull reward'
echo '==================================================='
echo \"  EXP_NAME          = ${EXP_NAME}\"
echo \"  OUT_DIR           = \${OUT_DIR}\"
echo \"  RESUME_CKPT_DIR   = ${RESUME_CKPT_DIR}\"
echo \"  START_LOOP        = ${START_LOOP}  (skipping loops 0..$(( START_LOOP - 1 )))\"
echo \"  RL_EPOCH          = ${RL_EPOCH}  (remaining = $(( RL_EPOCH - START_LOOP )))\"
echo \"  EVAL_SIZE         = ${EVAL_SIZE}\"
echo \"  SAMPLE_BATCH_SIZE = ${SAMPLE_BATCH_SIZE} x ${SAMPLE_NUM_BATCHES} batches\"
echo \"  FT_BATCH_SIZE     = ${FT_BATCH_SIZE}  (topk=${TOPK_RATIO})\"
echo \"  FT_TIMESTEPS      = ${FT_TIMESTEPS}\"
echo \"  FT_EPOCHS         = ${FT_EPOCHS}\"
echo \"  LR                = ${LR}\"
echo \"  SIGMA (KL)        = ${SIGMA}\"
echo \"  GRPO mode         = ${GRPO_MODE}\"
echo \"  GRPO adv_clip     = ${GRPO_ADV_CLIP}\"
echo \"  REPLAY            = ${REPLAY} (buf=${BUFFER_SIZE}, sample=${SAMPLE_SIZE}, cutoff=${REWARD_CUTOFF})\"
echo \"  DIV_FILTER        = ${DIV_FILTER} (tol=${DF_TOL}, buff=${DF_BUFF})\"
echo '==================================================='

cd \"${WORK_DIR}\"
source .venv/bin/activate
export MPLCONFIGDIR=\"${WORK_DIR}/.mplconfig\"
mkdir -p \"\${MPLCONFIGDIR}\"

export WANDB_MODE=offline
export WANDB_DIR=\"\${OUT_DIR}\"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
  +model.model_path=${RESUME_CKPT_DIR} \
  reward=${REWARD_NAME} \
  reward.mattersim_potential_path=${MATTERSIM_POTENTIAL_PATH} \
  reward.mattersim_reference_dataset_path=${REFERENCE_DATASET_PATH} \
  logger=${LOGGER_NAME} \
  rl_epoch=${RL_EPOCH} \
  eval_size=${EVAL_SIZE} \
  \
  model.sample_cfg.sampling_config_path=${SAMPLING_CONFIG_PATH} \
  model.sample_cfg.batch_size=${SAMPLE_BATCH_SIZE} \
  model.sample_cfg.num_batches=${SAMPLE_NUM_BATCHES} \
  model.finetune_cfg.batch_size=${FT_BATCH_SIZE} \
  model.finetune_cfg.timesteps=${FT_TIMESTEPS} \
  model.finetune_cfg.lr=${LR} \
  \
  sample_cfg.num_batches=${SAMPLE_NUM_BATCHES} \
  sample_cfg.max_num=${EVAL_SIZE} \
  ~sample_cfg.filter \
  \
  pipeline.topk_ratio=${TOPK_RATIO} \
  pipeline.save_freq=${SAVE_FREQ} \
  pipeline.start_loop=${START_LOOP} \
  pipeline.finetune_cfg.batch_size=${FT_BATCH_SIZE} \
  pipeline.finetune_cfg.accum_steps=${ACCUM_STEPS} \
  pipeline.finetune_cfg.epochs=${FT_EPOCHS} \
  pipeline.finetune_cfg.sigma=${SIGMA} \
  \
  pipeline.replay=${REPLAY} \
  pipeline.replay_args.buffer_size=${BUFFER_SIZE} \
  pipeline.replay_args.sample_size=${SAMPLE_SIZE} \
  pipeline.replay_args.reward_cutoff=${REWARD_CUTOFF} \
  \
  pipeline.div_filter=${DIV_FILTER} \
  pipeline.df_args.tol=${DF_TOL} \
  pipeline.df_args.buff=${DF_BUFF} \
  \
  pipeline.grpo.mode=${GRPO_MODE} \
  pipeline.grpo.adv_eps=${GRPO_ADV_EPS} \
  pipeline.grpo.adv_clip=${GRPO_ADV_CLIP} \
  pipeline.grpo.kl_weight_mode=${GRPO_KL_WEIGHT_MODE}

echo '>>> GRPO-lite resume training finished successfully'
echo \">>> Outputs saved to: \${OUT_DIR}\"
echo \">>> To sync wandb offline runs to cloud:\"
echo \"      wandb sync \${OUT_DIR}/wandb/offline-run-*/\"
"
