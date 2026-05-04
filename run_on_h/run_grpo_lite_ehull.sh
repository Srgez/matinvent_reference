#!/usr/bin/env bash
# ============================================================
# Full GRPO-lite training run — ehull reward
# ============================================================
# Design rationale (single-process, 8 GPU allocated for VRAM):
#
#  Sampling
#    SAMPLE_BATCH_SIZE=32, NUM_BATCHES=3  → ≤ 96 candidates generated,
#    top EVAL_SIZE=64 passed to scoring.  More per-step samples give a
#    more reliable reward baseline for advantage normalisation.
#
#  Top-k & advantage
#    TOPK_RATIO=0.5  → top-32 train samples.
#    Advantage computed over all 64 success samples as baseline.
#    GRPO_ADV_CLIP=3.0  → tighter than default (5.0) for early stability.
#
#  Replay
#    BUFFER_SIZE=300, SAMPLE_SIZE=16, REWARD_CUTOFF=0.65
#    Replay keeps history diverse; reward_cutoff keeps buffer quality high.
#    In GRPO-lite, replay samples are renormalised with the current-step
#    baseline, which is an acceptable approximation for off-policy samples.
#
#  Finetune
#    FT_BATCH_SIZE=32  → selects top-32 per step (32×0.5=16 + 16 replay ≈ 32 total).
#    FT_TIMESTEPS=300  → wider gradient coverage across the diffusion trajectory
#                        than the smoke-run value of 50.
#    FT_EPOCHS=3       → consistent with default config.
#    ACCUM_STEPS=50    → one optimizer step every 50 random-t iterations.
#    LR=5e-6           → slightly conservative vs. default 1e-5;
#                        larger training batch warrants smaller lr.
#    SIGMA=0.025       → KL weight (fixed mode, decoupled from advantage).
#
#  Diversity filter
#    tol=3, buff=6  — default, suitable for low-novelty early training.
#
#  RL epochs
#    120 (≈120 × ~6 min = ~12 h on 1 GPU; adjust RL_EPOCH as needed).
#    Checkpoint saved every 10 steps.
# ============================================================
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Job configuration
# ──────────────────────────────────────────────────────────────
JOB_NAME="${JOB_NAME:-matinvent-lite-ehull2}"
GPU_NUM="${GPU_NUM:-1}"
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
# MATTERGEN_MODEL_PATH="${MATTERGEN_MODEL_PATH:-${MATTERGEN_CHECKPOINT_ROOT}/mattergen_base}"
MATTERGEN_MODEL_PATH="${MATTERGEN_MODEL_PATH:-${MATTERGEN_CHECKPOINT_ROOT}/my_train_mattergen_base}"

MATTERSIM_POTENTIAL_PATH="${MATTERSIM_POTENTIAL_PATH:-${MATTERGEN_CHECKPOINT_ROOT}/mattersim/mattersim-v1.0.0-1M.pth}"
REFERENCE_DATASET_PATH="${REFERENCE_DATASET_PATH:-/mnt/shared-storage-user/zhangsizhe/mattergen/data-release/alex-mp/reference_MP2020correction.gz}"
GEMNET_SCALE_FILE="${GEMNET_SCALE_FILE:-/mnt/shared-storage-user/zhangsizhe/mattergen_steering_2/mattergen/common/gemnet/gemnet-dT.json}"
SAMPLING_CONFIG_PATH="${SAMPLING_CONFIG_PATH:-/mnt/shared-storage-user/zhangsizhe/mattergen_steering_2/sampling_conf}"

# ──────────────────────────────────────────────────────────────
# Experiment identity
# ──────────────────────────────────────────────────────────────
EXP_NAME="${EXP_NAME:-grpo_lite_ehull_v1_my}"
RESULTS_DIR="${RESULTS_DIR:-exp_res_all}"
LOGGER_NAME="${LOGGER_NAME:-wandb}"        # offline wandb; sync later with: wandb sync <wandb_dir>
MATTERGEN_MODEL_NAME="${MATTERGEN_MODEL_NAME:-my_train_mattergen_base}"

# ──────────────────────────────────────────────────────────────
# RL loop
# ──────────────────────────────────────────────────────────────
RL_EPOCH="${RL_EPOCH:-120}"

# ──────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────
EVAL_SIZE="${EVAL_SIZE:-64}"                  # samples scored per step
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-32}"  # MatterGen generation batch size
SAMPLE_NUM_BATCHES="${SAMPLE_NUM_BATCHES:-3}" # batches × batch_size ≥ eval_size
                                              # → 96 candidates, filter & cap at 64

# ──────────────────────────────────────────────────────────────
# Finetune
# ──────────────────────────────────────────────────────────────
FT_BATCH_SIZE="${FT_BATCH_SIZE:-32}"    # controls topk count via topk_ratio
FT_TIMESTEPS="${FT_TIMESTEPS:-300}"     # random diffusion steps per epoch
FT_EPOCHS="${FT_EPOCHS:-3}"
ACCUM_STEPS="${ACCUM_STEPS:-50}"
LR="${LR:-5e-6}"
SIGMA="${SIGMA:-0.025}"                 # KL loss weight (fixed mode)

TOPK_RATIO="${TOPK_RATIO:-0.5}"        # int(32×0.5)=16 top-k per step
SAVE_FREQ="${SAVE_FREQ:-10}"            # save model every N steps

# ──────────────────────────────────────────────────────────────
# Experience replay
# ──────────────────────────────────────────────────────────────
REPLAY="${REPLAY:-True}"
BUFFER_SIZE="${BUFFER_SIZE:-300}"
SAMPLE_SIZE="${SAMPLE_SIZE:-16}"        # samples drawn from buffer each step
REWARD_CUTOFF="${REWARD_CUTOFF:-0.65}"  # minimum reward to enter buffer

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
GRPO_ADV_CLIP="${GRPO_ADV_CLIP:-3.0}"  # tighter clip for training stability
GRPO_KL_WEIGHT_MODE="${GRPO_KL_WEIGHT_MODE:-fixed}"

# ──────────────────────────────────────────────────────────────
# Reward (ehull)
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

# ── Pre-create output directory so we can write logs/script there ──────────
OUT_DIR=\"${WORK_DIR}/${RESULTS_DIR}/${EXP_NAME}\"
mkdir -p \"\${OUT_DIR}\"

# Copy this launch script into the output dir for full reproducibility
cp \"${WORK_DIR}/run_on_h/run_grpo_lite_ehull.sh\" \"\${OUT_DIR}/launch.sh\"

# All subsequent output (stdout + stderr) goes to console AND train.log
LOG_FILE=\"\${OUT_DIR}/train.log\"
exec > >(tee -a \"\${LOG_FILE}\") 2>&1

echo '==================================================='
echo 'GRPO-lite full training — ehull reward'
echo '==================================================='
echo \"  EXP_NAME          = ${EXP_NAME}\"
echo \"  OUT_DIR           = \${OUT_DIR}\"
echo \"  RL_EPOCH          = ${RL_EPOCH}\"
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

# ── wandb offline mode ──────────────────────────────────────────────────────
export WANDB_MODE=offline
export WANDB_DIR=\"\${OUT_DIR}\"

# ── Block all HuggingFace Hub network calls ──────────────────────────────────
# OptFilter (and other tools) fall back to hf_hub_download when local paths
# are not passed.  Setting HF_HUB_OFFLINE=1 makes those calls raise immediately
# with a clear error instead of hanging for minutes.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Resolve GemNet scale-file symlink at runtime
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
  sample_cfg.batch_size=${SAMPLE_BATCH_SIZE} \
  sample_cfg.max_num=${EVAL_SIZE} \
  ~sample_cfg.filter \
  \
  pipeline.topk_ratio=${TOPK_RATIO} \
  pipeline.save_freq=${SAVE_FREQ} \
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

echo '>>> GRPO-lite training finished successfully'
echo \">>> Outputs saved to: \${OUT_DIR}\"
echo \">>> To sync wandb offline run to cloud:\"
echo \"      wandb sync \${OUT_DIR}/wandb/offline-run-*/\"
"
