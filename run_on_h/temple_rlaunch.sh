#!/bin/bash
set -ex

# ========== Job Configuration ==========
JOB_NAME=matinvent
GPU_NUM=1
MEMORY=200000
CPU_NUM=16
GROUP_NAME=ai4sdata_gpu
MOUNT_PATH=gpfs://gpfs1/zhangsizhe:/mnt/shared-storage-user/zhangsizhe
IMAGE_NAME=registry.h.pjlab.org.cn/ailab-ai4sdata/zhangsizhe-workspace:20251221230837

# ========== Paths ==========
WORK_DIR=/mnt/shared-storage-user/zhangsizhe/matinvent
# INPUT_PATH=/mnt/shared-storage-user/zhangsizhe/mineru/input
# OUTPUT_PATH=/mnt/shared-storage-user/zhangsizhe/mineru/output

# ========== Submit Job ==========
rjob submit \
  --name=${JOB_NAME} \
  --gpu=${GPU_NUM} \
  --memory=${MEMORY} \
  --cpu=${CPU_NUM} \
  --charged-group=${GROUP_NAME} \
  --private-machine=group \
  --mount=${MOUNT_PATH} \
  --image=${IMAGE_NAME} \
  -e DISTRIBUTED_JOB=true \
  -- bash -c "
set -euo pipefail
echo '>>> Using shell options: -euo pipefail'
printf '>>> Job started on: %s\n' \"\$(hostname)\"
echo '>>> GPU allocation: ${GPU_NUM}'
echo '>>> Memory: ${MEMORY} MB'
echo '>>> CPUs: ${CPU_NUM}'

cd ${WORK_DIR}
source .venv/bin/activate
python -u main.py \
    expname=test \
    pipeline=mat_invent \
    model=mattergen \
    reward=hhi \
    logger=wandb

echo '>>> Extracting completed successfully'
"
