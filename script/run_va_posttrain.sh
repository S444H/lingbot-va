#!/usr/bin/bash

set -euo pipefail

umask 007
 
NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"}  # robotwin_train, libero_train, default is robotwin_train
SAVE_ROOT=${SAVE_ROOT:-"./output"}  # self set the save root path, default is ./output
WANDB_DIR=${WANDB_DIR:-"./wandb"}  # self set the wandb dir path, default is ./wandb
WAN22_PRETRAINED_MODEL_NAME_OR_PATH=${WAN22_PRETRAINED_MODEL_NAME_OR_PATH:-"./models/pretrained/lingbot-va-base"}
DATASET_PATH=${DATASET_PATH:-"./datasets/libero-long-lerobot"}


## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}
save_root=${SAVE_ROOT}
wandb_dir=${WANDB_DIR}
wan22_pretrained_model_name_or_path=${WAN22_PRETRAINED_MODEL_NAME_OR_PATH}
dataset_path=${DATASET_PATH}

## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train \
    --config-name ${config_name} \
    --save-root ${save_root} \
    --wandb-dir ${wandb_dir} \
    --wan22-pretrained-model-name-or-path "${wan22_pretrained_model_name_or_path}" \
    --dataset-path "${dataset_path}"
