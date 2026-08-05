#!/usr/bin/bash

set -euo pipefail

umask 007
 
NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin"}
SAVE_ROOT=${SAVE_ROOT:-"./inference_output"}  # self set the save root path, default is ./output
WAN22_PRETRAINED_MODEL_NAME_OR_PATH=${WAN22_PRETRAINED_MODEL_NAME_OR_PATH:-"./models/pretrained/lingbot-va-base"}


## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}
save_root=${SAVE_ROOT}
wan22_pretrained_model_name_or_path=${WAN22_PRETRAINED_MODEL_NAME_OR_PATH}


## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.wan_va_server \
    --config-name ${config_name} \
    --save-root ${save_root} \
    --wan22-pretrained-model-name-or-path "${wan22_pretrained_model_name_or_path}"

