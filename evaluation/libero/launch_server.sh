
save_root='visualization/'
mkdir -p $save_root

# e.g.
wan22_pretrained_model_name_or_path='/datacc05/shenhao/models/checkpoints/libero_train/checkpoints/checkpoint_step_5000'

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 29061 \
    wan_va/wan_va_server.py \
    --config-name libero \
    --port 29056 \
    --save-root $save_root \
    --wan22-pretrained-model-name-or-path $wan22_pretrained_model_name_or_path
