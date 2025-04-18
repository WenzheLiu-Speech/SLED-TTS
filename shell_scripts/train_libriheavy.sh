OUTPUT_DIR=./runs/libriheavy
mkdir -p $OUTPUT_DIR
LOG_FILE=${OUTPUT_DIR}/log

BATCH_SIZE=8
UPDATE_FREQ=8
# assume 8 proc per node, then WORLD_SIZE * 8 * BATCH_SIZE * UPDATE_FREQ == 512

torchrun --nnodes ${WORLD_SIZE} --node_rank ${RANK} --nproc_per_node 8 --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
    ./scripts/train_libriheavy.py \
    --training_cfg 0.1 \
    --num_hidden_layers 12 --diffloss_d 6 --noise_channels 128 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --remove_unused_columns False \
    --label_names audio_inputs \
    --group_by_speech_length \
    --do_train \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 10000 \
    --prediction_loss_only \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 24 \
    --gradient_accumulation_steps ${UPDATE_FREQ} \
    --bf16 \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --max_steps 300000 \
    --lr_scheduler_type "linear" \
    --warmup_steps 32000 \
    --logging_first_step \
    --logging_steps 100 \
    --save_steps 10000 \
    --save_total_limit 10 \
    --output_dir ${OUTPUT_DIR} \
    --report_to tensorboard \
    --disable_tqdm True \
    --ddp_timeout 3600 --overwrite_output_dir \
    2>&1 |tee -a ${LOG_FILE}
