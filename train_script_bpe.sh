#export OMP_NUM_THREADS=20
EXP_NAME=bpe.pred_noise.normalize.linear.sigma_fixed_small.12+12_layers.bsz_64

OUTPUT_DIR=./experiments/${EXP_NAME}
mkdir -p $OUTPUT_DIR
LOG_FILE=./experiments/${EXP_NAME}/log

torchrun --nproc_per_node 2 --nnodes 1 scripts/train_bpe.py \
    --num_hidden_layers 12 --diffloss_d 12 \
    --learn_sigma False --sigma_small True \
    --eval_split "dev.clean" \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --remove_unused_columns False \
    --label_names audio_inputs \
    --group_by_speech_length \
    --do_train \
    --do_eval \
    --eval_on_start \
    --eval_strategy steps \
    --eval_steps 5000 \
    --prediction_loss_only \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --bf16 \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --max_steps 150000 \
    --lr_scheduler_type "linear" \
    --warmup_steps 10000 \
    --logging_first_step \
    --logging_steps 100 \
    --save_steps 5000 \
    --save_total_limit 20 \
    --output_dir ${OUTPUT_DIR} \
    --report_to tensorboard \
    --disable_tqdm True \
    --ddp_timeout 3600 --overwrite_output_dir \
    2>&1 |tee -a ${LOG_FILE}
    
    

#2>&1 |tee ${LOG_FILE}
#--max_train_samples 100 \
#--max_eval_samples 100 \
#--overwrite_output_dir \