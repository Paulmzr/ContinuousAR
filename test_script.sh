EXP_NAME=pred_noise.normalize.cosine.sigma_fixed_small.12+12_layers.bsz_128
CHECKPOINT=150000

python scripts/run_generation.py \
    --model_name_or_path /data/mazhengrui/SpeechLLaMA/experiments/${EXP_NAME}/checkpoint-${CHECKPOINT} \
    --output ${EXP_NAME}