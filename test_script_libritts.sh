EXP_NAME=score.bpe.normalize.tiny.12+12_layers.bsz_64
CHECKPOINT=average
BSZ=16
CFG=1.5

python scripts/eval_bpe_libritts.py \
    --model_name_or_path /data/mazhengrui/SpeechLLaMA/experiments/${EXP_NAME}/checkpoint-${CHECKPOINT} \
    --output ${EXP_NAME} --ckpt ${CHECKPOINT} \
    --batch_size ${BSZ} --cfg ${CFG} --seed 42