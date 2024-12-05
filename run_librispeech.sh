EXP_NAME=librispeech.score.pho.normalize.ffn_2752.12+12_layers.dropout_0.1.act_dropout_0.1.bsz_128
CHECKPOINT=80000

python scripts/run_librispeech_pho.py \
    --model_name_or_path /data/mazhengrui/SpeechLLaMA/experiments/${EXP_NAME}/checkpoint-${CHECKPOINT} \
    --output ${EXP_NAME} --ckpt ${CHECKPOINT} \
    --cfg 1.5 --input "And yesterday things went on just as usual" --seed 18