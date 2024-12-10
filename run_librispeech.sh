EXP_NAME=librispeech.score.pho.normalize.ffn_2752.12+3_layers.dropout_0.1.act_dropout_0.1.bsz_128
CHECKPOINT=120000

python scripts/run_librispeech_pho.py \
    --model_name_or_path /data/mazhengrui/SpeechLLaMA/experiments/${EXP_NAME}/checkpoint-${CHECKPOINT} \
    --output ${EXP_NAME} --ckpt ${CHECKPOINT} \
    --cfg 1.5 --input "BUT IN THIS FRIENDLY PRESSURE RAOUL COULD DETECT THE NERVOUS AGITATION OF A GREAT INTERNAL CONFLICT"