EXP_NAME=librispeech.score.pho.normalize.ffn_2752.12+3_layers.dropout_0.1.act_dropout_0.1.bsz_128
CHECKPOINT=120000
BSZ=4
CFG=1.5

python scripts/eval_librispeech_pho_continue.py \
    --model_name_or_path /data/mazhengrui/SpeechLLaMA/experiments/${EXP_NAME}/checkpoint-${CHECKPOINT} \
    --output ${EXP_NAME} --ckpt ${CHECKPOINT} \
    --batch_size ${BSZ} --cfg ${CFG} --seed 0

EXP_NAME=librispeech.score.pho.normalize.ffn_2752.12+3_layers.dropout_0.1.act_dropout_0.1.bsz_128
CHECKPOINT=120000
BSZ=4
CFG=1.0

python scripts/eval_librispeech_pho_continue.py \
    --model_name_or_path /data/mazhengrui/SpeechLLaMA/experiments/${EXP_NAME}/checkpoint-${CHECKPOINT} \
    --output ${EXP_NAME} --ckpt ${CHECKPOINT} \
    --batch_size ${BSZ} --cfg ${CFG} --seed 0