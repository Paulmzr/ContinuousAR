EXP_NAME=score.bpe.normalize.tiny.12+12_layers.bsz_64
CHECKPOINT=200000

python scripts/run_libritts_bpe.py \
    --model_name_or_path /data/mazhengrui/SpeechLLaMA/experiments/${EXP_NAME}/checkpoint-${CHECKPOINT} \
    --output ${EXP_NAME} --ckpt ${CHECKPOINT} \
    --cfg 2.0 --input "Happily the recent phenomena had no effect upon the compass; the magnetic needle, which in these regions had pointed about twenty two degrees from the north pole, had never deviated in the least-a proof that, although east and west had apparently changed places, north and south continued to retain their normal position as cardinal points."