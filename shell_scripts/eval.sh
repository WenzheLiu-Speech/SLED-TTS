
CHECKPOINT=/path/to/checkpoint
BSZ=1
CFG=2.0

python ./scripts/eval_stream.py \
    --model_name_or_path ${CHECKPOINT} \
    --batch_size ${BSZ} --cfg ${CFG} --seed 0
