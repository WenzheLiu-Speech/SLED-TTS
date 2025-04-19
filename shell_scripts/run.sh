

CHECKPOINT=/path/to/checkpoint
CFG=2.0

# Offline Inference
python scripts/run_offline.py \
    --model_name_or_path ${CHECKPOINT} \
    --cfg ${CFG} \
    --input "My remark pleases him, but I soon prove to him that it is not the right way to speak. However perfect may have been the language of that ancient writer." \
    --seed 42

# Or Streaming Inference
python scripts/run_stream.py \
    --model_name_or_path ${CHECKPOINT} \
    --cfg ${CFG} \
    --input "My remark pleases him, but I soon prove to him that it is not the right way to speak. However perfect may have been the language of that ancient writer." \
    --seed 42

# Or Voice Clone
python scripts/run_voice_clone.py \
    --prompt_text "Were I in the warm room with all the splendor and magnificence!" \
    --prompt_audio "example_prompt.flac" \
    --model_name_or_path ${CHECKPOINT} \
    --cfg ${CFG} \
    --input "Perhaps the other trees from the forest will come to look at me!" \
    --seed 42
