# 🛷SLED-TTS: Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space
> **Authors: [Zhengrui Ma](https://scholar.google.com/citations?user=dUgq6tEAAAAJ), [Yang Feng*](https://people.ucas.edu.cn/~yangfeng?language=en), [Chenze Shao](https://scholar.google.com/citations?user=LH_rZf8AAAAJ&hl), [Fandong Meng](https://fandongmeng.github.io/), [Jie Zhou](https://scholar.google.com.hk/citations?user=OijxQCMAAAAJ&hl=en), [Min Zhang](https://scholar.google.com/citations?user=CncXH-YAAAAJ&hl=en)**

[![HuggingFace](https://img.shields.io/badge/HuggingFace-FEC200?style=flat&logo=Hugging%20Face)](https://huggingface.co/collections/ICTNLP/sled-tts-680253e19c889010a1a376ac)
[![WeChat AI](https://img.shields.io/badge/WeChat%20AI-4CAF50?style=flat&logo=wechat)](https://www.wechat.com)
[![ICT/CAS](https://img.shields.io/badge/ICT%2FCAS-0066cc?style=flat&logo=school)](https://ict.cas.cn)


## Key features
- **Continuous Autoregressive Modeling**: SLED models speech in a continuous latent space, eliminating the need for complex hierarchical architectures.
- **Streaming Synthesis**: SLED supports streaming synthesis, enabling speech generation to start as soon as the text stream begins.
- **Voice Cloning**: Capable of generating speech based on a 3-second prefix or reference utterance as prompt.


## Demo
You can check SLED in action by exploring the [demo page](https://sled-demo.github.io/).
<div style="display: flex;">
   <img src="https://github.com/user-attachments/assets/0f6ee8a0-4258-48a2-a670-5556672dbc18" width="200" style="margin-right: 20px;"/>
   <img src="https://github.com/user-attachments/assets/f48848b0-58d9-403a-86d1-80683565a4d7" width="500"/>
</div>

## Available Models on Hugging Face

We are currently offering two English models trained on LibriHeavy on [Hugging Face](https://huggingface.co/collections/ICTNLP/sled-tts-680253e19c889010a1a376ac):

1. **[SLED-TTS-Libriheavy](https://huggingface.co/ICTNLP/SLED-TTS-Libriheavy)**: This model is trained on Libriheavy and provides high-quality text-to-speech synthesis.
  
2. **[SLED-TTS-Streaming-Libriheavy](https://huggingface.co/ICTNLP/SLED-TTS-Streaming-Libriheavy)**: This variant supports **streaming decoding**, which generates a 0.6-second speech chunk for every 5 text tokens received.

**Alternatively, you can train SLED on your own data by following the guidelines below.**

## Usage
**We provide the training and inference code for SLED-TTS.**

### Installation
``` sh
git clone https://github.com/ictnlp/SLED-TTS.git
cd SLED-TTS
pip install -e ./
```

We currently utilize the sum of the first 8 embedding vectors from [Encodec_24khz](https://huggingface.co/facebook/encodec_24khz) as the continuous latent vector. To proceed, ensure that [Encodec_24khz](https://huggingface.co/facebook/encodec_24khz) is downloaded and cached in your HuggingFace dir.

### Inference
- Set the `CHECKPOINT` variable to the path of the cached **[SLED-TTS-Libriheavy](https://huggingface.co/ICTNLP/SLED-TTS-Libriheavy)** or **[SLED-TTS-Streaming-Libriheavy](https://huggingface.co/ICTNLP/SLED-TTS-Streaming-Libriheavy)** model.
- Diverse generation results can be obtained by varying the `SEED` variable.
- Use `-bf16` flag to enable bf16 inference.
``` sh
CHECKPOINT=/path/to/checkpoint
CFG=2.0
SEED=0
```
***Offline Inference***
``` sh
python scripts/run_offline.py \
    --model_name_or_path ${CHECKPOINT} \
    --cfg ${CFG} \
    --input "My remark pleases him, but I soon prove to him that it is not the right way to speak. However perfect may have been the language of that ancient writer." \
    --seed ${SEED}
```
***Streaming Inference***
``` sh
python scripts/run_stream.py \
    --model_name_or_path ${CHECKPOINT} \
    --cfg ${CFG} \
    --input "My remark pleases him, but I soon prove to him that it is not the right way to speak. However perfect may have been the language of that ancient writer." \
    --seed ${SEED}
# Please note that we have simulated the generation in a streaming environment in run_stream.py for evaluating its quality.
# However, the existing code does not actually provide a streaming API.
```
***Voice Clone***

You can adjust the prompt speech by setting `--prompt_text` and `--prompt_audio`.
``` sh
python scripts/run_voice_clone.py \
    --prompt_text "Were I in the warm room with all the splendor and magnificence!" \
    --prompt_audio "example_prompt.flac" \
    --model_name_or_path ${CHECKPOINT} \
    --cfg ${CFG} \
    --input "Perhaps the other trees from the forest will come to look at me!" \
    --seed ${SEED}
```

### Training

***Data Processing***

Process the LibriHeavy data so that each line follows the JSON format shown below.
```
{"id": "large/10022/essayoncriticism_1505_librivox_64kb_mp3/essayoncriticism_01_pope_64kb_5", "start": 610.32, "duration": 19.76, "supervisions": [{"text": "Hail! bards triumphant! born in happier days; Immortal heirs of universal praise! Whose honors with increase of ages grow, As streams roll down, enlarging as they flow; Nations unborn your mighty names shall sound, [193] And worlds applaud that must not yet be found!"}], "recording": {"sources": [{"source": "download/librilight/large/10022/essayoncriticism_1505_librivox_64kb_mp3/essayoncriticism_01_pope_64kb.flac"}], "sampling_rate": 16000}, "type": "MonoCut"}
```
Or you can use the manifest of LibriHeavy available at this [URL](https://huggingface.co/datasets/ICTNLP/LibriHeavy_manifest). For your own datasets, process them into a similar format.

***Training Offline Model***
``` sh
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
    --ddp_timeout 3600 --overwrite_output_dir

```

***Training Streaming Model***
``` sh
OUTPUT_DIR=./runs/libriheavy_stream
mkdir -p $OUTPUT_DIR
LOG_FILE=${OUTPUT_DIR}/log

BATCH_SIZE=8
UPDATE_FREQ=8
# assume 8 proc per node, then WORLD_SIZE * 8 * BATCH_SIZE * UPDATE_FREQ == 512

torchrun --nnodes ${WORLD_SIZE} --node_rank ${RANK} --nproc_per_node 8 --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
    ./scripts/train_libriheavy_stream.py \
    --finetune_path ./runs/libriheavy/checkpoint-300000/model.safetensors \
    --stream_n 5 --stream_m 45 \
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
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --max_steps 100000 \
    --lr_scheduler_type "linear" \
    --warmup_steps 10000 \
    --logging_first_step \
    --logging_steps 100 \
    --save_steps 10000 \
    --save_total_limit 10 \
    --output_dir ${OUTPUT_DIR} \
    --report_to tensorboard \
    --disable_tqdm True \
    --ddp_timeout 3600 --overwrite_output_dir
```
### BF16 Support
By setting the `-bf16` flag, the model will load in bf16 during inference and in fp32 during training (for mixed precision training). To enable pure bf16 training, you can change
https://github.com/ictnlp/SLED-TTS/blob/69a0a77d37180ec711a21f39f1b6bffa8b068072/scripts/train_libriheavy.py#L298
to 
```
torch_dtype = torch.bfloat16 if training_args.bf16 else None
```
However, Encodec should always execute in fp32 to maintain the precision of latents. Therefore, we load Encodec in fp32 and downcast the encoded latent to bf16.


## Citation
If you have any questions, please feel free to submit an issue or contact `mazhengrui21b@ict.ac.cn`.

If our work is useful for you, please cite as:

```
@article{ma-etal-2025-sled-tts,
  title={Efficient Speech Language Modeling via Energy Distance in Continuous Space},
  author={Zhengrui Ma, Yang Feng, Chenze Shao, Fandong Meng, Jie Zhou, Min zhang},
  year={2025}
}
```
