import os
import argparse
import logging
from typing import Tuple
from pathlib import Path

import torch
import torchaudio


from datasets import load_dataset, Audio
from transformers import RobertaTokenizer, AutoProcessor
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from accelerate.utils import set_seed

import pdb


from sled.sled import SpeechLlamaForCausalLM

BANDWIDTH=6
SAMPLING_RATE=24000
STRIDE=320
FREQ=75
MIN_LEN=4.0
MAX_LEN=10.0
PROMPT_LEN=3



logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)



def adjust_length_to_model(length, max_sequence_length):
    assert max_sequence_length > 0
    if length <= 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    return length


def filter_function(example):
    return ((len(example["audio"]["array"]) / SAMPLING_RATE) < MAX_LEN) and ((len(example["audio"]["array"]) / SAMPLING_RATE) > MIN_LEN)



def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument(
        "--cfg",
        type=float,
        default=1.0,
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    logger.warning(f"device: {device}, 16-bits inference: {args.fp16 or args.bf16}")

    if args.seed is not None:
        set_seed(args.seed)

    # Initialize the model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    eos_token_id = tokenizer.eos_token_id

    model = SpeechLlamaForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch_dtype)
    model.infer_cfg = args.cfg
    model.initialize_codec("facebook/encodec_24khz") 
    model.to(device)


    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    
    assert tokenizer.pad_token is not None
    logger.info(f"tokenizer pad token: {tokenizer.pad_token}")


    max_seq_length = getattr(model.config, "max_position_embeddings", 0)
    args.max_length = adjust_length_to_model(args.max_length, max_sequence_length=max_seq_length)
    logger.info(args)

    
    eval_dataset = load_dataset("yoom618/librispeech_pc", split="test.clean")
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    
    
    logger.info(f"original eval dataset: {len(eval_dataset)} samples.")
    eval_dataset = eval_dataset.filter(filter_function)
    logger.info(f"filtered eval dataset: {len(eval_dataset)} samples.")
    
    tokenized_eval_dataset = eval_dataset.map(
        lambda example: tokenizer(example["text"]),
        batched=True,
    )
    

    batch_size = args.batch_size
    output_path = Path("eval_continue")
    output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad(): 
        for i in range(0, len(tokenized_eval_dataset), batch_size):
            batch = tokenized_eval_dataset.select(range(i, min(i + batch_size, len(tokenized_eval_dataset))))
                    
            input_ids = [{"input_ids":instance["input_ids"]} for instance in batch]
            
            encodes = pad_without_fast_tokenizer_warning(
                tokenizer,
                input_ids,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
            input_ids = encodes["input_ids"].to(device)
            attention_mask = encodes["attention_mask"].to(device)       
            text_input_length = input_ids.shape[1]

            audio_arrays = [instance["audio"]["array"] for instance in batch]
            audio_inputs = processor(raw_audio=audio_arrays, sampling_rate=SAMPLING_RATE, return_tensors="pt") # 'padding_mask': b,t  'input_values': b,c,t
            
            
            encoder_outputs = model.codec.encode(audio_inputs["input_values"].to(model.dtype).to(device), audio_inputs["padding_mask"].to(device), bandwidth=BANDWIDTH) #1,b,r,t, 1 due to one chunk
            speech_inputs_embeds = model.codec.quantizer.decode(encoder_outputs.audio_codes[0].transpose(0, 1)) #b,d,t
            
            speech_attention_mask = audio_inputs["padding_mask"][..., ::STRIDE].to(device)
            assert speech_inputs_embeds.size(-1) == speech_attention_mask.size(-1)
            speech_inputs_embeds = speech_inputs_embeds.transpose(1,2) #b,t,d
            
            
            speech_inputs_embeds = speech_inputs_embeds[:,:FREQ * PROMPT_LEN,:]
            speech_attention_mask = speech_attention_mask[:,:FREQ * PROMPT_LEN]
            speech_input_length = speech_inputs_embeds.shape[1]
            

            new_attention_mask = torch.concat([attention_mask, speech_attention_mask], dim=1)
            
            
            output_sequences = model.generate(
                input_ids=input_ids,
                inputs_embeds=speech_inputs_embeds,
                attention_mask=new_attention_mask,
                max_length=args.max_length,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
            )
            

            
            new_embeds = output_sequences[1]
            generated_ids = output_sequences[0][:, text_input_length:]

            new_audio_values = model.codec.decoder(new_embeds.transpose(-1,-2))

            wav_len = (generated_ids.ne(eos_token_id).sum(dim=-1) + speech_input_length) * STRIDE
            
            for i in range(len(wav_len)):
                id = batch["id"][i]
                torchaudio.save(output_path / f"{id}.wav", new_audio_values[i][:,:wav_len[i]].cpu(), SAMPLING_RATE)
        
    return


if __name__ == "__main__":
    main()
