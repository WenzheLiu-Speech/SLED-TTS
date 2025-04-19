import argparse
import logging
from typing import Tuple
from pathlib import Path

import pdb
import torch
import torchaudio
from accelerate.utils import set_seed

from transformers import AutoTokenizer, PreTrainedTokenizerFast, RobertaTokenizer, AutoProcessor
from sled.sled import SpeechLlamaForCausalLM


BANDWIDTH=6
STRIDE=320
SAMPLING_RATE=24000

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

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument("--prompt_audio", type=str)
    parser.add_argument("--prompt_text", type=str)
    parser.add_argument("--input", type=str, default="It was silent and gloomy, being tenanted solely by the captive and lighted by the dying embers of a fire which had been used for the purposes of cookery.")
    parser.add_argument("--max_length", type=int, default=0)

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
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

    prompt_text = args.prompt_text if args.prompt_text else input("Prompt Text >>> ")    
    prompt_audio = args.prompt_audio
    waveform, sample_rate = torchaudio.load(prompt_audio, normalize=True)
    if sample_rate != SAMPLING_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLING_RATE)
        waveform = resampler(waveform).squeeze().numpy()

    
    input_text = args.input if args.input else input("Model input >>> ")    
    #input_ids = tokenizer.encode(input_text + " " + prompt_text, return_tensors='pt').to(device)
    input_text = [prompt_text + " " + input_text]
            
    batch_encoded = tokenizer.batch_encode_plus(
        input_text,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = batch_encoded["input_ids"].to(device)
    attention_mask = batch_encoded["attention_mask"].to(device)
    text_input_length = input_ids.shape[1]


    audio_arrays = [waveform]
    audio_inputs = processor(raw_audio=audio_arrays, sampling_rate=SAMPLING_RATE, return_tensors="pt") # 'padding_mask': b,t  'input_values': b,c,t
            
            
    encoder_outputs = model.codec.encode(audio_inputs["input_values"].to(model.dtype).to(device), audio_inputs["padding_mask"].to(device), bandwidth=BANDWIDTH) #1,b,r,t, 1 due to one chunk
    speech_inputs_embeds = model.codec.quantizer.decode(encoder_outputs.audio_codes[0].transpose(0, 1)) #b,d,t
    
    speech_attention_mask = audio_inputs["padding_mask"][..., ::STRIDE].to(device)
    assert speech_inputs_embeds.size(-1) == speech_attention_mask.size(-1)
    speech_inputs_embeds = speech_inputs_embeds.transpose(1,2) #b,t,d
    
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
    new_audio_values = model.codec.decoder(new_embeds.transpose(-1,-2))

    
    output_path = "output.wav"    
    torchaudio.save(output_path, new_audio_values[0][:, speech_input_length* STRIDE:].cpu(), SAMPLING_RATE)

    return


if __name__ == "__main__":
    main()
