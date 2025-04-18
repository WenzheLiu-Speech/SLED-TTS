import argparse
import logging
from typing import Tuple
from pathlib import Path

import pdb
import torch
import torchaudio
from accelerate.utils import set_seed

from transformers import AutoTokenizer, PreTrainedTokenizerFast, RobertaTokenizer
from sled.sled import SpeechLlamaForCausalLM



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
    
    assert tokenizer.pad_token is not None
    logger.info(f"tokenizer pad token: {tokenizer.pad_token}")


    max_seq_length = getattr(model.config, "max_position_embeddings", 0)
    args.max_length = adjust_length_to_model(args.max_length, max_sequence_length=max_seq_length)
    logger.info(args)

    
    input_text = args.input if args.input else input("Model input >>> ")    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=args.max_length,
        do_sample=True,
        num_return_sequences=args.num_return_sequences,
    )
    
    
    new_embeds = output_sequences[1]
    new_audio_values = model.codec.decoder(new_embeds.transpose(-1,-2))

    
    output_path = "output.wav"    
    torchaudio.save(output_path, new_audio_values[0].cpu(), SAMPLING_RATE)

    return


if __name__ == "__main__":
    main()
