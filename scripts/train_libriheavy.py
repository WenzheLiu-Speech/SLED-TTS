import os
import sys
import math
import json
import logging
import pathlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence, List, Union

import numpy as np
import torch
import datasets
import transformers

import soundfile as sf
import librosa



from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from transformers import AutoProcessor, AutoTokenizer, RobertaTokenizer


from sled.sled import SpeechLlamaConfig, SpeechLlamaForCausalLM
from sled.trainer_libriheavy import SpeechLlamaTrainer

logger = logging.getLogger(__name__)



SAMPLING_RATE=24000
SAMPLING_RATE_LIBRIHEAVY=16000
SAMPLING_RATE_TOKENIZER=75
    

@dataclass
class ArchArguments:
    # --------------------------------------------------------------------------
    # Llama Arguments
    hidden_size: int = 1024
    intermediate_size: int = 2752
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: int = 0
    eos_token_id: int = 2
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    attention_bias: bool = False
    attention_dropout: float = 0.1
    mlp_bias: bool = False
    vocab_size: int = 32000
    dropout: float = 0.1
    activation_dropout: float = 0.1
    
    # --------------------------------------------------------------------------
    # Score Arguments
    vae_embed_dim: int = 128
    diffloss_d: int = 3
    diffloss_w: int = 1024
    training_cfg: float = 0.0
    noise_channels: int = 128



@dataclass
class ModelArguments:
    # --------------------------------------------------------------------------
    # Codec & Tokenizer Arguments
    codec: str = "facebook/encodec_24khz"
    tokenizer: str = "/path/tokenizer_bpe_libriheavy"



@dataclass
class DataArguments:
    data_path: str = "/path/libriheavy"
    train_manifest: List[str] = field(default_factory=lambda: ["/path/libriheavy/cases_and_punc/libriheavy_cuts_large.jsonl", "/path/libriheavy/cases_and_punc/libriheavy_cuts_medium.jsonl", "/path/libriheavy/cases_and_punc/libriheavy_cuts_small.jsonl"])
    eval_manifest: List[str] = field(default_factory=lambda: ["/path/libriheavy/cases_and_punc/filtered2/libriheavy_cuts_dev.jsonl"])
    pad_to_multiple_of: Optional[int] = None
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    group_by_speech_length: bool = field(default=True)



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    processor: transformers.PreTrainedTokenizer
    data_path: str
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids":instance["input_ids"]} for instance in instances]
    
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            input_ids,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        audio_files = [instance["recording"]["sources"][0]["source"] for instance in instances]
        durations = [instance["duration"] for instance in instances]
        start_times = [instance["start"] for instance in instances]
        
        audio_arrays = [self.load_audio(file_path, start, duration) for file_path, start, duration in zip(audio_files, start_times, durations)]
        
        audio_inputs = self.processor(raw_audio=audio_arrays, sampling_rate=SAMPLING_RATE, return_tensors="pt") # 'padding_mask': b,t  'input_values': b,c,t
        
        batch["audio_inputs"] = audio_inputs
        
        return batch

    def load_audio(self, file_path: str, start: float, duration: float) -> np.array:
        abs_path = Path(self.data_path) / file_path
        audio, sampling_rate = sf.read(abs_path, start=int(start * SAMPLING_RATE_LIBRIHEAVY), stop=int((start + duration) * SAMPLING_RATE_LIBRIHEAVY))
        resampled_audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=SAMPLING_RATE)
        return resampled_audio


def load_manifest(file_paths):
    all_data = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            all_data.extend([json.loads(line) for line in f])
    return all_data


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                arch_args, model_args, data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = None
    eval_dataset = None
    
    if training_args.do_train:
        train_manifest = load_manifest(data_args.train_manifest)
        
        if data_args.max_train_samples is not None:
            train_manifest = train_manifest[:data_args.max_train_samples]

        train_dataset = train_manifest


    if training_args.do_eval:
        eval_manifest = load_manifest(data_args.eval_manifest)
        
        if data_args.max_eval_samples is not None:
            eval_manifest = eval_manifest[:data_args.max_eval_samples]
        
        eval_dataset = eval_manifest

    def tokenize_example(example):
        text = example["supervisions"][0]["text"]
        return tokenizer(text)
    
    tokenized_train_dataset = None
    tokenized_eval_dataset = None
    
    with training_args.main_process_first(desc="dataset map tokenization"):
        if training_args.do_train and train_dataset:
            tokenized_train_dataset = [
                {**example, **tokenize_example(example)} for example in train_dataset
            ]
            
        if training_args.do_eval and eval_dataset:
            tokenized_eval_dataset = [
                {**example, **tokenize_example(example)} for example in eval_dataset
            ]
        

    def filter_function(example): 
        file_path = example["recording"]["sources"][0]["source"]
        abs_path = Path(data_args.data_path) / file_path
        exists = abs_path.exists()
        return ((len(example['input_ids']) + int(example["duration"] * SAMPLING_RATE_TOKENIZER)) < arch_args.max_position_embeddings) and exists
    
    if tokenized_train_dataset is not None:
        logger.info(f"original train dataset: {len(tokenized_train_dataset)} samples.")
        tokenized_train_dataset = [ex for ex in tokenized_train_dataset if filter_function(ex)]
        logger.info(f"filtered train dataset: {len(tokenized_train_dataset)} samples.")
        
            
    if tokenized_eval_dataset is not None:
        logger.info(f"original eval dataset: {len(tokenized_eval_dataset)} samples.")
        tokenized_eval_dataset = [ex for ex in tokenized_eval_dataset if filter_function(ex)]
        logger.info(f"filtered eval dataset: {len(tokenized_eval_dataset)} samples.")
    
    processor = AutoProcessor.from_pretrained(model_args.codec)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, processor=processor, pad_to_multiple_of=data_args.pad_to_multiple_of, data_path=data_args.data_path)
    
    return tokenized_train_dataset, tokenized_eval_dataset, data_collator


def train(attn_implementation="sdpa"):

    parser = HfArgumentParser(
        (ArchArguments, ModelArguments, DataArguments, TrainingArguments))
    arch_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # Set seed before initializing model.
    set_seed(training_args.seed)


    tokenizer = RobertaTokenizer.from_pretrained(
        model_args.tokenizer,
        padding_side="left",
        add_eos_token=True,
    )
    arch_args.vocab_size = tokenizer.vocab_size
    model_config = SpeechLlamaConfig(**asdict(arch_args))
    logger.info(f"config: {model_config}")
    
    
    torch_dtype = None #torch.bfloat16 if training_args.bf16 else None
    
    model = SpeechLlamaForCausalLM._from_config(model_config, attn_implementation=attn_implementation, torch_dtype=torch_dtype)
    model.initialize_codec(model_args)
    
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")


    train_dataset, eval_dataset, data_collator = make_supervised_data_module(tokenizer, arch_args, model_args, data_args, training_args)
    trainer = SpeechLlamaTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
if __name__ == "__main__":
    train(attn_implementation="sdpa")
