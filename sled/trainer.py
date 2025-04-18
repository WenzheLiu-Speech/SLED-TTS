import torch
from typing import List, Optional

from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer import has_length
from transformers.trainer_pt_utils import LengthGroupedSampler

class SpeechLengthGroupedSampler(LengthGroupedSampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        dataset: Optional[Dataset] = None,
        generator=None,
    ):

        self.batch_size = batch_size
        
        lengths = [len(feature["audio"]["array"]) for feature in dataset]  
        self.lengths = lengths
        self.generator = generator




class SpeechLlamaTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None


        if self.args.group_by_speech_length:
            return SpeechLengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
            )
        else:
            return super()._get_train_sampler()