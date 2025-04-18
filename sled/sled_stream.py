from typing import List, Optional, Tuple, Union, Dict, Any
from pathlib import Path
import torch
import torch.nn as nn

from transformers import EncodecModel
from .modeling_llama_with_dropout import LlamaConfig, LlamaModel, LlamaForCausalLM
from .modeling_llama_with_dropout import _prepare_4d_causal_attention_mask_with_cache_position
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.utils.import_utils import is_torchdynamo_compiling

from .energy_distance import ScoreLossZ
import copy


class ModifiedGenerateDecoderOnlyOutput(GenerateDecoderOnlyOutput):
    features = None
    

def interleave_embeddings_and_mask_efficient(text_embeds, speech_embeds, text_mask, speech_mask, n, m):
    
    """

    Parameter:
    - text_embeds: Tensor with shape (b, t1, d)
    - speech_embeds: Tensor with shape (b, t2, d)
    - text_mask: Tensor with shape: (b, t1)
    - speech_mask: Tensor with shape: (b, t2)
    - n: int，number of consecutive text_embeds 
    - m: int，number of consecutive speech_embeds

    Return:
    - interleaved: Tensor with shape (b, t1 + t2, d)
    - interleaved_mask: Tensor with shape (b, t1 + t2)
    - speech_positions_mask: Tensor with shape (b, t1 + t2)
    """
    
    b, t1, d = text_embeds.size()
    _, t2, _ = speech_embeds.size()

    
    num_cycles_text = t1 // n
    num_cycles_speech = t2 // m
    total_cycles = min(num_cycles_text, num_cycles_speech)
    
    interleaved_blocks = []
    interleaved_mask_blocks = []
    speech_positions_mask_blocks = []


    if total_cycles > 0:
        text_main = text_embeds[:, :total_cycles * n, :].reshape(b, total_cycles, n, d)
        speech_main = speech_embeds[:, :total_cycles * m, :].reshape(b, total_cycles, m, d)
        
        text_mask_main = text_mask[:, :total_cycles * n].reshape(b, total_cycles, n)
        speech_mask_main = speech_mask[:, :total_cycles * m].reshape(b, total_cycles, m)

        interleaved_main = torch.cat([text_main, speech_main], dim=2).reshape(b, total_cycles * (n + m), d)
        interleaved_blocks.append(interleaved_main)
        
        interleaved_mask_main = torch.cat([text_mask_main, speech_mask_main], dim=2).reshape(b, total_cycles * (n + m))
        interleaved_mask_blocks.append(interleaved_mask_main)
    
        
        text_zero_mask = torch.zeros_like(text_mask_main)
        speech_one_main = torch.ones_like(speech_mask_main)
        
        speech_positions_main = torch.cat([text_zero_mask, speech_one_main], dim=2).reshape(b, total_cycles * (n + m))
        speech_positions_mask_blocks.append(speech_positions_main)
    
    
    remaining_text = text_embeds[:, total_cycles * n:, :]
    remaining_speech = speech_embeds[:, total_cycles * m:, :]

    remaining_mask_text = text_mask[:, total_cycles * n:]
    remaining_mask_speech = speech_mask[:, total_cycles * m:]

    
    remaining_num_text = remaining_text.size(1)
    remaining_num_speech = remaining_speech.size(1)
    
    assert (remaining_num_text < n) or (remaining_num_speech < m)
    
    interleaved_blocks.append(remaining_text[:, :n, :])    
    interleaved_blocks.append(remaining_speech)
    interleaved_blocks.append(remaining_text[:, n:, :])
    
    interleaved_mask_blocks.append(remaining_mask_text[:, :n])    
    interleaved_mask_blocks.append(remaining_mask_speech)
    interleaved_mask_blocks.append(remaining_mask_text[:, n:])
    
    speech_positions_mask_blocks.append(torch.zeros_like(remaining_mask_text[:, :n]))
    speech_positions_mask_blocks.append(torch.ones_like(remaining_mask_speech))
    speech_positions_mask_blocks.append(torch.zeros_like(remaining_mask_text[:, n:]))
        
    interleaved = torch.cat(interleaved_blocks, dim=1)
    interleaved_mask = torch.cat(interleaved_mask_blocks, dim=1)    
    speech_positions_mask = torch.cat(speech_positions_mask_blocks, dim=1)
    
    assert interleaved.size(1) == (t1 + t2) == interleaved_mask.size(1) == speech_positions_mask.size(1)

    return interleaved, interleaved_mask, speech_positions_mask



def get_previous_non_pad_indices(attention_mask):
    """
    find previous non-pad index for each position
    
    Parameter:
    attention_mask (torch.Tensor): shape (batch_size, seq_length), 0 for pad，1 for non-pad。
    
    Return:
    torch.Tensor: shape (batch_size, seq_length), each element is the index of the previous non-pad position, or -1 if there is none.
    """
    
    batch_size, seq_length = attention_mask.shape
    
    
    indices = torch.arange(seq_length, device=attention_mask.device).unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_length)
    
    
    mask = attention_mask == 1
    indices = torch.where(mask, indices, torch.full_like(indices, -1))
    
    
    shifted_indices = torch.cat([torch.full((batch_size, 1), -1, device=attention_mask.device), indices[:, :-1]], dim=1)
    
    
    previous_non_pad = torch.cummax(shifted_indices, dim=1).values
    
    return previous_non_pad



class SpeechLlamaConfig(LlamaConfig):
    model_type = "speech_llama"
    
    def __init__(
        self,
        vae_embed_dim=128,
        diffloss_d=3,
        diffloss_w=1024,
        training_cfg=0.0,
        noise_channels=128,
        stream_n=5,
        stream_m=45,
        **kwargs,
    ):
        
        self.vae_embed_dim = vae_embed_dim
        self.diffloss_d = diffloss_d
        self.diffloss_w = diffloss_w
        self.training_cfg = training_cfg
        self.noise_channels = noise_channels
        self.stream_n = stream_n
        self.stream_m = stream_m

        
        super().__init__(**kwargs)
    




class SpeechLlamaForCausalLM(LlamaForCausalLM):
    config_class = SpeechLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlamaModel(config)
        self.codec = None
                
        # --------------------------------------------------------------------------
        # Speech Embedding
        self.token_embed_dim = config.vae_embed_dim
        self.hidden_size = config.hidden_size
        self.z_proj = nn.Linear(self.token_embed_dim, self.hidden_size, bias=True)
        self.z_proj_ln = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.eos_head = nn.Linear(self.hidden_size, 1, bias=False)
        self.embed_mean = None
        self.embed_std = None
        self.training_cfg = config.training_cfg

        # --------------------------------------------------------------------------
        # Score Loss
        self.scoreloss = ScoreLossZ(
            target_channels=self.token_embed_dim,
            z_channels=self.hidden_size,
            width=config.diffloss_w,
            depth=config.diffloss_d, 
            noise_channels=config.noise_channels,
        )
        
        # --------------------------------------------------------------------------
        # BCE Loss
        pos_weight = torch.Tensor([100.]) # Weight of EOS is equal to 100
        self.bceloss = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

        self.stream_n = config.stream_n
        self.stream_m = config.stream_m
        
        # Initialize weights and apply final processing
        self.post_init() # Check whether affects init of LlamaModel
        
    def initialize_codec(self, model_args):
        if hasattr(model_args, "codec"):
            self.codec = EncodecModel.from_pretrained(model_args.codec, torch_dtype=self.model.dtype)
        else:
            self.codec = EncodecModel.from_pretrained(model_args, torch_dtype=self.model.dtype)
        for param in self.codec.parameters():
            param.requires_grad = False



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
    

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values,
        audio_inputs,
    ):
        if self.training_cfg > 0.0:
            bsz = attention_mask.size(0)
            random_mask = torch.rand(bsz) < self.training_cfg
            attention_mask[random_mask] = 0
            attention_mask[random_mask, 0] = 1
            input_ids[random_mask, 0] = 2
        

        text_inputs_embeds = self.model.embed_tokens(input_ids)
        
        with torch.no_grad():
            encoder_outputs = self.codec.encode(audio_inputs["input_values"].to(self.model.dtype), audio_inputs["padding_mask"], bandwidth=6) #1,b,r,t, 1 due to one chunk
            speech_inputs_embeds = self.codec.quantizer.decode(encoder_outputs.audio_codes[0].transpose(0, 1)) #b,d,t
            
            speech_attention_mask = audio_inputs["padding_mask"][..., ::320]
            assert speech_inputs_embeds.size(-1) == speech_attention_mask.size(-1)
            speech_inputs_embeds = speech_inputs_embeds.transpose(1,2) #b,t,d
            
        
        net_speech_inputs_embeds = self.z_proj(speech_inputs_embeds)
        new_inputs_embeds, new_attention_mask, speech_positions_mask = interleave_embeddings_and_mask_efficient(text_inputs_embeds, net_speech_inputs_embeds, attention_mask, speech_attention_mask, self.stream_n, self.stream_m)
        
        new_labels = speech_inputs_embeds
    
        return None, position_ids, new_attention_mask, past_key_values, new_inputs_embeds, new_labels, speech_attention_mask, speech_positions_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        audio_inputs: Optional[Dict[str, Any]] = None,
        speech_inputs_embeds: Optional[torch.FloatTensor] = None,
        speech_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        
        if audio_inputs is not None:
            (
                _,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                speech_attention_mask,
                speech_positions_mask,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                audio_inputs,
            )
        else:
            assert not ((input_ids is None) and (inputs_embeds is None))
            if input_ids is not None and inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
            elif input_ids is None and inputs_embeds is not None:
                inputs_embeds = self.z_proj(inputs_embeds)
            else:
                inputs_embeds = torch.cat([self.z_proj(inputs_embeds), self.model.embed_tokens(input_ids)], dim=1)
            
        inputs_embeds = self.z_proj_ln(inputs_embeds)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = hidden_states[:, -num_logits_to_keep:, :]

        loss = None

        if labels is not None:
            
            bsz, speech_len, _ = labels.shape
            feat_dim = logits.size(-1)
            generate_index = get_previous_non_pad_indices(attention_mask)
            selected_indices = generate_index.masked_select((speech_positions_mask == 1))  # bsz * speech_len
            selected_indices = selected_indices.view(bsz, speech_len)  # bsz, speech_len
            selected_indices_expanded = selected_indices.unsqueeze(-1).expand(-1, -1, feat_dim)  # bsz * speech_len * feat_dim

            z = torch.gather(logits, dim=1, index=selected_indices_expanded)  # bsz * speech_len * feat_dim
            

            labels = labels.reshape(bsz * speech_len, -1)
            mask = speech_attention_mask.reshape(bsz * speech_len)
            loss = self.scoreloss(z=z.reshape(bsz * speech_len, -1), target=labels, mask=mask)
            
            current_index = torch.arange(attention_mask.size(1), device=attention_mask.device).unsqueeze(0).expand(bsz, -1)  # (bsz, full_seq_length)
            selected_current_indices = current_index.masked_select((speech_positions_mask == 1))  # bsz * speech_len
            selected_current_indices = selected_current_indices.view(bsz, speech_len)[:, -1:]  # bsz, 1
            selected_current_indices_expanded = selected_current_indices.unsqueeze(-1).expand(-1, -1, feat_dim)  # bsz * 1 * feat_dim
            
            z_last = torch.gather(logits, dim=1, index=selected_current_indices_expanded)  # bsz * 1 * feat_dim
            
            z = torch.cat([z, z_last], dim=1)
            
            eos_score = self.eos_head(z).squeeze(-1).float() #bsz, speech_len+1
            
            non_pad_counts = speech_attention_mask.sum(dim=1)
            eos_labels = torch.zeros(bsz, speech_len + 1)
            eos_labels[torch.arange(bsz), non_pad_counts] = 1 #bsz, speech_len+1
            eos_labels = eos_labels.to(eos_score.device)
           
            eos_loss = self.bceloss(eos_score, eos_labels) #bsz, speech_len+1
            
            #Check BCE loss weight BROADCASTING
            ones_column = torch.ones(bsz, 1).to(speech_attention_mask.device)
            loss_mask = torch.cat((ones_column, speech_attention_mask), dim=1) #bsz, speech_len+1
            
            eos_loss = (eos_loss * loss_mask).sum() / loss_mask.sum()
            
            loss = eos_loss + loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Check
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        codec_keys = [k for k in state_dict if 'codec' in k]
        for key in codec_keys:
            del state_dict[key]
        return state_dict

    # Check
    def load_state_dict(self, state_dict, strict=True):
        codec_keys = [k for k in state_dict if 'codec' in k]
        for key in codec_keys:
            del state_dict[key]
        return super().load_state_dict(state_dict, strict=False)
        
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        turn=0,
        num_write_turn=0,
        past_key_values=None,
        attention_mask=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        if num_write_turn !=0 :
            assert cache_position.shape[0] == 1
            assert past_key_values is not None
            inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :, :]
            
            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                
                position_ids = position_ids[:, -inputs_embeds.shape[1] :]
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}

            if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
                raise NotImplementedError

            if num_logits_to_keep is not None:
                model_inputs["num_logits_to_keep"] = num_logits_to_keep

            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "attention_mask": attention_mask,
                }
            )
        
        else:
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1 :, :]
                
            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

                position_ids = position_ids[:, -cache_position.shape[0] :]
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": inputs_embeds}

            if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
                raise NotImplementedError

            if num_logits_to_keep is not None:
                model_inputs["num_logits_to_keep"] = num_logits_to_keep

            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "attention_mask": attention_mask,
                }
            )
        return model_inputs
    
    
    def prepare_inputs_for_generation_cfg(
        self,
        input_ids,
        inputs_embeds=None,
        past_key_values=None,
        attention_mask=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        if cache_position[0] == 0:
            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

            if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
                raise NotImplementedError

            if num_logits_to_keep is not None:
                model_inputs["num_logits_to_keep"] = num_logits_to_keep

            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "attention_mask": attention_mask,
                }
            )
        else:
            if past_key_values is not None:
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :, :]
                
            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values:
                    position_ids = position_ids[:, -inputs_embeds.shape[1] :]
                    position_ids = position_ids.clone(memory_format=torch.contiguous_format)

            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}

            if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
                raise NotImplementedError

            if num_logits_to_keep is not None:
                model_inputs["num_logits_to_keep"] = num_logits_to_keep

            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "attention_mask": attention_mask,
                }
            )
            
        return model_inputs
    
    
    
    def _get_initial_cache_position_after_each_read(self, input_ids, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange` 
        cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()

            # TODO(joao): this is not torch.compile-friendly, find a work-around. If the cache is not empty,
            # end-to-end compilation will yield bad results because `cache_position` will be incorrect.
            if not is_torchdynamo_compiling():
                cache_position = cache_position + past_length
        if model_kwargs.get("cache_position", None) is not None:
            model_kwargs["cache_position"] = torch.cat([model_kwargs["cache_position"], cache_position + 1], dim=0)
        else:
            model_kwargs["cache_position"] = cache_position
        return model_kwargs
    
    
    def _get_initial_cache_position_cfg(self, input_ids, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()

            # TODO(joao): this is not torch.compile-friendly, find a work-around. If the cache is not empty,
            # end-to-end compilation will yield bad results because `cache_position` will be incorrect.
            if not is_torchdynamo_compiling():
                cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position
        return model_kwargs
    

    def _sample(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        streamer,
        **model_kwargs,
    ):
        r"""
        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        bos_token_id = generation_config._bos_token_tensor
        eos_token_id = generation_config._eos_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        
        
        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        assert batch_size == 1 # only support batch size 1
        turn = 0
        read_finished = False
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        last_src_read_len = 0
        cur_len = 0
        complete_attention_mask = model_kwargs.pop("attention_mask")
        model_kwargs["attention_mask"] = torch.zeros((batch_size, 0), dtype=complete_attention_mask.dtype, device=complete_attention_mask.device)
        inputs_embeds = None
        generate_ids = torch.zeros((batch_size, 0), dtype=input_ids.dtype, device=input_ids.device)


        if self.infer_cfg != 1.0:
            model_kwargs_cfg = copy.deepcopy(model_kwargs)
            model_kwargs_cfg["attention_mask"] = torch.ones((batch_size, 1), dtype=complete_attention_mask.dtype, device=complete_attention_mask.device)
            input_ids_cfg = torch.ones((batch_size, 1), dtype=input_ids.dtype, device=input_ids.device)
            input_ids_cfg[:, 0] = 2
            inputs_embeds_cfg = None
            model_kwargs_cfg = self._get_initial_cache_position_cfg(input_ids_cfg, model_kwargs_cfg)
        

        while read_finished == False:

            src_read_len = min(input_ids.shape[1], (turn + 1) * self.stream_n)
            if src_read_len == input_ids.shape[1]:
                read_finished = True

            
            this_turn_input_ids = input_ids[:, last_src_read_len:src_read_len]
            model_kwargs = self._get_initial_cache_position_after_each_read(this_turn_input_ids, model_kwargs)
            model_kwargs["attention_mask"] = torch.cat([model_kwargs["attention_mask"], complete_attention_mask[:, last_src_read_len:src_read_len]], dim=1)
            
            #self.prompt_length = input_ids.shape[1] if inputs_embeds is None else input_ids.shape[1] + inputs_embeds.shape[1]
            
            cur_len = cur_len + src_read_len - last_src_read_len
            
            
            last_src_read_len = src_read_len
            num_write_in_turn = 0


            while self._has_unfinished_sequences(
                this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
            ):
                if (not read_finished) and (num_write_in_turn == self.stream_m):
                    break

                # prepare model inputs
                model_inputs = self.prepare_inputs_for_generation(this_turn_input_ids, inputs_embeds, turn, num_write_in_turn, **model_kwargs)
                if self.infer_cfg != 1.0:
                    model_inputs_cfg = self.prepare_inputs_for_generation_cfg(input_ids_cfg, inputs_embeds_cfg, **model_kwargs_cfg)


                # prepare variable output controls (note: some models won't accept all output controls)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                # forward pass to get next token
                outputs = self(**model_inputs, return_dict=True)
                if self.infer_cfg != 1.0:
                    outputs_cfg = self(**model_inputs_cfg, return_dict=True)
                    

                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need

                # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits.clone()[:, -1, :].float()
                eos_next_token_logits = next_token_logits.clone()
                if self.infer_cfg != 1.0:
                    next_token_logits_cfg = outputs_cfg.logits.clone()[:, -1, :].float()
                    next_token_logits = next_token_logits_cfg + self.infer_cfg * (next_token_logits - next_token_logits_cfg)
                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_logits:
                        raw_logits += (next_token_logits,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

                # token selection
                if do_sample:
                    next_embeds = self.scoreloss.sample(next_token_logits, temperature=1.0) # bsz, dim

                    next_actions = torch.sigmoid(self.eos_head(eos_next_token_logits)) >= 0.5 # 0: continue, 1: stop
                    # if read_finished:
                    #     next_tokens = torch.where(next_actions == 0, bos_token_id, eos_token_id)
                    # else:
                    #     next_tokens = torch.full((batch_size,), bos_token_id ,device=next_actions.device)
                    next_tokens = torch.where(next_actions == 0, bos_token_id, eos_token_id)
                else:
                    raise NotImplementedError

                
                # finished sentences should have their next token be a padding token
                if has_eos_stopping_criteria:
                    next_tokens = next_tokens * unfinished_sequences.unsqueeze(1) + pad_token_id * (1 - unfinished_sequences.unsqueeze(1))

                # update generated ids, model inputs, and length for next step
                if inputs_embeds is not None:
                    inputs_embeds = torch.cat([inputs_embeds, next_embeds[:, None, :]], dim=1)
                else:
                    inputs_embeds = next_embeds[:, None, :]
                
                generate_ids = torch.cat([generate_ids, next_tokens], dim=-1)

                if self.infer_cfg != 1.0:
                    if inputs_embeds_cfg is not None:
                        inputs_embeds_cfg = torch.cat([inputs_embeds_cfg, next_embeds[:, None, :]], dim=1)
                    else:
                        inputs_embeds_cfg = next_embeds[:, None, :]

                    input_ids_cfg = torch.cat([input_ids_cfg, next_tokens], dim=-1)
                

                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                )


                if self.infer_cfg != 1.0:
                    model_kwargs_cfg = self._update_model_kwargs_for_generation(outputs_cfg, model_kwargs_cfg)

                

                unfinished_sequences = unfinished_sequences & ~stopping_criteria(generate_ids, None)
                this_peer_finished = unfinished_sequences.max() == 0
                cur_len += 1
                num_write_in_turn += 1

                # This is needed to properly delete outputs.logits which may be very large for first iteration
                # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                del outputs
            
            turn += 1

        if return_dict_in_generate:
            return ModifiedGenerateDecoderOnlyOutput(
                sequences=generate_ids,
                features=inputs_embeds,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return (generate_ids, inputs_embeds)

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values keeping its naming used in model code
        cache_name, cache = self._extract_past_from_model_output(outputs)
        model_kwargs[cache_name] = cache

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        
        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            raise NotImplementedError
        
        return model_kwargs
