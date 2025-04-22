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
    


class SpeechLlamaConfig(LlamaConfig):
    model_type = "speech_llama"
    
    def __init__(
        self,
        vae_embed_dim=128,
        diffloss_d=3,
        diffloss_w=1024,
        training_cfg=0.0,
        noise_channels=128,
        **kwargs,
    ):
        self.vae_embed_dim = vae_embed_dim
        self.diffloss_d = diffloss_d
        self.diffloss_w = diffloss_w
        self.training_cfg = training_cfg
        self.noise_channels = noise_channels

        
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

        
        # Initialize weights and apply final processing
        self.post_init() # Check whether affects init of LlamaModel
        
    def initialize_codec(self, model_args):
        if hasattr(model_args, "codec"):
            self.codec = EncodecModel.from_pretrained(model_args.codec, torch_dtype=torch.float32) # keep encodec model in fp32
        else:
            self.codec = EncodecModel.from_pretrained(model_args, torch_dtype=torch.float32)
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
            cfg_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            cfg_mask[:, :-1] = random_mask[:, None]
            attention_mask[cfg_mask] = 0
        
        text_inputs_embeds = self.model.embed_tokens(input_ids)
        
        with torch.no_grad():
            encoder_outputs = self.codec.encode(audio_inputs["input_values"], audio_inputs["padding_mask"], bandwidth=6) #1,b,r,t, 1 due to one chunk
            speech_inputs_embeds = self.codec.quantizer.decode(encoder_outputs.audio_codes[0].transpose(0, 1)) #b,d,t, always fp32
            
            speech_attention_mask = audio_inputs["padding_mask"][..., ::320]
            assert speech_inputs_embeds.size(-1) == speech_attention_mask.size(-1)
            speech_inputs_embeds = speech_inputs_embeds.transpose(1,2).to(self.model.dtype) #b,t,d, support full bf16 training
            
        net_speech_inputs_embeds = self.z_proj(speech_inputs_embeds)
        new_inputs_embeds = torch.concat([text_inputs_embeds, net_speech_inputs_embeds], dim=1) #bsz, seq_len, hidden_size
        new_attention_mask = torch.concat([attention_mask, speech_attention_mask], dim=1)
        new_labels = speech_inputs_embeds
    
        return None, position_ids, new_attention_mask, past_key_values, new_inputs_embeds, new_labels, speech_attention_mask
    
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
                inputs_embeds = torch.cat([self.model.embed_tokens(input_ids), self.z_proj(inputs_embeds)], dim=1)
            
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
            
            z = logits[:, -(speech_len+1):-1, :] #bsz, speech_len, hid_dim

            labels = labels.reshape(bsz * speech_len, -1)
            z = z.reshape(bsz * speech_len, -1)
            mask = speech_attention_mask.reshape(bsz * speech_len)
            loss = self.scoreloss(z=z, target=labels, mask=mask)
            
            eos_score = self.eos_head(logits[:, -(speech_len+1):, :]).squeeze(-1).float() #bsz, speech_len+1
           
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

            if inputs_embeds is not None:
                model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": inputs_embeds}
            else:
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
        attention_mask = attention_mask.clone()
        attention_mask[:, :self.prompt_length-1] = 0
        
        if cache_position[0] == 0:
            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

            if inputs_embeds is not None:
                model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

            if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
                raise NotImplementedError

            if num_logits_to_keep is not None:
                model_inputs["num_logits_to_keep"] = num_logits_to_keep

            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                    "past_key_values": copy.deepcopy(past_key_values),
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
                    "past_key_values":copy.deepcopy(past_key_values),
                    "use_cache": use_cache,
                    "attention_mask": attention_mask,
                }
            )
            
        return model_inputs
    
    
    
    def _get_initial_cache_position(self, input_ids, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        if "inputs_embeds" in model_kwargs:
            #cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
            cache_position = (torch.ones(model_kwargs["inputs_embeds"].shape[1] + input_ids.shape[1], dtype=torch.int64).cumsum(0) - 1).to(model_kwargs["inputs_embeds"].device)
        else:
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

        
        inputs_embeds = model_kwargs.get("inputs_embeds", None)

        if self.infer_cfg != 1.0:
            real_batch_size = input_ids.shape[0]
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds.repeat(2, 1, 1) # expand
            input_ids = input_ids.repeat(2, 1) # expand
            self.prompt_length = input_ids.shape[1]
            extended_attention_mask = model_kwargs["attention_mask"].clone()
            extended_attention_mask[:, :self.prompt_length-1] = 0
            model_kwargs["attention_mask"] = torch.cat([model_kwargs["attention_mask"], extended_attention_mask], dim=0) # expand

        
        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape if inputs_embeds is None else (input_ids.shape[0], input_ids.shape[1] + inputs_embeds.shape[1])
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        model_kwargs.pop("inputs_embeds", None)
        

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, inputs_embeds, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)


            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1, :]
            eos_next_token_logits = next_token_logits
            if self.infer_cfg != 1.0:
                next_token_logits_normal = next_token_logits[:real_batch_size, :]
                next_token_logits_cfg = next_token_logits[real_batch_size:, :]
                next_token_logits = next_token_logits_cfg + self.infer_cfg * (next_token_logits_normal - next_token_logits_cfg)
                eos_next_token_logits = eos_next_token_logits[:real_batch_size, :]
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                raise NotImplementedError
                

            # token selection
            if do_sample:
                next_embeds = self.scoreloss.sample(next_token_logits, temperature=1.0) # bsz, dim

                next_actions = torch.sigmoid(self.eos_head(eos_next_token_logits)) >= 0.8 # 0: continue, 1: stop
                next_tokens = torch.where(next_actions == 0, bos_token_id, eos_token_id)

                if self.infer_cfg != 1.0:
                    # exband
                    next_embeds = next_embeds.repeat(2, 1)
                    next_tokens = next_tokens.repeat(2, 1)


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
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
                 
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if self.infer_cfg != 1.0:
            input_ids = input_ids[:real_batch_size]
            inputs_embeds = inputs_embeds[:real_batch_size]

        if return_dict_in_generate:
            return ModifiedGenerateDecoderOnlyOutput(
                sequences=input_ids,
                features=inputs_embeds,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return (input_ids, inputs_embeds)


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
