from typing import Optional, Tuple, Union, Dict
from rich import print
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from dataclasses import dataclass
from transformers.utils import logging
from transformers.modeling_outputs import Seq2SeqModelOutput
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
from transformers.models.whisper.tokenization_whisper import WhisperTokenizer
from transformers.models.whisper.processing_whisper import WhisperProcessor
from transformers.models.whisper.modeling_whisper import (
    WHISPER_INPUTS_DOCSTRING, 
    add_start_docstrings_to_model_forward, 
    replace_return_docstrings, 
    _CONFIG_FOR_DOC, 
    shift_tokens_right,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutput,
    WhisperForConditionalGeneration,
    WhisperDecoderLayer, 
    WhisperDecoder, 
    WhisperModel)
import whisper

logger = logging.get_logger("Clairaudience")

def bregman_div(input: torch.FloatTensor, target: torch.FloatTensor, reduction: str = "mean") -> torch.FloatTensor:
    r"""
    Begman Divergence. A form of KL divergence estimator with much lower variance.

    See http://joschu.net/blog/kl-approx.html 

    Input and target are required to be log_prob

    Args:
        input: a tensor of log_prob. modeled by a NN, i.e., logP(Y|X, \theta)
        target; a tensor of log_prob. modeled by another NN, i.e., logP(Y|X, \theta^*)
    """
    # log_input, log_target = torch.log(input), torch.log(target)
    log_input, log_target = input, target
    log_ratio = log_input - log_target
    term = ((torch.exp(log_ratio) - 1) - log_ratio) * torch.exp(log_target)

    # Use batchmean when there is batch dimension
    if reduction == 'mean':
        return torch.mean(term)
    elif reduction == 'batchmean':
        return torch.sum(term) / term.size(0)
    else:
        raise NotImplementedError


class ClairaudienceConfig(WhisperConfig):
    def __init__(self, 
                 use_kl_loss: bool = True,
                 kl_type: str = "KL_div",
                 kl_coeff: str = 0.2,
                 use_cross_attn: bool = True,
                 use_no_speech_bias: bool = False,
                 **kwargs):
        r"""
        use_kl_loss (bool)
            whether to use kl divergence as a loss term
        
        kl_coeff (`float`, *optional*):
            The weighting coefficient of the KL divergence loss term in ILMA.
            L_TOT = (1 - rho) * L + rho * L_KL

        kl_type (str)
            Options: {KL_div, Bregman_div}. 
            KL_div is the torch.nn.KLDivLoss
            Bregman_div is the Bregman Divergence implmented from http://joschu.net/blog/kl-approx.html 

        use_cross_attn (bool)
            Default = True
            Turn off the decoder's cross attention. Only use the decoder. The encoder forward is not used
            
        use_no_speech_bias (bool)
            Default = False
        
        """
        # note: use data to make model prompt-tunable

        # case 1: normal Whisper
        # => use_kl_loss=False, use_cross_attn=True, use_no_speech_bias=False
        # case 2: ILMA (pure skipping)
        # => use_kl_loss=True, use_cross_attn=False, use_no_speech_bias=False
        # case 3: ILMA (add bias)
        # => use_kl_loss=True, use_cross_attn=False, use_no_speech_bias=True
        # case 4: ILMA + noisy sound
        # => use_kl_loss=True, use_cross_attn=True, use_no_speech_bias=False

        self.use_kl_loss = use_kl_loss
        self.kl_type = kl_type
        self.kl_coeff = kl_coeff
        self.use_cross_attn = use_cross_attn
        self.use_no_speech_bias = use_no_speech_bias
        
        assert not (use_cross_attn and use_no_speech_bias), f"use_cross_attn: {use_cross_attn}; use_no_speech_bias: {use_no_speech_bias}"
        super().__init__(**kwargs)
        

@dataclass
class ClairaudienceSeq2SeqOutput(Seq2SeqModelOutput):
    r"""
    Inherited from Seq2SeqModelOutput used by HF Whisper

    Args:
    
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.

        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

        lm_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Internal Language modeling loss.

        kl_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            KL divergence loss for regularization

        kl_target ('torch.FloatTensor' of shape (batch_size, seq_len, vocab_size), *optional*, returned when use_kl_loss is True)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            Used for KL divergence
        
        target_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
            The hidden_state before the linear projection
            Used for KL divergence

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.

    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    lm_loss: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.FloatTensor] = None
    kl_target: Optional[torch.FloatTensor] = None
    target_decoder_last_hidden_state: Optional[torch.FloatTensor] = None


# Copied from transformers.models.mbart.modeling_mbart.MBartDecoderLayer with MBart->Clairaudience
class ClairaudienceDecoderLayer(WhisperDecoderLayer):
    def __init__(self, config: ClairaudienceConfig):
        super().__init__(config)

        self.config = config
        if self.config.use_no_speech_bias:
            self.no_speech_bias = torch.nn.parameter.Parameter(
                torch.zeros(config.d_model, dtype=torch.float32), requires_grad=True)
        self._is_inference = False

    def set_inference(self, mode=True):
        self._is_inference = mode

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if self._is_inference or self.config.use_cross_attn:
            # Cross-Attention Block
            cross_attn_present_key_value = None
            cross_attn_weights = None
            if encoder_hidden_states is not None:
                residual = hidden_states
                hidden_states = self.encoder_attn_layer_norm(hidden_states)

                # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
                cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
                hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                )
                hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
                hidden_states = residual + hidden_states

                # add cross-attn to positions 3,4 of present_key_value tuple
                present_key_value = present_key_value + cross_attn_present_key_value

        # Clairaudience speific: added to indicate that the cross attention is not used. Decoder trained with text-only data
        if not self._is_inference and self.config.use_no_speech_bias:
            hidden_states += self.no_speech_bias

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    

# Copied from transformers.models.mbart.modeling_mbart.MBartDecoderLayer with MBart->Clairaudience
class ClairaudienceDecoder(WhisperDecoder):
    def __init__(self, config: ClairaudienceConfig):
        super().__init__(config)
        del self.layers
        self.layers = torch.nn.ModuleList([ClairaudienceDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.post_init()

    def set_inference(self, mode=True):
        for layer in self.layers:
            layer.set_inference(mode)

    def shift_by_attention_mask(self, attention_mask, positions):
        """ 
        Shift position embedding weight by attention mask
        """
        seq_len = attention_mask.shape[1]
        bz = attention_mask.shape[0]
        d_model = positions.shape[-1]
        # compute indices for torch.gather operation
        shifts = (seq_len - attention_mask.sum(1)).to(torch.int64)
        shift_indices = torch.arange(seq_len, device=shifts.device).view((1, seq_len)).repeat((bz, 1)) - shifts.view((bz,1))
        shift_indices = (shift_indices + seq_len) % seq_len
        # expand position embedding to match the batched input (seq_len, d_model) -> (bz, seq_len, d_model)
        batch_positions = positions.view((1, seq_len, positions.shape[-1])).expand(bz, seq_len, d_model)
        # (bz, seq_len) -> (bz, seq_len, d_model)
        expanded_shift_indices = shift_indices.view((bz, seq_len, 1)).expand(bz, seq_len, d_model)
        # shift position embeddings
        shifted_positions = torch.gather(batch_positions, 1, expanded_shift_indices)
        # reset the embedding at index 0 to handle cases when beam search forces a start token
        shifted_positions[:,0,:] = positions[0,:]
        return shifted_positions
    
    def select_by_attention_mask(self, attention_mask, full_positions):
        """ 
        Select position embedding weight by attention mask from full position embedding weight
        """
        seq_len = attention_mask.shape[1]
        bz = attention_mask.shape[0]
        d_model = full_positions.shape[-1]
        # compute shift
        shifts = (seq_len - attention_mask.sum(1)).to(torch.int64)
        # select position embedding base on shifts
        positions = full_positions[:seq_len].view((1, seq_len, d_model)).expand(bz, seq_len, d_model)[torch.arange(bz), seq_len - 1 - shifts, :]
        # (bz, d_model) -> (bz, 1, d_model)
        positions = positions.unsqueeze(1)
        return positions
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # store raw attention mask for position shifting as decoder attention mask gets overwritten with causal attention mask 
        if attention_mask is not None:
            raw_attention_mask = attention_mask.clone()
        else:
            raw_attention_mask = None

        # create causal attention mask 
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # get embed positions
        if input_ids is not None:
            positions = self.embed_positions(input_ids, past_key_values_length=past_key_values_length)
        else:
            positions = self.embed_positions(inputs_embeds, past_key_values_length=past_key_values_length)

        # adapt positional embedding to left padding in generation tasks
        if (raw_attention_mask is not None) and (past_key_values_length == 0):
            # Initial pass, input_embeds' shape = (bz, seq_len, d_model)
            shifted_positions = self.shift_by_attention_mask(attention_mask = raw_attention_mask, positions = positions)
            hidden_states = inputs_embeds + shifted_positions
        elif (raw_attention_mask is not None) and (past_key_values_length != 0):
            # With kv cache, input_embeds' shape = (bz, 1, d_model)
            shifted_positions = self.select_by_attention_mask(attention_mask = raw_attention_mask, full_positions = self.embed_positions.weight)
            hidden_states = inputs_embeds + shifted_positions
        else:
            hidden_states = inputs_embeds + positions

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    None,  # encoder attention mask
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,  # past_key_value
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )



class ClairaudienceModel(WhisperModel):
    def __init__(self, config: ClairaudienceConfig):
        super().__init__(config)

        self.target_decoder = None
        if config.use_kl_loss:
            self.target_decoder = self.decoder
        else:
            del self.decoder
        self.decoder = ClairaudienceDecoder(config)
        self._is_inference = False
        self.post_init()

    def set_inference(self, mode=True):
        self._is_inference = mode
        self.decoder.set_inference(mode)
        
    def freeze_target_decoder(self):
        for name, param in self.target_decoder.named_parameters():
            param.requires_grad = False

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], ClairaudienceSeq2SeqOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.use_cross_attn or self._is_inference:
            if encoder_outputs is None:
                input_features = self._mask_input_features(input_features, attention_mask=attention_mask)
                logger.debug(f"CL MODEL: use encoder forward. encoder_outputs is none? {encoder_outputs is None}; use_cross_attn = {self.config.use_cross_attn}")
                encoder_outputs = self.encoder(
                    input_features,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
            elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_input_dict = dict(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0] if encoder_outputs else None,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        if self.config.use_kl_loss:
            target_decoder_outputs = self.target_decoder(**decoder_input_dict)
        
        decoder_outputs = self.decoder(**decoder_input_dict)

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return ClairaudienceSeq2SeqOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state if encoder_outputs else None,
            encoder_hidden_states=encoder_outputs.hidden_states if encoder_outputs else None,
            encoder_attentions=encoder_outputs.attentions if encoder_outputs else None,
            target_decoder_last_hidden_state=target_decoder_outputs.last_hidden_state if self.config.use_kl_loss else None
        )


class ClairaudienceForConditionalGeneration(WhisperForConditionalGeneration):
    def __init__(self, config: ClairaudienceConfig):
        """
        HACK: ClairaudienceForConditionalGeneration.from_pretrained("openai/whisper-tiny") will not set the no_speech_bias correctly when loading from a pretrained whisper model
                use below instead
                => config = ClairaudienceConfig.from_pretrained("openai/whisper-tiny", use_kl_loss=True kl_coeff=0.2, kl_type="KL_div", use_cross_attn=True, use_no_speech_bias=False)
                => model = ClairaudienceForConditionalGeneration("openai/whisper-tiny", config=config)
        """
        super().__init__(config)

        del self.model
        self.model = ClairaudienceModel(config)
        
        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_whisper_pretrained(cls, pretrained_model_name_or_path, use_kl_loss=True, use_cross_attn=True, use_no_speech_bias=False, kl_coeff=0.2, kl_type="KL_div", **kwargs):
        model_config = ClairaudienceConfig(use_kl_loss=use_kl_loss, use_cross_attn=use_cross_attn, use_no_speech_bias=use_no_speech_bias, kl_coeff=kl_coeff, kl_type=kl_type,
                                           **WhisperConfig.from_pretrained(pretrained_model_name_or_path).to_dict())
        model = ClairaudienceForConditionalGeneration(model_config)
        model.load_state_dict(WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, **kwargs).state_dict(), strict=False)
        if use_kl_loss:
            model.model.target_decoder.load_state_dict(model.model.decoder.state_dict(), strict=False)
            model.model.freeze_target_decoder()
            logger.info(f"use kl loss; target_decoder initialized")
        model.freeze_encoder()
        return model
        
    def set_inference(self, mode=True):
        logger.info(f"Set model inference mode = {mode}")
        self.model.set_inference(mode)

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], ClairaudienceSeq2SeqOutput]:        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Use openai's whisper model logits calculation
        # https://github.com/openai/whisper/blob/main/whisper/model.py#L214
        last_hidden_state = outputs.last_hidden_state
        lm_logits = last_hidden_state @ torch.transpose(self.model.decoder.embed_tokens.weight.to(last_hidden_state.dtype), 0, 1)
        
        kl_target = None
        if self.config.use_kl_loss:
            target_hidden_state = outputs.target_decoder_last_hidden_state
            kl_target = target_hidden_state @ torch.transpose(self.model.target_decoder.embed_tokens.weight.to(target_hidden_state.dtype), 0, 1)

        # Huggingface's proj_out method
        # lm_logits = self.proj_out(decoder_outputs[0])

        loss = None
        lm_loss = None
        kl_loss = None
        if labels is not None:
            lm_loss_fct = torch.nn.CrossEntropyLoss()  # TODO: use loss_mask
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            lm_loss = lm_loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
            loss = lm_loss

        if lm_loss is not None and kl_target is not None:
            kl_target = kl_target.to(lm_logits.device)
            log_softmax_lm_logits = F.log_softmax(lm_logits, dim=-1)
            if self.config.kl_type == "KL_div":
                softmax_kl_target = F.softmax(kl_target, dim=-1)
                kl_loss_func = torch.nn.KLDivLoss(reduction='batchmean')
                kl_loss = kl_loss_func(log_softmax_lm_logits, softmax_kl_target)
            elif self.config.kl_type == 'Bregman_div':
                def kl_loss_func(x, y): return bregman_div(x, y, reduction='batchmean')
                log_softmax_kl_target = F.log_softmax(kl_target, dim=-1)
                kl_loss = kl_loss_func(log_softmax_lm_logits, log_softmax_kl_target)
            else:
                raise NotImplementedError
            kl_coeff = self.config.kl_coeff
            loss = (1.0 - kl_coeff) * lm_loss + kl_coeff * kl_loss

        # Adapted from WhisperModel.forward and WhisperForConditionalGeneration.forward outputs to keep consistent for trainer
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss, ) + output) if loss is not None else output

        return ClairaudienceSeq2SeqOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            lm_loss=lm_loss,
            kl_loss=kl_loss,
            kl_target=kl_target
        )
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs,
    ):
        """
        Overwrite Whisper's `prepare_inputs_for_generation` to pass in decoder attention mask
        """
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": kwargs.get("decoder_attention_mask", None),
        }



def init_model(cfg: Dict[str, any]) -> Tuple[ClairaudienceForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor]:
    r"""
    Cfg contains the config that orchestrate the training and evaluation of the clairaudience project
    """
    resume_from_checkpoint = cfg.get("resume_from_checkpoint", None)
    model_name = cfg["model_name"]
    if resume_from_checkpoint is None:
        model = ClairaudienceForConditionalGeneration.from_whisper_pretrained(model_name,
                                                                             use_kl_loss=cfg["use_kl_loss"],
                                                                             kl_coeff=cfg["kl_coeff"], 
                                                                             kl_type=cfg["kl_type"], 
                                                                             use_cross_attn=cfg["use_cross_attn"],
                                                                             use_no_speech_bias=cfg["use_no_speech_bias"]
                                                                             )
        
        logger.info(f"Load model from whisper pretrained checkpoint: {model_name}")
    else:
        model = ClairaudienceForConditionalGeneration.from_pretrained(resume_from_checkpoint)
        logger.info(f"Load model from Clairaudience trained checkpoint: {resume_from_checkpoint}")

    feature_extractor = whisper_feature_extractor
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=cfg["model_force_lang"], task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, language=cfg["model_force_lang"], task="transcribe")
    return model, feature_extractor, tokenizer, processor


def whisper_feature_extractor(raw_audio: np.array):
    audio_padded = whisper.pad_or_trim(raw_audio.flatten())
    input_feature = whisper.log_mel_spectrogram(audio_padded)
    return input_feature