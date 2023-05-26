import paddle
import numpy as np
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union
from paddlenlp.transformers.activations import ACT2FN
from paddlenlp.transformers.model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.trainer.utils.doc import add_start_docstrings, add_start_docstrings_to_model_forward
from paddlenlp.transformers.llama.configuration import LlamaConfig
import logging
import paddle.nn as nn
import paddle.nn.functional as F
logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = 'LlamaConfig'

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

def _make_causal_mask(input_ids_shape, past_key_values_length, dtype):
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape

    mask = paddle.full((target_length, target_length), float(finfo(dtype).min)).astype(dtype)

    mask_cond = paddle.arange(mask.shape[-1])
    mask = masked_fill(mask, mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1]), 0)

    if past_key_values_length > 0:
        mask = paddle.concat([paddle.zeros([target_length, past_key_values_length]), mask], axis=-1)

    return mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_key_values_length])

def _expand_mask(mask: paddle.Tensor, dtype: paddle.dtype, tgt_len:
    Optional[int]=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(shape=[bsz, 1, tgt_len,
        src_len]).cast(dtype)
    inverted_mask = 1.0 - expanded_mask
    return masked_fill(inverted_mask, inverted_mask.cast("bool"), float(finfo(dtype).min))


class LlamaRMSNorm(paddle.nn.Layer):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

    def forward(self, hidden_states):
        variance = hidden_states.pow(y=2).mean(axis=-1, keepdim
            =True)
        hidden_states = hidden_states * paddle.rsqrt(x=variance + self.
            variance_epsilon)
        return self.weight * hidden_states


class LlamaRotaryEmbedding(paddle.nn.Layer):

    def __init__(self, dim, max_position_embeddings=2048, base=10000,
        device=None):
        super().__init__()
        inv_freq = 1.0 / base ** (paddle.arange(start=0, end=dim, step=2).
            astype(dtype='float32') / dim)
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = paddle.arange(end=self.max_seq_len_cached).astype(self.inv_freq
            .dtype)
        freqs = paddle.einsum('i,j->ij', t, self.inv_freq)
        emb = paddle.concat(x=(freqs, freqs), axis=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :],
            persistable=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :],
            persistable=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = paddle.arange(end=self.max_seq_len_cached).astype(self.
                inv_freq.dtype)
            freqs = paddle.einsum('i,j->ij', t, self.inv_freq)
            if isinstance(x.place, paddle.dtype):
                dtype = x.place
            elif isinstance(x.place, str) and x.place not in ['cpu', 'cuda',
                'ipu', 'xpu']:
                dtype = x.place
            elif isinstance(x.place, paddle.Tensor):
                dtype = x.place.dtype
            else:
                dtype = paddle.concat(x=(freqs, freqs), axis=-1).dtype
            emb = paddle.concat(x=(freqs, freqs), axis=-1).cast(dtype)
            self.register_buffer('cos_cached', emb.cos()[None, None, :, :],
                persistable=False)
            self.register_buffer('sin_cached', emb.sin()[None, None, :, :],
                persistable=False)
        return self.cos_cached[:, :, :seq_len, ...].cast(x.dtype
            ), self.sin_cached[:, :, :seq_len, ...].cast(x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return paddle.concat(x=(-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]
    gather_indices = gather_indices.tile(repeat_times=[1, cos.shape[1], 1,
        cos.shape[3]])
    cos = paddle.take_along_axis(arr=cos.tile(repeat_times=[gather_indices.
        shape[0], 1, 1, 1]), axis=2, indices=gather_indices)
    sin = paddle.take_along_axis(arr=sin.tile(repeat_times=[gather_indices.
        shape[0], 1, 1, 1]), axis=2, indices=gather_indices)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class LlamaMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        if config.tensor_parallel_degree > 1:
            self.gate_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
            self.down_proj = fleet.meta_parallel.RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
            self.up_proj = fleet.meta_parallel.ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(paddle.nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).'
                )
        self.q_proj = paddle.nn.Linear(in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim, bias_attr=False)
        self.k_proj = paddle.nn.Linear(in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim, bias_attr=False)
        self.v_proj = paddle.nn.Linear(in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim, bias_attr=False)
        self.o_proj = paddle.nn.Linear(in_features=self.num_heads * self.
            head_dim, out_features=self.hidden_size, bias_attr=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
            max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        x = tensor.reshape([bsz, seq_len, self.num_heads, self.head_dim])
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        return x.transpose(perm=perm_0)

    def forward(self, hidden_states: paddle.Tensor, attention_mask:
        Optional[paddle.Tensor]=None, position_ids: Optional[paddle.Tensor]
        =None, past_key_value: Optional[Tuple[paddle.Tensor]]=None,
        output_attentions: bool=False, use_cache: bool=False) ->Tuple[
        paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]
        ]:
        bsz, q_len, _ = hidden_states.shape
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        x = self.q_proj(hidden_states).reshape([bsz, q_len, self.num_heads,
            self.head_dim])
        perm_1 = list(range(x.ndim))
        perm_1[1] = 2
        perm_1[2] = 1
        query_states = x.transpose(perm=perm_1)
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        x = self.k_proj(hidden_states).reshape([bsz, q_len, self.num_heads,
            self.head_dim])
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        key_states = x.transpose(perm=perm_2)
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        x = self.v_proj(hidden_states).reshape([bsz, q_len, self.num_heads,
            self.head_dim])
        perm_3 = list(range(x.ndim))
        perm_3[1] = 2
        perm_3[2] = 1
        value_states = x.transpose(perm=perm_3)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states,
            key_states, cos, sin, position_ids)
        if past_key_value is not None:
            key_states = paddle.concat(x=[past_key_value[0], key_states],
                axis=2)
            value_states = paddle.concat(x=[past_key_value[1], value_states
                ], axis=2)
        past_key_value = (key_states, value_states) if use_cache else None
        x = key_states
        perm_4 = list(range(x.ndim))
        perm_4[2] = 3
        perm_4[3] = 2
        attn_weights = paddle.matmul(x=query_states, y=x.transpose(perm=perm_4)
            ) / math.sqrt(self.head_dim)
        if attn_weights.shape != [bsz, self.num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f'Attention weights should be of size {bsz * self.num_heads, q_len, kv_seq_len}, but is {attn_weights.shape}'
                )
        if attention_mask is not None:
            if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
                raise ValueError(
                    f'Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {attention_mask.shape}'
                    )
            attn_weights = attn_weights + attention_mask
            attn_weights = paddle.maximum(
                attn_weights, paddle.full([1], float(finfo(query_states.dtype).min), dtype=attn_weights.dtype)
            )
        attn_weights = paddle.nn.functional.softmax(x=attn_weights, axis=-1
            ).astype('float16')
        attn_output = paddle.matmul(x=attn_weights, y=value_states)
        if attn_output.shape != [bsz, self.num_heads, q_len, self.head_dim]:
            raise ValueError(
                f'`attn_output` should be of size {bsz, self.num_heads, q_len, self.head_dim}, but is {attn_output.shape}'
                )
        x = attn_output
        perm_5 = list(range(x.ndim))
        perm_5[1] = 2
        perm_5[2] = 1
        attn_output = x.transpose(perm=perm_5)
        """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(paddle.nn.Layer):

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        # self.mlp = LlamaMLP(hidden_size=self.hidden_size, intermediate_size
        #     =config.intermediate_size, hidden_act=config.hidden_act)
        # self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.
        #     rms_norm_eps)
        # self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size,
        #     eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)

    def forward(self, hidden_states: paddle.Tensor, attention_mask:
        Optional[paddle.Tensor]=None, position_ids: Optional[paddle.Tensor]
        =None, past_key_value: Optional[Tuple[paddle.Tensor]]=None,
        output_attentions: Optional[bool]=False, use_cache: Optional[bool]=
        False) ->Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.
        Tensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = hidden_states,
        if output_attentions:
            outputs += self_attn_weights,
        if use_cache:
            outputs += present_key_value,
        return outputs


LLAMA_START_DOCSTRING = """
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    'The bare LLaMA Model outputting raw hidden-states without any specific head on top.'
    , LLAMA_START_DOCSTRING)
class LlamaPreTrainedModel(PretrainedModel):
    config_class = LlamaConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['LlamaDecoderLayer']
    _keys_to_ignore_on_load_unexpected = ['decoder\\.version']

    def _init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            # layer.weight
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.config.initializer_range
                        if hasattr(self.config, "initializer_range")
                        else self.llama.config.initializer_range,
                        shape=layer.weight.shape,
                    )
                )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = """
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    'The bare LLaMA Model outputting raw hidden-states without any specific head on top.'
    , LLAMA_START_DOCSTRING)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = paddle.nn.Embedding(config.vocab_size, config.
            hidden_size, self.padding_idx)
        self.layers = paddle.nn.LayerList(sublayers=[LlamaDecoderLayer(
            config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
        inputs_embeds, past_key_values_length,dtype):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, past_key_values_length=past_key_values_length, dtype=dtype
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (expanded_attn_mask if 
                combined_attention_mask is None else expanded_attn_mask +
                combined_attention_mask)
        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(self, input_ids: paddle.Tensor=None, attention_mask:
        Optional[paddle.Tensor]=None, position_ids: Optional[paddle.Tensor]
        =None, past_key_values: Optional[List[paddle.Tensor]]=None,
        inputs_embeds: Optional[paddle.Tensor]=None, query_embeds: Optional
        [paddle.Tensor]=None, use_cache: Optional[bool]=None,
        output_attentions: Optional[bool]=None, output_hidden_states:
        Optional[bool]=None, return_dict: Optional[bool]=None) ->Union[
        Tuple, BaseModelOutputWithPast]:
        output_attentions = (output_attentions if output_attentions is not
            None else self.config.output_attentions)
        output_hidden_states = (output_hidden_states if 
            output_hidden_states is not None else self.config.
            output_hidden_states)
        use_cache = (use_cache if use_cache is not None else self.config.
            use_cache)
        return_dict = (return_dict if return_dict is not None else self.
            config.use_return_dict)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time'
                )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                'You have to specify either decoder_input_ids or decoder_inputs_embeds'
                )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if query_embeds is not None:
            inputs_embeds = paddle.concat(x=[query_embeds, inputs_embeds],
                axis=1)
            batch_size, seq_length, _ = inputs_embeds.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = (seq_length_with_past +
                past_key_values_length)
        self.place=inputs_embeds.place
        if position_ids is None:
            device = (input_ids.place if input_ids is not None else
                inputs_embeds.place)
            position_ids = paddle.arange(start=past_key_values_length, end=
                seq_length + past_key_values_length).astype('int64')
            """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            position_ids =paddle.to_tensor( position_ids.unsqueeze(axis=0).reshape([-1, seq_length]),place=self.place)
        else:
            """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            position_ids = paddle.to_tensor(position_ids.reshape([-1, seq_length]).astype(dtype=
                'int64'),  place=self.place)
        if attention_mask is None:
            attention_mask = paddle.to_tensor(paddle.ones(shape=(batch_size,
                seq_length_with_past),dtype='bool'),place=self.place)
        attention_mask=paddle.to_tensor(attention_mask,place=self.place)
        attention_mask = self._prepare_decoder_attention_mask(attention_mask,
            (batch_size, seq_length), inputs_embeds, past_key_values_length,inputs_embeds.dtype)
        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                    )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += hidden_states,
            past_key_value = past_key_values[idx
                ] if past_key_values is not None else None
#             if self.gradient_checkpointing and self.training:

#                 def create_custom_forward(module):

#                     def custom_forward(*inputs):
#                         return module(*inputs, output_attentions, None)
#                     return custom_forward
# >>>                layer_outputs = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(decoder_layer), hidden_states,
#                     attention_mask, position_ids, None)
            # else:
            layer_outputs = decoder_layer(hidden_states, attention_mask
                =attention_mask, position_ids=position_ids,
                past_key_value=past_key_value, output_attentions=
                output_attentions, use_cache=use_cache)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += layer_outputs[2 if output_attentions else
                    1],
            if output_attentions:
                all_self_attns += layer_outputs[1],
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += hidden_states,
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache,
                all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states,
            past_key_values=next_cache, hidden_states=all_hidden_states,
            attentions=all_self_attns)


class LlamaForCausalLM(LlamaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = paddle.nn.Linear(in_features=config.hidden_size,
            out_features=config.vocab_size, bias_attr=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def prepare_inputs_for_generation(self, input_ids, query_embeds=None,
        past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
        ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            
            position_ids = attention_mask.astype(dtype='int64').cumsum(axis=-1
                ) - 1
            """Class Method: *.masked_fill_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            position_ids= masked_fill(position_ids,attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(axis=-1)
                query_embeds = None
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}
        model_inputs.update({'position_ids': position_ids, 'query_embeds':
            query_embeds, 'past_key_values': past_key_values, 'use_cache':
            kwargs.get('use_cache'), 'attention_mask': attention_mask})
        return model_inputs
    # @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=CausalLMOutputWithPast,
    #     config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: paddle.Tensor=None, attention_mask:
        Optional[paddle.Tensor]=None, position_ids: Optional[paddle.Tensor]
        =None, past_key_values: Optional[List[paddle.Tensor]]=None,
        inputs_embeds: Optional[paddle.Tensor]=None, query_embeds: Optional
        [paddle.Tensor]=None, labels: Optional[paddle.Tensor]=None,
        use_cache: Optional[bool]=None, output_attentions: Optional[bool]=
        None, output_hidden_states: Optional[bool]=None, return_dict:
        Optional[bool]=None):
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\\nI'm not consciours, but I can talk to you."
        ```"""
        output_attentions = (output_attentions if output_attentions is not
            None else self.config.output_attentions)
        output_hidden_states = (output_hidden_states if 
            output_hidden_states is not None else self.config.
            output_hidden_states)
        return_dict = True
        use_cache=True
        outputs = self.model(input_ids=input_ids, attention_mask=
            paddle.ones_like(attention_mask), position_ids=position_ids, past_key_values=
            past_key_values, inputs_embeds=inputs_embeds, query_embeds=
            query_embeds, use_cache=use_cache, output_attentions=
            output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss_fct = paddle.nn.CrossEntropyLoss(reduction='none')
            """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            shift_logits = shift_logits.reshape([-1, self.config.vocab_size])
            """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            shift_labels = shift_labels.reshape([-1])
            shift_labels = shift_labels
            loss = loss_fct(shift_logits, shift_labels).sum()/(shift_labels>0).sum()
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(loss=loss, logits=logits,
            past_key_values=outputs.past_key_values, hidden_states=outputs.
            hidden_states, attentions=outputs.attentions)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += tuple(past_state.index_select(axis=0, index=
                beam_idx) for past_state in layer_past),
        return reordered_past
def finfo(dtype: paddle.dtype = None):
    if dtype is None:
        dtype = paddle.get_default_dtype()

    if dtype == paddle.bfloat16:
        # Numpy do not support `np.finfo(np.uint16)`, so try to construct a finfo object to fetch min value
        class BFloatFInfo:
            min = -3.3895313892515355e38

        return BFloatFInfo
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)