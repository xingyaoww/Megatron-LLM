# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Transformer based language model."""
from typing import Callable

import torch
from torch import nn

import megatron
from megatron.core import mpu, tensor_parallel
from .module import MegatronModule
from megatron.core.parallel_state import get_tensor_model_parallel_rank

import megatron.model.transformer
import megatron.model.utils
from megatron.core.tensor_parallel.utils import VocabUtility

from megatron.model.enums import LayerType, AttnMaskType, PositionEmbeddingType
from megatron.model.utils import init_method_normal, scaled_init_method_normal

from megatron.core.tensor_parallel.layers import _initialize_affine_weight_cpu, _initialize_affine_weight_gpu


def parallel_lm_logits(input_,
                       word_embeddings_weight,
                       parallel_output,
                       bias=None):
    """ LM logits using word embedding weights. """

    args = megatron.get_args()
    # Parallel logits.
    if args.async_tensor_model_parallel_allreduce or\
            args.sequence_parallel:
        input_parallel = input_
        model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        async_grad_allreduce = args.async_tensor_model_parallel_allreduce and \
            model_parallel and not args.sequence_parallel
    else:
        input_parallel = tensor_parallel.copy_to_tensor_model_parallel_region(input_)
        async_grad_allreduce = False

    # Matrix multiply.
    logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
        input=input_parallel,
        weight=word_embeddings_weight,
        bias=bias,
        gradient_accumulation_fusion=args.gradient_accumulation_fusion,
        async_grad_allreduce=async_grad_allreduce,
        sequence_parallel_enabled=args.sequence_parallel)
    # Gather if needed.
    if parallel_output:
        return logits_parallel
    return tensor_parallel.gather_from_tensor_model_parallel_region(logits_parallel)


def get_language_model(num_tokentypes,
                       add_pooler: bool,
                       encoder_attn_mask_type,
                       init_method=None,
                       scaled_init_method=None,
                       add_encoder=True,
                       add_decoder=False,
                       decoder_attn_mask_type=AttnMaskType.causal,
                       pre_process=True,
                       post_process=True,
                       args=None,
                       model_type=None):
    assert args is not None
    # model_type = args.model_type
    """Build language model and return along with the key to save."""
    if init_method is None:
        init_method = init_method_normal(args.init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)
    # Language model.
    language_model = TransformerLanguageModel(
        init_method,
        scaled_init_method,
        encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process,
        args=args,
        model_type=model_type
    )
    # key used for checkpoints.
    language_model_key = 'language_model'
    return language_model, language_model_key


class Pooler(MegatronModule):
    """
    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method, args):
        super(Pooler, self).__init__()
        self.dense = megatron.model.utils.get_linear_layer(hidden_size,
                                                           hidden_size,
                                                           init_method,
                                                           args.perform_initialization)
        self.sequence_parallel = args.sequence_parallel

    def forward(self, hidden_states, sequence_index=0):
        # hidden_states: [s, b, h]
        # sequence_index: index of the token to pool.

        # gather data along sequence dimensions
        # same pooler is run on all tensor parallel nodes
        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False)

        pooled = hidden_states[sequence_index, :, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_position_embeddings,
                 embedding_dropout_prob,
                 init_method,
                 num_tokentypes=0):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes

        args = megatron.get_args()

        # Word embeddings (parallel).
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            vocab_size, self.hidden_size,
            init_method=self.init_method,
            params_dtype=args.params_dtype,
            use_cpu_initialization=args.use_cpu_initialization,
            perform_initialization=args.perform_initialization
        )
        self._word_embeddings_key = 'word_embeddings'

        # Position embedding (serial).
        self.position_embedding_type = args.position_embedding_type
        if self.position_embedding_type == PositionEmbeddingType.absolute:
            assert max_position_embeddings is not None
            self.position_embeddings = torch.nn.Embedding(
                max_position_embeddings, self.hidden_size)
            self._position_embeddings_key = 'position_embeddings'
            # Initialize the position embeddings.
            # if args.perform_initialization: # NOTE: always initialize them if absolute?
            self.init_method(self.position_embeddings.weight)
        else:
            self.position_embeddings = None

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes,
                                                           self.hidden_size)
            # Initialize the token-type embeddings.
            if args.perform_initialization:
                self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        self.fp32_residual_connection = args.fp32_residual_connection 
        self.sequence_parallel = args.sequence_parallel
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            print('adding embedding for {} tokentypes'.format(num_tokentypes),
                  flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes,
                                                       self.hidden_size)
        # Initialize the token-type embeddings.
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings

        if self.position_embedding_type == PositionEmbeddingType.absolute:
            assert self.position_embeddings is not None
            embeddings = embeddings + self.position_embeddings(position_ids)
        else:
            assert self.position_embeddings is None

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] \
            = self.word_embeddings.state_dict(prefix=prefix,
                                              keep_vars=keep_vars)
        if self.position_embedding_type == PositionEmbeddingType.absolute:
            state_dict_[self._position_embeddings_key] \
                = self.position_embeddings.state_dict(prefix=prefix,
                                                    keep_vars=keep_vars)
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] \
                = self.tokentype_embeddings.state_dict(prefix=prefix,
                                                       keep_vars=keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] \
                        = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self.position_embedding_type == PositionEmbeddingType.absolute:
            if self._position_embeddings_key in state_dict:
                state_dict_ = state_dict[self._position_embeddings_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'position_embeddings' in key:
                        state_dict_[key.split('position_embeddings.')[1]] \
                            = state_dict[key]
            self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if 'tokentype_embeddings' in key:
                        state_dict_[key.split('tokentype_embeddings.')[1]] \
                            = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_,
                                                          strict=strict)
            else:
                print('***WARNING*** expected tokentype embeddings in the '
                      'checkpoint but could not find it', flush=True)


class TransformerLanguageModel(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 init_method: Callable,
                 output_layer_init_method,
                 encoder_attn_mask_type,
                 num_tokentypes=0,
                 add_encoder=True,
                 add_decoder=False,
                 decoder_attn_mask_type=AttnMaskType.causal,
                 add_pooler=False,
                 pre_process=True,
                 post_process=True,
                 args=None,
                 model_type=None):
        super(TransformerLanguageModel, self).__init__()
        assert args is not None

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.add_encoder = add_encoder
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_decoder = add_decoder
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.encoder_hidden_state = None
        self.sequence_parallel = args.sequence_parallel

        self.vision_patch_size = args.vision_patch_size

        s = args.max_position_embeddings
        ell = args.num_layers
        v = args.padded_vocab_size
        h = args.hidden_size
        mlp_mult_term = 64 if args.glu_activation else 16

        qkv_estimate = 6 * s * (h ** 2)
        attention_mat_estimate = 2 * (s ** 2) * h
        attention_vals_estimate = 2 * (s ** 2) * h
        linear_proj_estimate = 2 * s * (h ** 2)
        mlp_estimate = mlp_mult_term * s * h ** 2
        embedding_estimate = 6 * s * h * v

        per_layer_estimate = (qkv_estimate + attention_mat_estimate + attention_vals_estimate + linear_proj_estimate + mlp_estimate)
        self.flop_estimate = ell * per_layer_estimate + embedding_estimate

        # Embeddings.
        if self.pre_process:
            self.embedding = Embedding(self.hidden_size,
                                       args.padded_vocab_size,
                                       args.max_position_embeddings,
                                       args.hidden_dropout if not args.lima_dropout else 0.0,
                                       self.init_method,
                                       self.num_tokentypes)
            self._embedding_key = 'embedding'

        # Is MultiModal Model
        if self.vision_patch_size is not None:
            world_size = megatron.core.mpu.get_tensor_model_parallel_world_size()
            extra_kwargs = megatron.model.transformer._args_to_kwargs(args)
            # NOTE: we explicitly disable sequence_parallel_enabled for the vision patch embedding
            # since the input to self.embed_vision_patch is NOT YET in sequence parallel format
            extra_kwargs["sequence_parallel_enabled"] = False
            # print(f"extra_kwargs: {extra_kwargs}")
            self.embed_vision_patch = megatron.core.tensor_parallel.ColumnParallelLinear(
                self.vision_patch_size * self.vision_patch_size * 3,  # 32 * 32 * 3,
                self.hidden_size,
                bias=args.use_bias,
                gather_output=True,
                init_method=init_method,
                skip_bias_add=True,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **extra_kwargs,
                world_size=world_size)
            self._embed_vision_patch_key = 'embed_vision_patch'

        # Transformer.
        # Encoder (usually set to True, False if part of an encoder-decoder
        # architecture and in encoder-only stage).
        if self.add_encoder:
            self.encoder = megatron.model.transformer.ParallelTransformer(
                self.init_method,
                output_layer_init_method,
                self_attn_mask_type=self.encoder_attn_mask_type,
                pre_process=self.pre_process,
                post_process=self.post_process,
                args=args,
                model_type=model_type
            )
            self._encoder_key = 'encoder'
        else:
            self.encoder = None

        # Decoder (usually set to False, True if part of an encoder-decoder
        # architecture and in decoder-only stage).
        if self.add_decoder:
            self.decoder = megatron.model.transformer.ParallelTransformer(
                self.init_method,
                output_layer_init_method,
                layer_type=LayerType.decoder,
                self_attn_mask_type=self.decoder_attn_mask_type,
                pre_process=self.pre_process,
                post_process=self.post_process,
                args=args,
                model_type=model_type
            )
            self._decoder_key = 'decoder'
        else:
            self.decoder = None

        if self.post_process:
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method, args)
                self._pooler_key = 'pooler'

        # Classifiaction head.
        self.tie_embed_logits = args.tie_embed_logits
        if self.post_process and not self.tie_embed_logits:
            # instantiate head
            vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                args.padded_vocab_size, get_tensor_model_parallel_rank(),
                args.tensor_model_parallel_size
            )
            num_embeds = vocab_end_index - vocab_start_index
            data = torch.empty(num_embeds, self.hidden_size, dtype=args.params_dtype,
                               device=None if args.use_cpu_initialization else torch.cuda.current_device())
            self.lm_head = nn.Parameter(data)
            self._lm_key = "lm_head"
            init_method = nn.init.xavier_uniform_ if args.init_method_xavier_uniform else nn.init.xavier_normal_
            # init weights
            if args.perform_initialization:
                if args.use_cpu_initialization:
                    _initialize_affine_weight_cpu(self.lm_head, args.padded_vocab_size,
                                                  num_embeds, 0, init_method,
                                                  params_dtype=args.params_dtype)
                else:
                    _initialize_affine_weight_gpu(self.lm_head, init_method,
                                                  partition_dim=0, stride=1)


    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""

        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        if self.add_encoder and self.add_decoder:
            assert len(input_tensor) == 1, \
                'input_tensor should only be length 1 for stage with both encoder and decoder'
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            assert len(input_tensor) == 1, \
                'input_tensor should only be length 1 for stage with only encoder'
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_decoder:
            if len(input_tensor) == 2:
                self.decoder.set_input_tensor(input_tensor[0])
                self.encoder_hidden_state = input_tensor[1]
            elif len(input_tensor) == 1:
                self.decoder.set_input_tensor(None)
                self.encoder_hidden_state = input_tensor[0]
            else:
                raise Exception('input_tensor must have either length 1 or 2')
        else:
            raise Exception('Stage must have at least either encoder or decoder')

    def get_vision_embeds(self, vision_patch_indices, vision_patches):
        # === Handle vision patches ===
        # print(f"vision_patch_indices: {vision_patch_indices.shape}")
        # print(f"vision_patches: {vision_patches.shape}")
        # add dummy dimension for vision_patches
        vision_patches = vision_patches.unsqueeze(0)  # (1, n_patches, 32 * 32 * 3)
        vision_embeds, _unused_bias = self.embed_vision_patch(
            vision_patches
        )  # (1, n_patches, hidden_size)
        # print(f"vision_embeds right after linear: {vision_embeds.dtype} {vision_embeds.shape}")
        vision_embeds = torch.cat(
            [
                vision_embeds.squeeze(0),
                # add a dummy token (for text)
                torch.zeros(1, vision_embeds.shape[-1], dtype=vision_embeds.dtype).to(vision_embeds.device),
            ],
        )  # (n_patches + 1, hidden_size)

        # arrange embeddings according to vision_patch_indices
        # - text tokens are -1 (map to the dummy zero tensor)
        # - vision tokens are 0~n_patches (map to the corresponding vision_embeds)
        vision_embeds = vision_embeds[vision_patch_indices]  # (batch_size, seq_length, hidden_size)
        # print(f"vision_embeds after selection: {vision_embeds.dtype} {vision_embeds.shape}")

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        # This is required by sequence parallelism (check Embedding implementation)
        vision_embeds = vision_embeds.transpose(0, 1).contiguous()
        # print(f"vision_embeds after moving the seq_dim to 0: {vision_embeds.shape}")

        # vision_embeds = tensor_parallel.gather_from_tensor_model_parallel_region(vision_embeds)
        # print(f"vision_embeds: {vision_embeds.shape}")

        if self.sequence_parallel:
            # print(f"vision_embeds before scatter: {vision_embeds.shape}")
            vision_embeds = tensor_parallel.scatter_to_sequence_parallel_region(vision_embeds)
            # print(f"vision_embeds after scatter: {vision_embeds.shape}")

        return vision_embeds

    def forward(self, 
                enc_input_ids, 
                enc_position_ids, 
                enc_attn_mask,
                dec_input_ids=None, 
                dec_position_ids=None, 
                dec_attn_mask=None,
                enc_dec_attn_mask=None, 
                tokentype_ids=None,
                inference_params=None,
                pooling_sequence_index=0,
                enc_hidden_states=None, 
                output_enc_hidden=False,
                vision_patch_indices=None,  # (batch_size, seq_length), "-1" for text token
                vision_patches=None,  # (n_patches, 32 * 32 * 3)
                ):

        # Encoder embedding.
        if self.pre_process:
            # print(f"enc_input_ids: {enc_input_ids.shape}")
            # print(f"enc_position_ids: {enc_position_ids.shape}")
            encoder_input = self.embedding(enc_input_ids, enc_position_ids,
                                           tokentype_ids=tokentype_ids)
            # print(f"encoder_input: {encoder_input.shape}")
            if vision_patches is not None:
                # assert vision_patch_indices is not None
                # print(f"vision_patches dtype: {vision_patches.dtype}")
                vision_embeds = self.get_vision_embeds(
                    vision_patch_indices, vision_patches
                )
                # print(f"vision_embeds dtype: {vision_embeds.dtype}")
                # print(f"encoder_input dtype: {encoder_input.dtype}")
                encoder_input = encoder_input + vision_embeds
        else:
            encoder_input = None

        # Run encoder.
        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output = self.encoder(
                    encoder_input,
                    enc_attn_mask,
                    inference_params=inference_params,
                    position_ids=enc_position_ids)
            else:
                encoder_output = self.encoder_hidden_state
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)

        if self.post_process:
            if self.add_pooler:
                pooled_output = self.pooler(encoder_output,
                                            pooling_sequence_index)

        # output_enc_hidden refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output
            else:
                return encoder_output

        # Decoder embedding.
        if self.pre_process:
            decoder_input = self.embedding(dec_input_ids,
                                           dec_position_ids)
            if vision_patches is not None:
                assert vision_patch_indices is not None
                vision_embeds = self.get_vision_embeds(
                    vision_patch_indices, vision_patches
                )
                encoder_input = encoder_input + vision_embeds
        else:
            decoder_input = None

        # Run decoder.
        decoder_output = self.decoder(
            decoder_input,
            dec_attn_mask,
            encoder_output=encoder_output,
            enc_dec_attn_mask=enc_dec_attn_mask,
            inference_params=inference_params)

        if self.add_pooler and self.post_process:
            return decoder_output, encoder_output, pooled_output
        else:
            return decoder_output, encoder_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        if self.pre_process:
            state_dict_[self._embedding_key] \
                = self.embedding.state_dict_for_save_checkpoint(prefix=prefix,
                                                                keep_vars=keep_vars)
        if self.vision_patch_size is not None:
            state_dict_[self._embed_vision_patch_key] \
                = self.embed_vision_patch.state_dict(prefix=prefix,
                                                     keep_vars=keep_vars)
        if self.add_encoder:
            state_dict_[self._encoder_key] \
                = self.encoder.state_dict_for_save_checkpoint(prefix=prefix,
                                                              keep_vars=keep_vars)
        if self.post_process:
            if self.add_pooler:
                state_dict_[self._pooler_key] \
                    = self.pooler.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
            if not self.tie_embed_logits:
                state_dict_[self._lm_key] = self.lm_head.data
        if self.add_decoder:
            state_dict_[self._decoder_key] \
                = self.decoder.state_dict_for_save_checkpoint(prefix=prefix,
                                                              keep_vars=keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if self.pre_process:
            if self._embedding_key in state_dict:
                state_dict_ = state_dict[self._embedding_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if '_embeddings' in key:
                        state_dict_[key] = state_dict[key]

            # handle the case when the model is loaded from a checkpoint
            # and the vocab size is smaller than the current padded vocab size
            if "word_embeddings.weight" in state_dict_:
                _state_dict_vocab_size = state_dict_["word_embeddings.weight"].shape[0]
                _current_vocab_size = self.embedding.word_embeddings.weight.shape[0]
                if _state_dict_vocab_size < _current_vocab_size:
                    # expand the state_dict to match the current vocab size
                    _state_dict_vocab = state_dict_["word_embeddings.weight"]
                    _state_dict_vocab = torch.cat([
                        _state_dict_vocab,
                        torch.zeros(_current_vocab_size - _state_dict_vocab_size, _state_dict_vocab.shape[1])
                    ], dim=0)
                    state_dict_["word_embeddings.weight"] = _state_dict_vocab
                    print(f"Expanded the state_dict 'word_embeddings.weight' to match the current padded vocab size: {_state_dict_vocab_size} -> {_current_vocab_size}")
            self.embedding.load_state_dict(state_dict_, strict=strict)

            # Vision patch embedding.
            if self.vision_patch_size is not None:
                if self._embed_vision_patch_key in state_dict:
                    state_dict_ = state_dict[self._embed_vision_patch_key]
                else:
                    # for backward compatibility.
                    state_dict_ = {}
                    for key in state_dict.keys():
                        if 'embed_vision_patch' in key:
                            state_dict_[key] = state_dict[key]
                self.embed_vision_patch.load_state_dict(state_dict_, strict=strict)

        # Classifiaction head.
        if self.post_process and not self.tie_embed_logits:

            _lm_head_vocab_size = state_dict[self._lm_key].shape[0]
            _current_vocab_size = self.lm_head.data.shape[0]
            if _lm_head_vocab_size < _current_vocab_size:
                # expand the state_dict to match the current vocab size
                _state_dict_lm_head = state_dict[self._lm_key]
                _state_dict_lm_head = torch.cat([
                    _state_dict_lm_head,
                    torch.zeros(_current_vocab_size - _lm_head_vocab_size, _state_dict_lm_head.shape[1])
                ], dim=0)
                state_dict[self._lm_key] = _state_dict_lm_head
                print(f"Expanded the state_dict 'lm_head' to match the current padded vocab size: {_lm_head_vocab_size} -> {_current_vocab_size}")
            self.lm_head.data.copy_(state_dict[self._lm_key])

        # Encoder.
        if self.add_encoder:
            if self._encoder_key in state_dict:
                state_dict_ = state_dict[self._encoder_key]
            # For backward compatibility.
            elif 'transformer' in state_dict:
                state_dict_ = state_dict['transformer']
            else:
                # For backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'transformer.' in key:
                        state_dict_[key.split('transformer.')[1]] = state_dict[key]

            # For backward compatibility.
            state_dict_self_attention = {}
            for key in state_dict_.keys():
                if '.attention.' in key:
                    state_dict_self_attention[key.replace(".attention.",
                        ".self_attention.")] = state_dict_[key]
                else:
                    state_dict_self_attention[key] = state_dict_[key]
            state_dict_ = state_dict_self_attention

            self.encoder.load_state_dict(state_dict_, strict=strict)

        if self.post_process:
            if self.add_pooler:
                assert 'pooler' in state_dict, \
                    'could not find data for pooler in the checkpoint'
                self.pooler.load_state_dict(state_dict[self._pooler_key],
                                            strict=strict)
        # Decoder.
        if self.add_decoder:
            assert 'decoder' in state_dict, \
                'could not find data for pooler in the checkpoint'
            self.decoder.load_state_dict(state_dict[self._decoder_key],
                                         strict=strict)
