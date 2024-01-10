"""Mixtral Model."""

import warnings

from megatron import get_args
from .enums import PositionEmbeddingType
from . import GPTModel


class MixtralModel(GPTModel):
    def __init__(self,
                 num_tokentypes: int = 0,
                 parallel_output: bool = True,
                 pre_process: bool = True,
                 post_process: bool = True,
                 model_type=None
                 ):

        args = get_args()

        # mandatory arguments
        assert args.position_embedding_type == PositionEmbeddingType.rotary, \
            f"Mixtral uses rotary embedding, not {args.position_embedding_type}"
        assert not args.use_post_ln, "Mixtral does not use post_ln"
        assert args.glu_activation == "swiglu", "Mixtral works with swiglu activation"
        assert not args.use_bias, "Mixtral does not use bias"
        assert not args.parallel_attn, "Mixtral does not use parallel_attn"
        assert args.use_rms_norm, "Mixtral uses rms_norm"
        assert not args.tie_embed_logits , "Mixtral unties embedding and lm_head weights"
        # https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/discussions/23
        assert args.sliding_window_size is None, "Mixtral do NOT use sliding window attention"

        # check moe arguments
        assert args.do_moe_mlp, "Mixtral uses moe_mlp"
        assert args.num_experts_per_tok == 2, "Mixtral uses 2 experts per token"
        assert args.num_local_experts == 8, "Mixtral uses 8 local experts"

        # recomended arguments
        if not args.use_flash_attn:
            warnings.warn("Mixtral should use flash attn (for sliding window local attention)")

        if args.bias_gelu_fusion:
            warnings.warn("Mixtral is not intended to use bias_gelu_fusion")
        if args.bias_dropout_fusion:
            warnings.warn("Mixtral is not intended to use bias_dropout_fusion")
        if args.hidden_dropout > 0.0 and not args.lima_dropout:
            warnings.warn("Mixtral is not intended to use dropout")
        if args.attention_dropout > 0.0:
            warnings.warn("Mixtral is not intended to use dropout")
        super().__init__(num_tokentypes=num_tokentypes, parallel_output=parallel_output,
                         pre_process=pre_process, post_process=post_process,
                         model_type=model_type)
