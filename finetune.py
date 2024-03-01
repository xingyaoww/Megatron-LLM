"""Fine-tune gpt, llama or falcon"""

import datetime as dt
from functools import partial

import torch

from megatron import get_args, get_tokenizer, get_timers, get_counters, print_rank_0
from megatron.training import pretrain
from megatron.core import tensor_parallel
from megatron.core.parallel_state import get_data_parallel_group
from megatron.model import GPTModel, ModelType, LlamaModel, FalconModel, MistralModel, MixtralModel
from megatron.utils import get_ltor_masks_and_position_ids, average_losses_across_data_parallel_group
from megatron.data.gpt_dataset import build_train_valid_test_datasets as gpt_build_datasets
from megatron.data.instruction_dataset import instruction_collator
from megatron.data.instruction_dataset import build_train_valid_test_datasets as instruct_build_datasets
from megatron.initialize import initialize_megatron
from megatron.metrics import MetricInput, get_metric


##
# Model provider utilities
##


def model_provider(pre_process: bool = True, post_process: bool = True):
    """Build the model."""

    print_rank_0("Building model ...")

    args = get_args()
    if args.model_name == "gpt": 
        cls = GPTModel
    elif args.model_name == "falcon":
        cls = FalconModel
    elif args.model_name in {"llama", "llama2", "codellama"}:
        cls = partial(LlamaModel, version=1 if args.model_name == "llama" else 2)
    elif args.model_name == "mistral":
        cls = MistralModel
        if args.sliding_window_size != 4096:
            print_rank_0("Mistral uses sliding window attention (set sliding_window=4096)")
            args.sliding_window_size = 4096
    elif args.model_name == "mixtral":
        cls = MixtralModel
    else:
        raise KeyError(f"Unkown model {args.model_name}")

    if isinstance(args.model_type, ModelType):
        model_type = args.model_type
    elif args.model_type == "encoder_or_decoder":
        model_type = ModelType.encoder_or_decoder
    elif args.model_type == "encoder_and_decoder":
        model_type = ModelType.encoder_and_decoder
    else:
        raise KeyError(f"Unsupported model_type {args.model_type}")

    model = cls(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
        model_type=model_type
    )
    return model


##
# Dataset utilities
##

def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    if args.data_type == "gpt":
        keys = ["text"]
        float_keys = []
    elif args.data_type == "instruction":
        keys = ["text", "attention_mask", "position_ids"]
        float_keys = ["loss_mask"]
    else:
        raise KeyError(f"Unknown dataset type {args.data_type}")

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, torch.int64)
    if float_keys:
        data_b_float = tensor_parallel.broadcast_data(float_keys, data, torch.float32)
        for key in float_keys:
            data_b[key] = data_b_float[key]
        del data_b_float

    # Unpack.
    tokens = data_b["text"]
    labels = tokens[:, 1:].contiguous()
    tokens = tokens[:, :-1].contiguous()

    # Update tokens counter.
    counters = get_counters()
    n_tokens = torch.tensor(tokens.numel(), device=tokens.device)
    if args.data_parallel_size == 1:
        n_tokens = n_tokens.item()
    else:
        group = get_data_parallel_group()
        torch.distributed.all_reduce(
            n_tokens, op=torch.distributed.ReduceOp.SUM, group=group
        )
        n_tokens = n_tokens.item()
    counters["tokens"] += n_tokens

    if args.data_type == "gpt":
        # Get the masks and position ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss
        )
        return tokens, labels, loss_mask, attention_mask, position_ids

    # The shift by 1 is handled in the collator for instruction dataset.
    position_ids = data_b["position_ids"].to(tokens.device)
    attention_mask = data_b["attention_mask"].to(tokens.device)
    loss_mask = data_b["loss_mask"].to(tokens.device)
    # # Instruction dataset.
    # # Heavily inspired by Andreas Köpf: https://github.com/andreaskoepf/epfl-megatron/tree/local_changes/
    # attention_mask = data_b["attention_mask"][:, :-1]
    # example_ids = data_b["example_ids"][:, :-1]
    # attention_mask, position_ids = get_attention_mask_and_position_ids(
    #     tokens, attention_mask, example_ids
    # )

    return tokens, labels, loss_mask, attention_mask, position_ids


def data_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    if args.data_type == "gpt":
        builder = gpt_build_datasets
    elif args.data_type == "instruction":
        builder = instruct_build_datasets

    print_rank_0("> building train, validation, and test datasets ...")
    train_ds, valid_ds, test_ds = builder(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path
    )
    print_rank_0("> finished creating datasets ...")

    return train_ds, valid_ds, test_ds


##
# Loss and forward
##


def loss_func(is_training, batch, outputs):
    loss_mask = batch[2]
    losses, logits = outputs
    losses = losses.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    out_dict = {"lm loss": averaged_loss[0]}

    # Calculate other metrics
    if not is_training:
        inputs = MetricInput(batch, logits, averaged_loss[0])
        args = get_args()
        for metric in map(get_metric, args.metrics):
            out_dict.update(metric(inputs))

    return loss, out_dict

# TODO: This is problematic, we include this solely for testing for now
def load_balancing_loss_func(gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2) -> float:
    r"""
    https://github.com/huggingface/transformers/blob/701298d2d3d5c7bde45e71cce12736098e3f05ef/src/transformers/models/mixtral/modeling_mixtral.py#L77C1-L119C38

    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    # treat `top_k` as tokens (shape is `top_k X [batch_size X sequence_length]`)
    selected_experts = selected_experts.reshape(-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    expert_mask = torch.max(expert_mask, dim=-2).values

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(-1))
    return overall_loss * num_experts

def moe_loss_func(is_training, batch, outputs, all_router_logits, args):
    loss, out_dict = loss_func(is_training, batch, outputs)
    
    # add aux loss
    aux_loss = load_balancing_loss_func(all_router_logits, args.num_local_experts, args.num_experts_per_tok)
    loss += args.router_aux_loss_coef * aux_loss
    out_dict["aux loss"] = aux_loss
    out_dict["total loss"] = loss
    return loss, out_dict

def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    batch = get_batch(data_iterator)
    tokens, labels, loss_mask, attention_mask, position_ids = batch
    timers("batch-generator").stop()

    model_output = model(tokens, position_ids, attention_mask,
                          labels=labels)
    if args.do_moe_mlp:
        # model_output = ((losses, logits), all_router_logits)
        # print_rank_0(f"Using MOE MLP, model_output: {model_output}")
        if len(model_output) == 2:
            logits, all_router_logits = model_output
            model_output = logits
        else:
            losses, logits, all_router_logits = model_output
            model_output = (losses, logits)
        # print_rank_0(f"Using MOE MLP, model_output: {model_output}")
        # print_rank_0(f"Using MOE MLP, all_router_logits: {all_router_logits}")
        return model_output, partial(moe_loss_func, model.training, batch, all_router_logits=all_router_logits, args=args)
    else:    
        return model_output, partial(loss_func, model.training, batch)


##
# Main
##


def extra_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='validation set')
    group.add_argument("--model_name",
                       choices={"gpt", "llama", "falcon", "llama2", "codellama", "mistral", "mixtral"},
                       default="gpt")
    group.add_argument("--model_type", choices={"encoder_or_decoder", "encoder_and_decoder"},
                       default="encoder_or_decoder")
    group.add_argument("--loss_role", choices={"assistant", "user", "all"},
                       default="assistant")
    group.add_argument("--data_type", choices={"gpt", "instruction"},
                       default="gpt")
    group.add_argument("--log_learning_rate_to_tensorboard", type=bool, default=True)
    group.add_argument("--log_loss_scale_to_tensorboard", type=bool, default=True)
    return parser


if __name__ == "__main__":
    args_defaults = {"tokenizer_type": "GPT2BPETokenizer"}
    initialize_megatron(extra_args, args_defaults)
    args = get_args()

    if args.data_type == "gpt":
        collate_fn = None
    else:
        return_attention_mask_in_length = args.packed_input and args.use_flash_attn
        collate_fn = partial(
            instruction_collator,
            scalar_loss_mask=args.scalar_loss_mask,
            return_attention_mask_in_length=return_attention_mask_in_length,
            loss_role=args.loss_role,
        )
        print_rank_0(f"Loss role set to {args.loss_role} for instruction dataset")


    pretrain(args, data_provider, model_provider,  ModelType.encoder_or_decoder,
             forward_step, collate_fn=collate_fn)
    print(f"Done {dt.datetime.now(dt.timezone.utc)}")
