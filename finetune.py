"""Fine-tune gpt, llama or falcon"""

import datetime as dt
from functools import partial

import torch

from megatron import get_args, get_tokenizer, get_timers, get_counters, print_rank_0
from megatron.training import pretrain
from megatron.core import tensor_parallel
from megatron.core.parallel_state import get_data_parallel_group
from megatron.model import GPTModel, ModelType, LlamaModel, FalconModel, MistralModel, MultimodalMistralModel
from megatron.utils import get_ltor_masks_and_position_ids, average_losses_across_data_parallel_group
from megatron.data.gpt_dataset import build_train_valid_test_datasets as gpt_build_datasets
from megatron.data.instruction_dataset import instruction_collator
from megatron.data.instruction_dataset import build_train_valid_test_datasets as instruct_build_datasets
from megatron.data.multimodal_instruction_dataset import build_train_valid_test_datasets as multimodal_instruct_build_datasets
from megatron.data.multimodal_instruction_dataset import instruction_collator as multimodal_instruction_collator
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
    elif args.model_name == "multimodal_mistral":
        cls = MultimodalMistralModel
        if args.sliding_window_size != 4096:
            print_rank_0("MultimodalMistral uses sliding window attention (set sliding_window=4096)")
            args.sliding_window_size = 4096
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
    elif args.data_type == "multimodal_instruction":
        keys = ["text", "attention_mask", "position_ids", "vision_patch_indices"]
        float_keys = ["loss_mask", "vision_patches"]
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

    if args.data_type == "instruction":
        return tokens, labels, loss_mask, attention_mask, position_ids

    vision_patch_indices = data_b["vision_patch_indices"].to(tokens.device)
    vision_patches = data_b["vision_patches"].to(args.params_dtype).to(tokens.device)
    return tokens, labels, loss_mask, attention_mask, position_ids, vision_patch_indices, vision_patches


def data_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    if args.data_type == "gpt":
        builder = gpt_build_datasets
    elif args.data_type == "instruction":
        builder = instruct_build_datasets
    elif args.data_type == "multimodal_instruction":
        builder = multimodal_instruct_build_datasets

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

    if len(batch) == 7:
        tokens, labels, loss_mask, attention_mask, position_ids, vision_patch_indices, vision_patches = batch
        is_image_token = vision_patch_indices != -1
        is_image_example = is_image_token.any(dim=1)

        img_loss = torch.sum(losses[is_image_example].view(-1) * loss_mask[is_image_example].view(-1))
        img_loss_normalized = img_loss / loss_mask[is_image_example].sum()

        txt_loss = torch.sum(losses[~is_image_example].view(-1) * loss_mask[~is_image_example].view(-1))
        txt_loss_normalized = txt_loss / loss_mask[~is_image_example].sum()

        loss = torch.nansum(torch.stack([img_loss, txt_loss])) / loss_mask.sum()

        avg_img_loss = average_losses_across_data_parallel_group([img_loss_normalized], skip_nan=True)
        avg_txt_loss = average_losses_across_data_parallel_group([txt_loss_normalized], skip_nan=True)
        average_loss = average_losses_across_data_parallel_group([loss])

        out_dict = {"lm loss": average_loss[0], "img lm loss": avg_img_loss[0], "txt lm loss": avg_txt_loss[0]}
    else:
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


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    batch = get_batch(data_iterator)

    if args.data_type == "multimodal_instruction":
        tokens, labels, loss_mask, attention_mask, position_ids, vision_patch_indices, vision_patches = batch

        if not forward_step.first_batch_printed:
            print_rank_0("First batch:")
            print_rank_0(f"===============================")
            print_rank_0(f"tokens: {tokens[:, :]}")
            print_rank_0(f"labels: {labels[:, :]}")
            print_rank_0(f"loss_mask (sum): {loss_mask.sum()}")
            print_rank_0(f"loss_mask: {loss_mask[:, :]}")
            print_rank_0(f"attention_mask: {attention_mask[:, :]}")
            print_rank_0(f"position_ids: {position_ids[:, :]}")
            print_rank_0(f"vision_patch_indices: {vision_patch_indices[:, :]}")
            print_rank_0(f"vision_patches: {vision_patches}")
            # print where loss_mask is not 0
            print_rank_0(f"tokens M: {tokens[:, loss_mask[0] == 1]}")
            print_rank_0(f"labels M: {labels[:, loss_mask[0] == 1]}")
            print_rank_0(f"vision_patch_indices M: {vision_patch_indices[:, loss_mask[0] == 1]}")
            forward_step.first_batch_printed = True

        timers("batch-generator").stop()
        output_tensor = model(tokens, position_ids, attention_mask,
                              vision_patch_indices=vision_patch_indices,
                              vision_patches=vision_patches,
                              labels=labels)
    else:
        tokens, labels, loss_mask, attention_mask, position_ids = batch
        timers("batch-generator").stop()
        output_tensor = model(tokens, position_ids, attention_mask,
                             labels=labels)

    return output_tensor, partial(loss_func, model.training, batch)
forward_step.first_batch_printed = False

##
# Main
##


def extra_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='validation set')
    group.add_argument("--model_name",
                       choices={"gpt", "llama", "falcon", "llama2", "codellama", "mistral", "multimodal_mistral"},
                       default="gpt")
    group.add_argument("--model_type", choices={"encoder_or_decoder", "encoder_and_decoder"},
                       default="encoder_or_decoder")
    group.add_argument("--loss_role", choices={"assistant", "user", "all"},
                       default="assistant")
    group.add_argument("--no_loss_beyond_token_id", type=int, default=None)
    group.add_argument("--no_loss_on_token_ids", type=str, default=None)

    group.add_argument("--data_type", choices={"gpt", "instruction", "multimodal_instruction"},
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
    elif args.data_type == "multimodal_instruction":
        return_attention_mask_in_length = args.packed_input and args.use_flash_attn
        collate_fn = partial(
            multimodal_instruction_collator,
            scalar_loss_mask=args.scalar_loss_mask,
            return_attention_mask_in_length=return_attention_mask_in_length,
            loss_role=args.loss_role,
            no_loss_beyond_token_id=args.no_loss_beyond_token_id,
            no_loss_on_token_ids=list(map(int, args.no_loss_on_token_ids.split(","))) if args.no_loss_on_token_ids else []
        )
        if args.no_loss_beyond_token_id:
            print_rank_0(f"Loss will not be calculated beyond token id {args.no_loss_beyond_token_id}")
        if args.no_loss_on_token_ids:
            print_rank_0(f"Loss will not be calculated for token id {args.no_loss_on_token_ids}")
        print_rank_0(f"Loss role set to {args.loss_role} for multimodal-instruction dataset")
        print_rank_0("NOTE: You can still use this for pre-training, as long as you set the loss role to 'assistant' for every token")
    else:
        return_attention_mask_in_length = args.packed_input and args.use_flash_attn
        collate_fn = partial(
            instruction_collator,
            scalar_loss_mask=args.scalar_loss_mask,
            return_attention_mask_in_length=return_attention_mask_in_length,
            loss_role=args.loss_role,
            vision_patch_size=args.vision_patch_size,
        )
        print_rank_0(f"Loss role set to {args.loss_role} for instruction dataset")


    pretrain(args, data_provider, model_provider,  ModelType.encoder_or_decoder,
             forward_step, collate_fn=collate_fn)
    print(f"Done {dt.datetime.now(dt.timezone.utc)}")
