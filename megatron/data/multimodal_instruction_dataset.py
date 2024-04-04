# Instruction code heavily inspired by Andreas Köpf
# source: https://github.com/andreaskoepf/epfl-megatron/tree/local_changes/
import time
from enum import IntEnum
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from megatron import get_args, get_tokenizer, print_rank_0
from megatron.data.indexed_dataset import make_dataset
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import (
    get_train_valid_test_split_,
    get_datasets_weights_and_num_samples
)
from megatron.data.instruction_dataset import Role

# imported from megatron.data.instruction_dataset
# class Role(IntEnum):
#     system = 0
#     user = 1
#     assistant = 2
#     image = 3
#     PACK_SEP = 1000  # This is used to separate two conversations packed together in to one sample


class MultimodalInstructionDataset(Dataset):
    def __init__(self, name: str, sample_indices: np.ndarray,
                 indexed_datasets: dict[str, Dataset], seq_length: int):

        self.indexed_text = indexed_datasets["text"]
        self.indexed_role = indexed_datasets["role"]
        self.indexed_vision_patch_indices = indexed_datasets["vision_patch_indices"]
        self.indexed_vison_patch = indexed_datasets["vision_patch"]

        # validate indices
        assert np.min(sample_indices) >= 0
        assert np.max(sample_indices) < len(self.indexed_text)
        assert len(self.indexed_text) == len(self.indexed_role)

        self.name = name
        self.sample_indices = sample_indices
        self.seq_length = seq_length

    def __len__(self) -> int:
        return self.sample_indices.shape[0]

    def __getitem__(self, idx) -> dict:
        # Get the shuffled index.
        idx = self.sample_indices[idx]
        text = self.indexed_text.get(idx)
        role = self.indexed_role.get(idx)
        vision_patch_indices = self.indexed_vision_patch_indices.get(idx)
        vision_patch = self.indexed_vison_patch.get(idx)
        assert text is not None and role is not None and text.shape == role.shape
        assert vision_patch_indices is not None and vision_patch is not None
        assert vision_patch_indices.shape == text.shape
        return {
            "text": text.astype(np.int64),
            "role": role.astype(np.int64),
            "vision_patch_indices": vision_patch_indices.astype(np.int64),
            "vision_patch": vision_patch.astype(np.float32)
        }


def _build_dataset_kernel(
    dataset_name: str,
    data_prefix,
    data_impl: str,
    num_samples: int,
    seq_length: int,
    seed: int,
    skip_warmup: bool,
) -> MultimodalInstructionDataset:
    """
    Build dataset. This method is called when individual
    train, valid, test datasets are provided
    """

    # Indexed dataset.
    indexed_datasets = get_indexed_datasets_(data_prefix, data_impl, skip_warmup)

    total_num_of_documents = len(indexed_datasets["text"])

    print_rank_0("    {}:".format(dataset_name))
    print_rank_0(
        "     document indices in [0, {}) total of {} "
        "documents".format(total_num_of_documents, total_num_of_documents)
    )

    documents = np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32)
    np_rng = np.random.RandomState(seed=seed)
    dataset = _sample_dataset(
        np_rng, documents, indexed_datasets, dataset_name, num_samples, seq_length
    )

    return dataset


def _build_dataset(
    dataset_name: str,
    data_prefix,
    data_impl: str,
    num_samples: int,
    seq_length: int,
    seed: int,
    skip_warmup: bool,
):
    dataset = None
    if len(data_prefix) == 1:
        dataset = _build_dataset_kernel(
            dataset_name,
            data_prefix[0],
            data_impl,
            num_samples,
            seq_length,
            seed,
            skip_warmup,
        )
    else:
        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        prefixes, weights, dataset_num_samples = output

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_dataset_kernel(
                dataset_name,
                prefixes[i],
                data_impl,
                dataset_num_samples[i],
                seq_length,
                seed,
                skip_warmup,
            )
            if ds:
                datasets.append(ds)

        if datasets:
            dataset = BlendableDataset(datasets, weights)
    return dataset



def get_indexed_datasets_(data_prefix: str, data_impl: str,
                          skip_warmup: bool) -> dict[str, Dataset]:
    print_rank_0(" > building dataset index ...")
    start_time = time.time()
    indexed_text = make_dataset(f"{data_prefix}-text", data_impl, skip_warmup)
    indexed_role = make_dataset(f"{data_prefix}-role", data_impl, skip_warmup)
    indexed_vision_patch = make_dataset(f"{data_prefix}-vision_patch", data_impl, skip_warmup)
    indexed_vision_patch_indices = make_dataset(f"{data_prefix}-vision_patch_indices", data_impl, skip_warmup)
    assert indexed_text is not None
    assert indexed_role is not None
    assert indexed_vision_patch is not None
    assert indexed_vision_patch_indices is not None

    print_rank_0(" > finished creating indexed dataset in "
                 f"{time.time() - start_time:4f} seconds")
    num_docs = len(indexed_text)
    print_rank_0("    number of documents: {}".format(num_docs))
    indices = np.arange(start=0, stop=num_docs, step=1, dtype=np.int32)
    n_tokens = np.sum(indexed_text.sizes[indices])
    print_rank_0("    number of tokens: {}".format(n_tokens))
    n_patch_indices = np.sum(indexed_vision_patch_indices.sizes[indices])
    print_rank_0("    number of vision patch indices (should be the same as # tokens): {}".format(n_patch_indices))
    n_patches = np.sum(indexed_vision_patch.sizes[indices]) // (32 * 32 * 3)
    print_rank_0("    number of vision patches: {}".format(n_patches))

    return {"text": indexed_text, "role": indexed_role, "vision_patch": indexed_vision_patch, "vision_patch_indices": indexed_vision_patch_indices}


def _sample_dataset(np_rng: np.random.RandomState, document_indices: np.ndarray,
                    indexed_datasets: dict[str, Dataset], name: str,
                    num_samples: int, seq_length: int) -> Optional[MultimodalInstructionDataset]:
    """Compute randomized index of samples for all epochs (num_samples)"""
    assert num_samples > 0

    remaining = num_samples
    index_list = []
    while remaining > 0:
        count = min(remaining, len(document_indices))
        index_list.append(np_rng.permutation(document_indices)[:count])
        remaining -= count
    sample_indices = np.concatenate(index_list)

    dataset = MultimodalInstructionDataset(
        name, sample_indices, indexed_datasets, seq_length
    )
    return dataset


def _build_train_valid_test_datasets(data_prefix, data_impl: str, splits_string: str,
                                     train_valid_test_num_samples: list[int],
                                     seq_length: int, seed: int, skip_warmup: bool):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_datasets = get_indexed_datasets_(data_prefix, data_impl, skip_warmup)
    total_num_of_documents = len(indexed_datasets["text"])
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(" > dataset split:")
    for index, name in enumerate(["train", "validation", "test"]):
        print_rank_0(f"    {name}")
        print_rank_0(f"    document indices in [{splits[index]}, {splits[index + 1]})"
                     f" total of {splits[index + 1] - splits[index]}")

    # generate random permutation of documents
    np_rng = np.random.RandomState(seed=seed)
    document_indices = np_rng.permutation(total_num_of_documents)

    datasets = {}
    for index, name in enumerate(["train", "validation", "test"]):
        begin, end = splits[index], splits[index + 1]
        if end <= begin:
            datasets[name] = None
        else:
            split_subset = document_indices[begin:end]
            num_samples = train_valid_test_num_samples[index]
            datasets[name] = _sample_dataset(np_rng, split_subset, indexed_datasets,
                                             name, num_samples, seq_length)

    return datasets["train"], datasets["validation"], datasets["test"]


# TODO: somewhat similar to gpt_dataset._build_train_valid_test_datasets, could we merge them?
def build_train_valid_test_datasets(data_prefix: Optional[str],
        data_impl: str,
        splits_string: str,
        train_valid_test_num_samples: list[int],
        seq_length: int,
        seed: int,
        skip_warmup: bool,
        train_data_prefix=None,
        valid_data_prefix=None,
        test_data_prefix=None,
    ):

    """Build train, valid, and test datasets."""
    if data_prefix:
        print_rank_0("Single data path provided for train, valid & test")
        # Single dataset.
        if len(data_prefix) == 1:
            return _build_train_valid_test_datasets(
                data_prefix[0],
                data_impl,
                splits_string,
                train_valid_test_num_samples,
                seq_length,
                seed,
                skip_warmup,
            )
        # Blending dataset.
        # Parse the values.
        (
            prefixes,
            weights,
            datasets_train_valid_test_num_samples,
        ) = get_datasets_weights_and_num_samples(
            data_prefix, train_valid_test_num_samples
        )

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                prefixes[i],
                data_impl,
                splits_string,
                datasets_train_valid_test_num_samples[i],
                seq_length,
                seed,
                skip_warmup,
            )
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        # Blend.
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = BlendableDataset(train_datasets, weights)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = BlendableDataset(valid_datasets, weights)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = BlendableDataset(test_datasets, weights)

        return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)
    else:
        print_rank_0(
            "Separate data paths provided for train, valid & test. Split string will be ignored."
        )
        train_dataset, valid_dataset, test_dataset = None, None, None
        # Single dataset.
        print_rank_0(" > train data path: {}".format(train_data_prefix))
        if train_data_prefix is not None:
            train_dataset = _build_dataset(
                "train",
                train_data_prefix,
                data_impl,
                train_valid_test_num_samples[0],
                seq_length,
                seed,
                skip_warmup,
            )

        print_rank_0(" > valid data path: {}".format(valid_data_prefix))
        if valid_data_prefix is not None:
            valid_dataset = _build_dataset(
                "valid",
                valid_data_prefix,
                data_impl,
                train_valid_test_num_samples[1],
                seq_length,
                seed,
                skip_warmup,
            )

        print_rank_0(" > test data path: {}".format(test_data_prefix))
        if test_data_prefix is not None:
            test_dataset = _build_dataset(
                "test",
                test_data_prefix,
                data_impl,
                train_valid_test_num_samples[2],
                seq_length,
                seed,
                skip_warmup,
            )
        return train_dataset, valid_dataset, test_dataset


def round_to_multiple_of(x: int, y: int) -> int:
        return ((x + y - 1) // y) * y


# Heavily inspired by Andreas Köpf: https://github.com/andreaskoepf/epfl-megatron/tree/local_changes/
def get_attention_mask_and_position_ids(data, attention_mask, example_ids):
    """
    Constructs causal attention masks and position IDs for sequences, based on provided example IDs.

    The function creates a causal attention mask to ensure each token in a sequence only attends 
    to previous tokens and itself. When sequences are packed, the attention mask also ensures 
    that tokens from one sequence do not attend to tokens from a subsequent packed sequence. 

    Additionally, position IDs are generated such that they reset for each new example in the packed sequences.

    Args:
    - data (torch.Tensor): Input data tensor of shape (batch_size, seq_length).
    - attention_mask (torch.Tensor): Initial attention mask of shape (batch_size, seq_length) where
                                     values close to 1 indicate tokens and values close to 0 indicate padding.
    - example_ids (torch.Tensor): Tensor of shape (batch_size, seq_length) indicating the IDs of packed examples.

    Returns:
    - attention_mask (torch.Tensor): Updated binary attention mask of shape (batch_size, 1, seq_length, seq_length).
    - position_ids (torch.Tensor): Position IDs tensor of shape (batch_size, seq_length) where IDs reset for each 
                                   new example in the packed sequences.
    """

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Expand example_ids for comparison
    expanded_example_ids = example_ids.unsqueeze(2).expand(micro_batch_size, seq_length, seq_length)
    
    # Create a comparison mask where each position is compared to every other position in the sequence
    comparison_mask = (expanded_example_ids == expanded_example_ids.transpose(1, 2)).float()

    # Attention mask based on example_ids
    causal_mask = torch.tril(comparison_mask).float()
    
    # Merge the two masks
    merged_mask = attention_mask.unsqueeze(2) * causal_mask

    # Convert attention mask to binary, True entries will masked
    attention_mask = (merged_mask < 0.5).to(data.device)

    # Position ids. reset for each new example
    position_ids = torch.zeros_like(data, dtype=torch.long)
    for i in range(micro_batch_size):
        pos = 0
        for j in range(seq_length):
            position_ids[i, j] = pos
            pos += 1
            # Check if this token is the last one in an example
            if j < seq_length - 1 and example_ids[i, j] != example_ids[i, j+1]:
                pos = 0  # reset
    position_ids.to(data.device)

    return attention_mask, position_ids

def instruction_collator(
    data,
    scalar_loss_mask=0.0,
    return_attention_mask_in_length: bool = False,
    loss_role: str = "assistant",
    no_loss_beyond_token_id: int = None,
    vision_patch_size: int = 32,
):
    assert loss_role in ["assistant", "user", "all"]
    args = get_args()
    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad
    seq_len = args.seq_length

    if args.variable_seq_lengths:
        max_sample_length = max(len(x["text"]) for x in data)
        seq_len = min(args.seq_length, round_to_multiple_of(max_sample_length, 16))
    seq_len += 1  # +1 to get seq_len tokens after shifting (token[t+1] is label for token[t])

    # pad data to seq_len, create attention mask
    batch_size = len(data)
    # INPUTS
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    role = torch.full_like(attention_mask, -1)
    input = torch.full_like(attention_mask, pad_id)
    vision_patch_indices = torch.full_like(attention_mask, -1)
    vision_patches = [] # list of vision patches (this is dynamic, so we can't use torch.full_like)

    # For loss and example segmentation
    # 1 means optimize loss, 0 means no loss
    loss_mask = torch.full_like(attention_mask, scalar_loss_mask, dtype=torch.float)
    # example id for each token, used for packed sequences
    example_ids = torch.zeros_like(attention_mask)

    attention_mask_in_length = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for i, x in enumerate(data):
        t = x["text"]
        r = x["role"]
        # print(f"batch {i} text shape", t.shape)
        # text shape (32723,)
        cur_vision_patch_indices = x["vision_patch_indices"]
        # vision_patch shape (59006976,)
        cur_vision_patch = x["vision_patch"].reshape(
            -1,
            vision_patch_size * vision_patch_size * 3
        )
        # print("cur_vision_patch_indices shape", cur_vision_patch_indices.shape)

        l = len(t)

        # Increment cur_vision_patch_indices by the number of vision patches already seen
        # since we are appending vision patches to a list
        cur_vision_patch_indices += len(vision_patches)
        
        # print("seq_len", seq_len)
        # print("token len", l)
        if l < seq_len:
            attention_mask[i, l:] = 0
            input[i, :l] = torch.from_numpy(t)
            role[i, :l] = torch.from_numpy(r)
            vision_patch_indices[i, :l] = torch.from_numpy(cur_vision_patch_indices)
        else:
            input[i] = torch.from_numpy(t[:seq_len])
            role[i] = torch.from_numpy(r[:seq_len])

            vision_patch_indices[i] = torch.from_numpy(cur_vision_patch_indices[:seq_len])
            # # find value in cur_vision_patch_indices[:seq_len] that are not -1
            # # and use that to index into cur_vision_patch
            # indices_non_0 = torch.where(cur_vision_patch_indices != -1)
            # patch_indices_kept = cur_vision_patch_indices[indices_non_0]
            # vision_patches.extend(cur_vision_patch[indices])
            
        # note: we just append everything for simplicity
        # since the dataset are pre-packed, so it less likely to waste memory
        vision_patches.extend(cur_vision_patch)

        # Segmentation for packed sequences
        current_example_id = 0
        cur_count = 0
        for j in range(min(l, seq_len)):
            # Switch to a new example if we encounter a PACK_SEP token
            if role[i, j] == Role.PACK_SEP.value:
                # Add the count of tokens in the previous example
                attention_mask_in_length[i, current_example_id] = cur_count
                # Switch to the next example
                current_example_id += 1
                cur_count = 0 # reset

            example_ids[i, j] = current_example_id
            cur_count += 1

        # Check if j is the last token in the sequence
        # If so, subtract 1 from the current example's count
        if j == seq_len - 1:
            attention_mask_in_length[i, current_example_id] = cur_count - 1
        # Handle the case where the last token is not a PACK_SEP token
        else:
            attention_mask_in_length[i, current_example_id] = cur_count

    # Loss mask
    # - only calculate loss for loss role
    if loss_role == "all":
        loss_mask = torch.ones_like(attention_mask, dtype=torch.float)
    else:
        loss_role = Role[loss_role].value
        loss_mask[role == loss_role] = 1.0
    
    if no_loss_beyond_token_id:
        no_loss_beyond_token_id = int(no_loss_beyond_token_id)
        loss_mask[input >= no_loss_beyond_token_id] = 0.0

    # - completely ignore padding tokens
    loss_mask[input == pad_id] = 0.0

    # -- Previous handled by get_batch
    # Shift input to the right by one
    tokens = input[:, :-1].contiguous()
    attention_mask = attention_mask[:, :-1]
    assert torch.all(attention_mask_in_length[:, -1] == 0)
    attention_mask_in_length = attention_mask_in_length[:, :-1]
    example_ids = example_ids[:, :-1]
    attention_mask, position_ids = get_attention_mask_and_position_ids(
        tokens, attention_mask, example_ids
    )
    # convert to torch.int64
    attention_mask = attention_mask.to(torch.int64)

    # labels = input[:, 1:].contiguous() therefore we need to shift the loss_mask similarly
    loss_mask = loss_mask[:, 1:].contiguous()
    vision_patch_indices = vision_patch_indices[:, :-1]
    # aggregate vision patches
    vision_patches = torch.tensor(np.array(vision_patches), dtype=torch.float32)
    vision_patches = vision_patches.view(-1, vision_patch_size * vision_patch_size * 3)

    # print("vision_patches shape", vision_patches.shape)
    # print("vision_patch_indices shape", vision_patch_indices.shape)
    # print("attention_mask shape", attention_mask.shape)
    # print("position_ids shape", position_ids.shape)
    # print("vision_patches", vision_patches)
    # print("vision_patch_indices", vision_patch_indices)
    
    return {
        "text": input,
        "attention_mask": attention_mask if not return_attention_mask_in_length \
            else attention_mask_in_length,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "vision_patch_indices": vision_patch_indices,
        "vision_patches": vision_patches,
    }
