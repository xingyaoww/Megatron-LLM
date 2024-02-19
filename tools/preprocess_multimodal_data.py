"""Processing data for multi-modal {instruction tuning, pre-training}."""

import sys
import json
import time
import itertools
import numpy as np
from pathlib import Path
from typing import Optional
from multiprocessing import Pool
from argparse import ArgumentParser, Namespace

import torch

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from megatron.tokenizer import build_tokenizer
from megatron.tokenizer.tokenizer import AbstractTokenizer
from megatron.data.indexed_dataset import make_builder, MMapIndexedDatasetBuilder
from megatron.data.instruction_dataset import Role

# ========
# Handle Multimodal Images
import io
import torch
import base64
import torchvision.transforms as transforms
from math import ceil
from PIL import Image

PATCH_SIZE = 32
MAX_RESOLUTION = 1024 # 32 * 32

def get_resize_output_image_size(
    image_size,
) -> tuple:
    l1, l2 = image_size # the order of width/height should not matters
    short, long = (l2, l1) if l2 <= l1 else (l1, l2)

    # set the nearest multiple of PATCH_SIZE for `long`
    requested_new_long = min(
        [
            ceil(long / PATCH_SIZE) * PATCH_SIZE,
            MAX_RESOLUTION,
        ]
    )

    new_long, new_short = requested_new_long, int(requested_new_long * short / long)
    # Find the nearest multiple of 64 for new_short
    new_short = ceil(new_short / PATCH_SIZE) * PATCH_SIZE
    return (new_long, new_short) if l2 <= l1 else (new_short, new_long)


def preprocess_image(
    image_tensor: torch.Tensor,
    patch_size=PATCH_SIZE
) -> torch.Tensor:
    # Reshape the image to get the patches
    # shape changes: (C=3, H, W) 
    # -> (C, N_H_PATCHES, W, PATCH_H)
    # -> (C, N_H_PATCHES, N_W_PATCHES, PATCH_H, PATCH_W)
    patches = image_tensor.unfold(1, patch_size, patch_size)\
        .unfold(2, patch_size, patch_size)

    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    # -> (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
    return patches


def get_transform(height, width):
    preprocess_transform = transforms.Compose([
            transforms.Resize((height, width)),             
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                std=[0.229, 0.224, 0.225])   # standard deviation for pre-trained models on ImageNet
        ])
    return preprocess_transform

def get_reverse_transform():
    reverse_transform = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        transforms.ToPILImage()
    ])
    return reverse_transform

def load_image_to_base64(image_path: str) -> str:
    # convert image to jpeg, then to data:image/jpeg;base64,
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

def load_base64_to_PILImage(base64_string: str) -> Image:
    # convert data:image/jpeg;base64, to jpeg
    base64_string = base64_string.split(",")[1]
    decoded_string = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(decoded_string)).convert('RGB')

# ========

# def format_message(message: str, role: str) -> str:
#     return f"<|im_start|>{role}\n{message}<|im_end|>\n"
MESSAGE_PREFIX = "<|im_start|>{role}\n"
MESSAGE_SUFFIX = "<|im_end|>\n"

class Encoder(object):
    tokenizer: Optional[AbstractTokenizer] = None

    def __init__(self, args: Namespace):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = build_tokenizer(self.args)
        
        Encoder.image_start_token = Encoder.tokenizer.tokenize(self.args.image_start_token)
        assert len(Encoder.image_start_token) == 1
        Encoder.image_start_token = Encoder.image_start_token[0]
        
        Encoder.image_end_token = Encoder.tokenizer.tokenize(self.args.image_end_token)
        assert len(Encoder.image_end_token) == 1
        Encoder.image_end_token = Encoder.image_end_token[0]

    def encode(self, line: str) -> tuple[int, list[int], list[int], np.ndarray]:
        # get data
        assert Encoder.tokenizer is not None
        data = json.loads(line)
        _id = data["id"]
        conversations = data["conversations"]
        
        # tokenize and get roles
        tokens = []
        roles = []
        all_image_patches = [] # list of C*PATCH_H*PATCH_W = 3*32*32 = 3072

        for turn in conversations:
            role = turn["role"]
            assert isinstance(turn["content"], list), "Content must be a list for multimodal data."

            # add format prefix/suffix if not pre-training
            if not self.args.do_pretrain:
                prefix_tokens = Encoder.tokenizer.tokenize(MESSAGE_PREFIX.format(role=role))
                tokens += prefix_tokens
                roles += [Role[role].value]*len(prefix_tokens)

            # TODO: handle multimodal
            for item in turn["content"]:
                if item["type"] == "text":
                    tokenized_text = Encoder.tokenizer.tokenize(item["text"])
                    tokens += tokenized_text
                    roles += [Role[role].value]*len(tokenized_text)
                
                elif item["type"] == "image_url":
                    # load image
                    img_content = item["image_url"]["url"]
                    assert "base64" in img_content, "Only base64 image is currently supported."
                    img_pil = load_base64_to_PILImage(img_content)
                    width, height = img_pil.size
                    new_width, new_height = get_resize_output_image_size((width, height))
                    img_tensor = get_transform(new_height, new_width)(img_pil)
                    img_patches = preprocess_image(img_tensor)
                    
                    # -> (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
                    n_patches = img_patches.shape[0] * img_patches.shape[1]
                    # flatten the patches -> (N_PATCHES, C*PATCH_H*PATCH_W)
                    img_patches = img_patches.view(n_patches, -1)

                    # assign patch_idx (after insertion) and insert patches
                    patch_indices = [len(all_image_patches) + i for i in range(n_patches)]
                    all_image_patches.extend(img_patches.numpy())
                    # Add image tokens (use patch_idx to refer to the image patches)
                    tokens += [Encoder.image_start_token] + patch_indices + [Encoder.image_end_token]
                    # Set role of all patches to be role of 'image', except for the start token
                    roles += [Role[role].value] + [Role.image.value] * n_patches + [Role[role].value]
                else:
                    raise ValueError(f"Unknown content type (only 'text' and 'image_url' are supported): {item['type']}")

            if not self.args.do_pretrain:
                suffix_tokens = Encoder.tokenizer.tokenize(MESSAGE_SUFFIX)
                tokens += suffix_tokens
                roles += [Role[role].value]*len(suffix_tokens)

        assert len(all_image_patches) == len(list(filter(lambda r: r == Role.image.value, roles))), \
            "Number of image patches should be equal to the number of image tokens."
        all_image_patches = np.array(all_image_patches)
        return len(line), tokens, roles, all_image_patches

    @property
    def special_tokens(self) -> dict:
        return self.tokenizer._special_tokens


class DatasetWriter:
    def __init__(self, prefix: str, vocab_size: int, dataset_impl: str = "mmap",
                 feature: str = "text"):
        self.vocab_size = vocab_size
        self.dataset_impl = dataset_impl
        self.bin_fname = f"{prefix}-{feature}.bin"
        self.idx_fname = f"{prefix}-{feature}.idx"
        self.builder = None

    def add_item(self, tokens: list[int]):
        self.builder.add_item(torch.IntTensor(tokens))

    def __enter__(self):
        self.builder = make_builder(self.bin_fname, impl=self.dataset_impl,
                                    vocab_size=self.vocab_size)
        return self

    def __exit__(self, *_):
        self.builder.finalize(self.idx_fname)
        self.builder = None

class ImageDatasetWriter:
    def __init__(self, prefix: str, feature: str = "image_patch"):
        self.bin_fname = f"{prefix}-{feature}.bin"
        self.idx_fname = f"{prefix}-{feature}.idx"
        self.builder = None

    def add_item(self, tokens: list[np.ndarray]):
        self.builder.add_item(torch.FloatTensor(tokens))

    def __enter__(self):
        self.builder = MMapIndexedDatasetBuilder(
            self.bin_fname,
            dtype=np.float32
        )
        return self

    def __exit__(self, *_):
        self.builder.finalize(self.idx_fname)
        self.builder = None

def get_args():
    parser = ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, nargs="+",
                       help='Path(s) to input JSON file(s)')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer', 'FalconTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab_file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge_file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output_prefix', type=Path, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset_impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help='Number of worker processes to launch')
    group.add_argument('--chunk_size', type=int, required=True,
                       help='Chunk size assigned to each worker process')
    group.add_argument('--log_interval', type=int, default=100,
                       help='Interval between progress updates')
    group.add_argument('--image_start_token', type=str, default='<image>',
                       help='reserved token for image start')
    group.add_argument('--image_end_token', type=str, default='</image>',
                       help='reserved token for image end')
    group.add_argument('--vocab_extra_ids', type=int, default=0)
    group.add_argument('--vocab_extra_ids_list', type=str, default=None,
                       help='comma separated list of special vocab ids to add to the tokenizer')
    group.add_argument("--no_new_tokens", action="store_false", dest="new_tokens",
                       help=("Whether to add special tokens (e.g. CLS, MASK, etc) "
                             "in the sentencepiece tokenizer or not"))
    group.add_argument("--do_packing", action="store_true",
                       help=("Whether to pack documents into sequences of max_seq_length."))
    group.add_argument("--do_pretrain", action="store_true",
                       help=("Whether to format data for pretraining by removing all the chat format."))
    group.add_argument("--max_seq_length", type=int, default=4096)
    args = parser.parse_args()
    args.keep_empty = False

    if args.vocab_extra_ids_list is None:
        args.vocab_extra_ids_list = f"{args.image_start_token},{args.image_end_token}"
    else:
        args.vocab_extra_ids_list += f",{args.image_start_token},{args.image_end_token}"
    print(f"vocab_extra_ids_list: {args.vocab_extra_ids_list}")

    if args.do_packing:
        assert args.max_seq_length, "Must specify max_seq_length when packing documents."
        print(f"Packing documents into sequences of max_seq_length {args.max_seq_length}.")

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    return args

# def pack_docs(docs, tokenizer, max_seq_length):
#     packed_docs = []

#     current_pack_tokens = []
#     current_pack_roles = []
#     current_seq_length = 0
#     current_size = 0

#     for size, tokens, roles in docs:
#         # Check if adding the current text (with separator) will exceed max_seq_length
#         if current_seq_length + len(tokens) + 1 <= max_seq_length:
#             if current_pack_tokens:  # not the first sentence in pack
#                 current_pack_tokens.append(tokenizer.bos)
#                 current_pack_roles.append(Role.PACK_SEP.value)
#                 current_seq_length += 1
#                 current_size += len(tokenizer._inv_special_tokens[tokenizer.bos])

#             current_pack_tokens.extend(tokens)
#             current_pack_roles.extend(roles)
#             current_seq_length += len(tokens)
#             current_size += size
#         elif current_seq_length == 0:
#             # The only possible reason for this is len(tokens) >= max_seq_length 
#             assert len(current_pack_tokens) == len(current_pack_roles) == 0
#             assert len(tokens) >= max_seq_length
#             assert len(tokens) == len(roles)
#             # We truncate the tokens to max_seq_length AND treat it as a single pack
#             packed_docs.append((size, tokens[:max_seq_length], roles[:max_seq_length]))
#         else:
#             # Finish the current pack and start a new one
#             assert len(current_pack_tokens) > 0
#             assert len(current_pack_roles) > 0
#             assert current_size > 0
#             packed_docs.append((current_size, current_pack_tokens, current_pack_roles))

#             current_pack_tokens = tokens
#             current_pack_roles = roles
#             current_seq_length = len(tokens)
#             current_size = size

#     # Add any remaining packed sequences
#     if current_pack_tokens:
#         assert len(current_pack_tokens) > 0, "Should not have empty pack."
#         packed_docs.append((current_size, current_pack_tokens, current_pack_roles))
    
#     print(f"Packed {len(docs)} documents into {len(packed_docs)} documents.")
#     return packed_docs


def main():
    args = get_args()
    startup_start = time.time()

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    vocab_size = tokenizer.vocab_size

    if ".gz" in "".join(args.input):
        import gzip
        fs = map(gzip.open, args.input)
        print(f"Detected .gz file(s), will use gzip.open to read them.")
    else:
        fs = map(open, args.input)

    with Pool(args.workers, initializer=encoder.initializer) as pool, \
            DatasetWriter(args.output_prefix, vocab_size, args.dataset_impl,
                          "text") as token_writer, \
            DatasetWriter(args.output_prefix, 16, args.dataset_impl,
                          "role") as role_writer, \
            ImageDatasetWriter(args.output_prefix, "image_patch") as img_writer:

        f = itertools.chain(*fs)
        docs = pool.imap(encoder.encode, f, args.chunk_size)
        # encoder.initializer()
        # docs = [encoder.encode(i) for i in f]

        # if args.do_packing:
        #     # make sure it works when docs is a generator
        #     print(f"Sorting loaded documents by length for efficient packing. This can be slow for large dataset.")
        #     docs = sorted(docs, key=lambda x: len(x[1]), reverse=True)
        #     docs = pack_docs(docs, tokenizer, args.max_seq_length)

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)

        for i, (size, tokens, roles, image_patches) in enumerate(docs, start=1):
            total_bytes_processed += size
            if len(tokens) == 0:
                print("WARNING: Encountered empty document, skipping.")
                exit(1)
            assert size > 0
            assert len(tokens) == len(roles)
            assert len(tokens) > 0

            token_writer.add_item(tokens)
            role_writer.add_item(roles)
            img_writer.add_item(image_patches)

            if i % args.log_interval == 0:
                elapsed = time.time() - proc_start
                mbs = total_bytes_processed/1024/1024/elapsed
                print(f"Processed {i} documents ({i/elapsed} docs/s, {mbs} MB/s).")
        print("Done! Now finalizing.")

    for f in fs:
        f.close()


if __name__ == '__main__':
    main()
