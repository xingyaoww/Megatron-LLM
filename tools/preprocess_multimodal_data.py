"""Processing data for multi-modal {instruction tuning, pre-training}."""

import sys
import json
import time
import itertools
import numpy as np
from pathlib import Path
from typing import Optional, Iterable
from multiprocessing import Pool
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from collections import defaultdict
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
NON_VISION_TOKEN = -1

class Encoder(object):
    tokenizer: Optional[AbstractTokenizer] = None

    def __init__(self, args: Namespace):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = build_tokenizer(self.args)
        
        Encoder.vision_start_token = Encoder.tokenizer.vocab[self.args.vision_start_token]
        Encoder.vision_patch_token = Encoder.tokenizer.vocab[self.args.vision_patch_token]
        Encoder.vision_row_sep_token = Encoder.tokenizer.vocab[self.args.vision_row_sep_token]
        Encoder.vision_frame_sep_token = Encoder.tokenizer.vocab[self.args.vision_frame_sep_token]
        Encoder.vision_end_token = Encoder.tokenizer.vocab[self.args.vision_end_token]

    def encode(self, line: str) -> tuple[int, list[int], list[int], np.ndarray]:
        # get data
        assert Encoder.tokenizer is not None

        data = json.loads(line)

        # each line can be:
        if isinstance(data, list):
            # 1 conversations: [{"role": "human", "content": [{"type": "text", "text": "XXX"}]}]
            conversations = data
        elif isinstance(data, dict) and "conversations" in data:
            # 2 a dict with "id": {"id": "XXX", "conversations": [{"role": "human", "content": [{"type": "text", "text": "XXX"}]}]}
            assert isinstance(data, dict), "Data must be a list or a dictionary."
            # _id = data["id"]
            conversations = data["conversations"]
        elif isinstance(data, dict) and "content" in data and "role" in data:
            # 3 a dict with {"role": "human", "content": [{"type": "text", "text": "XXX"}]}
            conversations = [data]
        else:
            raise ValueError(f"Unknown data format: {data}")

        # tokenize and get roles
        tokens = []
        roles = []
        vision_patch_indices = [] # same shape as tokens, NON_VISION_TOKEN=-1 for non-vision tokens
        vision_patches = [] # list of C*PATCH_H*PATCH_W = 3*32*32 = 3072
        n_images = 0

        for turn in conversations:
            role = turn["role"]
            assert isinstance(turn["content"], list), "Content must be a list for multimodal data."

            # add format prefix/suffix if not pre-training
            if not self.args.do_pretrain:
                prefix_tokens = Encoder.tokenizer.tokenize(MESSAGE_PREFIX.format(role=role))
                tokens += prefix_tokens
                roles += [Role[role].value]*len(prefix_tokens)
                vision_patch_indices += [NON_VISION_TOKEN]*len(tokenized_text)

            for item in turn["content"]:
                if item["type"] == "text":
                    tokenized_text = Encoder.tokenizer.tokenize(item["text"])
                    tokens += tokenized_text
                    roles += [Role[role].value]*len(tokenized_text)
                    vision_patch_indices += [NON_VISION_TOKEN]*len(tokenized_text)
                
                elif item["type"] == "image_url":
                    # load image
                    img_content = item["image_url"]["url"]
                    assert "base64" in img_content, "Only base64 image is currently supported."
                    img_pil = load_base64_to_PILImage(img_content)
                    width, height = img_pil.size
                    new_width, new_height = get_resize_output_image_size((width, height))
                    img_tensor = get_transform(new_height, new_width)(img_pil)
                    cur_vision_patches = preprocess_image(img_tensor)
                    
                    # -> (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
                    n_h_patches, n_w_patches, c, patch_h, patch_w = cur_vision_patches.shape
                    n_patches = n_h_patches * n_w_patches
                    # flatten the patches -> (N_PATCHES, C*PATCH_H*PATCH_W)
                    cur_vision_patches = cur_vision_patches.view(n_patches, -1)
                    n_images += 1

                    # add separator tokens to patches per row
                    dummy_vision_tokens = []
                    dummy_roles = []
                    cur_patch_indices = []
                    for i in range(n_h_patches):
                        for j in range(n_w_patches):
                            if i != 0 and j == 0:  # when new row starts
                                dummy_vision_tokens.append(Encoder.vision_row_sep_token)
                                dummy_roles.append(Role[role].value)  # separator token has the same role as the image
                                cur_patch_indices.append(NON_VISION_TOKEN)

                            dummy_vision_tokens.append(Encoder.vision_patch_token)
                            dummy_roles.append(Role.image.value)
                            cur_patch_indices.append(len(vision_patches) + i * n_w_patches + j)

                    # Update data
                    tokens += [Encoder.vision_start_token] + dummy_vision_tokens + [Encoder.vision_end_token]
                    roles += [Role[role].value] + dummy_roles + [Role[role].value]
                    
                    vision_patch_indices += [NON_VISION_TOKEN] + cur_patch_indices + [NON_VISION_TOKEN]
                    vision_patches.extend(cur_vision_patches.numpy().astype(np.float16))

                else:
                    raise ValueError(f"Unknown content type (only 'text' and 'image_url' are supported): {item['type']}")

            if not self.args.do_pretrain:
                suffix_tokens = Encoder.tokenizer.tokenize(MESSAGE_SUFFIX)
                tokens += suffix_tokens
                roles += [Role[role].value]*len(suffix_tokens)

        assert len(vision_patches) == len(list(filter(lambda r: r == Role.image.value, roles))), \
            f"Number of image patches should be equal to the number of image tokens."
        # vision_patches = np.array(vision_patches)
        assert len(tokens) == len(vision_patch_indices)
        assert len(tokens) == len(roles)
        return len(line), tokens, roles, vision_patches, vision_patch_indices, n_images

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
        self.bin_fname = f"{prefix}-{feature}.bin.lz4" # SAVE AS LZ4
        self.idx_fname = f"{prefix}-{feature}.idx"
        self.builder = None

    def add_item(self, list_of_image_patches: list[np.ndarray]):
        self.builder.add_item(torch.HalfTensor(list_of_image_patches))

    def __enter__(self):
        self.builder = MMapIndexedDatasetBuilder(
            self.bin_fname,
            dtype=np.float16,
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
    group.add_argument('--workers', type=int, default=4,
                       help='Number of worker processes to launch')
    group.add_argument('--chunk_size', type=int, default=32,
                       help='Chunk size assigned to each worker process')
    group.add_argument('--no_mp', action='store_true',
                       help='Disable multiprocessing')
    group.add_argument('--log_interval', type=int, default=100,
                       help='Interval between progress updates')
    group.add_argument('--vision_start_token', type=str, default='<vision>')
    group.add_argument('--vision_patch_token', type=str, default='<vpatch>')
    group.add_argument('--vision_row_sep_token', type=str, default='<vrow_sep>')
    group.add_argument('--vision_frame_sep_token', type=str, default='<vframe_sep>')
    group.add_argument('--vision_end_token', type=str, default='</vision>')
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

def pack_docs(docs: Iterable, tokenizer, max_seq_length, stats_counter, keep_data_after_truncation=True):
    current_pack_tokens = []
    current_pack_roles = []
    current_pack_vision_patches = []
    current_pack_vision_patch_indices = []

    current_pack_n_images = 0
    current_seq_length = 0
    current_size = 0
    pbar = tqdm(desc="Packing documents")
    for size, tokens, roles, vision_patches, vision_patch_indices, n_images in docs:

        stats_counter["n_images"] += n_images

        # Check if adding the current text (with separator) will exceed max_seq_length
        if current_seq_length + len(tokens) + 1 <= max_seq_length:
            if current_pack_tokens:  # not the first sentence in pack
                # Add separator token
                current_pack_tokens.append(tokenizer.bos)
                current_pack_roles.append(Role.PACK_SEP.value)
                current_pack_vision_patch_indices.append(NON_VISION_TOKEN)
                
                # increment `vision_patch_indices` by the number of existing visual patches in the pack
                vision_patch_indices = [
                    idx if idx == NON_VISION_TOKEN else idx + len(current_pack_vision_patches)
                    for idx in vision_patch_indices
                ]

                current_seq_length += 1
                current_size += len(tokenizer._inv_special_tokens[tokenizer.bos])

            current_pack_tokens.extend(tokens)
            current_pack_roles.extend(roles)
            current_pack_vision_patches.extend(vision_patches)
            current_pack_vision_patch_indices.extend(vision_patch_indices)

            current_pack_n_images += n_images
            current_seq_length += len(tokens)
            current_size += size

        elif current_seq_length == 0:
            # The only possible reason for this is len(tokens) >= max_seq_length 
            assert len(current_pack_tokens) == len(current_pack_roles) == 0
            assert len(tokens) >= max_seq_length
            assert len(tokens) == len(roles)
            assert len(tokens) == len(vision_patch_indices)
            # We truncate the tokens to max_seq_length AND treat it as a single pack
            # packed_docs.append((size, tokens[:max_seq_length], roles[:max_seq_length]))
            stats_counter["n_total_packed_doc"] += 1
            stats_counter["n_total_tokens"] += len(current_pack_tokens)
            yield (
                size,
                tokens[:max_seq_length],
                roles[:max_seq_length],
                vision_patches, # no need to truncate vision patches since it is indexed by vision_patch_indices
                vision_patch_indices[:max_seq_length],
                n_images * (max_seq_length / len(tokens)) # use fraction of images to approximate the number of images
            )

            if keep_data_after_truncation:
                # Put the remaining tokens into the next pack
                current_pack_tokens = tokens[max_seq_length:]
                current_pack_roles = roles[max_seq_length:]
                current_pack_vision_patches = vision_patches # no need to truncate vision patches
                current_pack_vision_patch_indices = vision_patch_indices[max_seq_length:]

                current_pack_n_images = n_images * (len(current_pack_tokens) / len(tokens)) # use fraction of images to approximate the number of images
                current_seq_length = len(current_pack_tokens)
                current_size = size * (len(current_pack_tokens) / len(tokens))

        else:
            # Finish the current pack and start a new one
            assert len(current_pack_tokens) > 0
            assert current_size > 0
            assert len(current_pack_tokens) == len(current_pack_roles)
            assert len(current_pack_tokens) == len(current_pack_vision_patch_indices)
            # packed_docs.append((current_size, current_pack_tokens, current_pack_roles))
            stats_counter["n_total_packed_doc"] += 1
            stats_counter["n_total_tokens"] += len(current_pack_tokens)
            yield (
                current_size,
                current_pack_tokens,
                current_pack_roles,
                current_pack_vision_patches,
                current_pack_vision_patch_indices,
                current_pack_n_images
            )

            current_pack_tokens = tokens
            current_pack_roles = roles
            current_pack_vision_patches = vision_patches
            current_pack_vision_patch_indices = vision_patch_indices
            
            current_pack_n_images = n_images
            current_seq_length = len(tokens)
            current_size = size
        
        stats_counter["n_total_doc"] += 1
        pbar.update(1)
        # add status update
        pbar.set_postfix(stats_counter)
    pbar.close()

    # Add any remaining packed sequences
    if current_pack_tokens:
        assert len(current_pack_tokens) > 0, "Should not have empty pack."
        assert len(current_pack_tokens) == len(current_pack_roles)
        assert len(current_pack_tokens) == len(current_pack_vision_patch_indices)
        # packed_docs.append((current_size, current_pack_tokens, current_pack_roles))
        stats_counter["n_total_packed_doc"] += 1
        stats_counter["n_total_tokens"] += len(current_pack_tokens)
        yield (
            current_size,
            current_pack_tokens,
            current_pack_roles,
            current_pack_vision_patches,
            current_pack_vision_patch_indices,
            current_pack_n_images
        )
    
    print(f"Packed {stats_counter['n_total_doc']} documents into {stats_counter['n_total_packed_doc']} documents ({stats_counter['n_total_tokens']} tokens)")


def main():
    args = get_args()
    startup_start = time.time()

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    vocab_size = tokenizer.vocab_size
    
    # These tokens should already in the tokenizer
    assert args.vision_start_token in tokenizer.vocab
    assert args.vision_patch_token in tokenizer.vocab
    assert args.vision_row_sep_token in tokenizer.vocab
    assert args.vision_frame_sep_token in tokenizer.vocab
    assert args.vision_end_token in tokenizer.vocab


    if ".gz" in "".join(args.input):
        import gzip
        fs = map(gzip.open, args.input)
        print(f"Detected .gz file(s), will use gzip.open to read them.")
    elif ".lz4" in "".join(args.input):
        import lz4.frame
        fs = map(lz4.frame.open, args.input)
        print(f"Detected .lz4 file(s), will use lz4.frame.open to read them.")
    else:
        fs = map(open, args.input)

    output_jsonl = f"{args.output_prefix}.jsonl"
    with DatasetWriter(args.output_prefix, vocab_size, args.dataset_impl,
                          "text") as token_writer, \
            DatasetWriter(args.output_prefix, 16, args.dataset_impl,
                          "role") as role_writer, \
            ImageDatasetWriter(args.output_prefix, "vision_patch") as vision_patches_writer, \
            DatasetWriter(args.output_prefix, None, args.dataset_impl,
                          "vision_patch_indices") as vision_patch_indices_writer, \
            open(output_jsonl, "w") as output_file:

        f = itertools.chain(*fs)
        if not args.no_mp:
            print(f"Using multiprocessing with {args.workers} workers.")
            pool = Pool(args.workers, initializer=encoder.initializer)
            docs = pool.imap(encoder.encode, f, args.chunk_size)
        else:
            encoder.initializer()
            docs = (encoder.encode(i) for i in f)

        stats_counter = defaultdict(int)
        if args.do_packing:
            # make sure it works when docs is a generator
            # print(f"Sorting loaded documents by length for efficient packing. This can be slow for large dataset.")
            # docs = sorted(docs, key=lambda x: len(x[1]), reverse=True)
            docs = pack_docs(docs, tokenizer, args.max_seq_length, stats_counter)

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)

        for i, content in enumerate(docs, start=1):
            (size, tokens, roles, vision_patches, vision_patch_indices, n_images) = content
            total_bytes_processed += size
            if len(tokens) == 0:
                print("WARNING: Encountered empty document, skipping.")
                exit(1)
            assert size > 0
            assert len(tokens) == len(roles)
            assert len(tokens) > 0

            if len(vision_patches) == 0:
                # Add a dummy patch IF no vision patches are found
                # This is to ensure that the MMAP in the training code works
                vision_patches = [
                    np.zeros((3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                ]

            token_writer.add_item(tokens)
            role_writer.add_item(roles)
            vision_patches_writer.add_item(np.array(vision_patches))
            vision_patch_indices_writer.add_item(vision_patch_indices)
            stats = {
                "n_tokens": len(tokens),
                "n_image_tokens": sum(list(map(lambda r: r == Role.image.value, roles))),
                "n_images": n_images,
            }
            output_file.write(json.dumps(stats) + "\n")

            if i % args.log_interval == 0:
                elapsed = time.time() - proc_start
                mbs = total_bytes_processed/1024/1024/elapsed
                print(f"Processed {i} documents ({i/elapsed} docs/s, {mbs} MB/s).")
        print(f"Done processing {i} documents! Now finalizing.")

    # Save stats
    with open(f"{args.output_prefix}_stats.json", "w") as stats_file:
        stats = dict(
            total_bytes_processed=total_bytes_processed,
            total_docs_processed=i,
            total_time=time.time() - startup_start,
            total_time_startup=startup_end - startup_start,
            total_time_processing=time.time() - proc_start,
            **stats_counter
        )
        stats_file.write(json.dumps(stats))
    Path(f"{args.output_prefix}_DONE").touch()

    for f in fs:
        f.close()

    if not args.no_mp:
        pool.close()
        pool.join()
    
if __name__ == '__main__':
    main()
