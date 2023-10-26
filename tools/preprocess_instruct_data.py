# Instruction code heavily inspired by Andreas Köpf
# source: https://github.com/andreaskoepf/epfl-megatron/tree/local_changes/
"""Processing data for instruction tuning.
Example:
python instruct/preprocess_instruct_data.py --input=/pure-mlo-scratch/alhernan/data/medmc/medmc-v1.jsonl \
    --output_prefix=/pure-mlo-scratch/alhernan/data/medmc/medmc-v1 \
    --tokenizer_type=SentencePieceTokenizer \
    --vocab_file=/pure-mlo-scratch/llama/tokenizer.model \
    --chunk_size=32 --workers=32 \
    --vocab_extra_ids_list "[bib_ref],[/bib_ref],[fig_ref],[/fig_ref],[bib],[/bib],[fig],[/fig],[table],[/table],[formula],[/formula],<|im_start|>,<|im_end|>" \
    --question_key=input \
    --answer_key=output \
    --system_key=instruction
"""

import sys
import json
import time
import itertools
from pathlib import Path
from typing import Optional
from multiprocessing import Pool
from argparse import ArgumentParser, Namespace

import torch

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from megatron.tokenizer import build_tokenizer
from megatron.tokenizer.tokenizer import AbstractTokenizer
from megatron.data.indexed_dataset import make_builder
from megatron.data.instruction_dataset import Role


class Encoder(object):
    tokenizer: Optional[AbstractTokenizer] = None

    def __init__(self, args: Namespace):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, line: str) -> tuple[int, list[int], list[int]]:
        # get data
        assert Encoder.tokenizer is not None
        data = json.loads(line)
        _id = data["id"]
        conversations = data["conversations"]
        # tokenize and get roles
        tokens = []
        roles = []
        for turn in conversations:
            role = turn["role"]
            message = format_message(turn["content"], role)
            message = Encoder.tokenizer.tokenize(message)
            tokens += message
            roles += [Role[role].value]*len(message)
        return len(line), tokens, roles

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


def format_message(message: str, role: str) -> str:
    return f"<|im_start|>{role}\n{message}<|im_end|>\n"


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
    group.add_argument('--vocab_extra_ids', type=int, default=0)
    group.add_argument('--vocab_extra_ids_list', type=str, default=None,
                       help='comma separated list of special vocab ids to add to the tokenizer')
    group.add_argument("--no_new_tokens", action="store_false", dest="new_tokens",
                       help=("Whether to add special tokens (e.g. CLS, MASK, etc) "
                             "in the sentencepiece tokenizer or not"))
    group.add_argument("--do_packing", action="store_true",
                       help=("Whether to pack documents into sequences of max_seq_length."))
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

def pack_docs(docs, tokenizer, max_seq_length):
    packed_docs = []

    current_pack_tokens = []
    current_pack_roles = []
    current_seq_length = 0
    current_size = 0

    for size, tokens, roles in docs:
        # Check if adding the current text (with separator) will exceed max_seq_length
        if current_seq_length + len(tokens) + 1 <= max_seq_length:
            if current_pack_tokens:  # not the first sentence in pack
                current_pack_tokens.append(tokenizer.bos)
                current_pack_roles.append(Role.PACK_SEP.value)
                current_seq_length += 1
                current_size += len(tokenizer._inv_special_tokens[tokenizer.bos])

            current_pack_tokens.extend(tokens)
            current_pack_roles.extend(roles)
            current_seq_length += len(tokens)
            current_size += size
        else:
            # Finish the current pack and start a new one
            packed_docs.append((current_size, current_pack_tokens, current_pack_roles))

            current_pack_tokens = tokens
            current_pack_roles = roles
            current_seq_length = len(tokens)
            current_size = size

    # Add any remaining packed sequences
    if current_pack_tokens:
        packed_docs.append((current_size, current_pack_tokens, current_pack_roles))
    
    print(f"Packed {len(docs)} documents into {len(packed_docs)} documents.")
    return packed_docs


def main():
    args = get_args()
    startup_start = time.time()

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    vocab_size = tokenizer.vocab_size
    fs = map(open, args.input)
    with Pool(args.workers, initializer=encoder.initializer) as pool, \
            DatasetWriter(args.output_prefix, vocab_size, args.dataset_impl,
                          "text") as token_writer, \
            DatasetWriter(args.output_prefix, 16, args.dataset_impl,
                          "role") as role_writer:

        f = itertools.chain(*fs)
        docs = pool.imap(encoder.encode, f, args.chunk_size)

        if args.do_packing:
            # make sure it works when docs is a generator
            print(f"Sorting loaded documents by length for efficient packing. This can be slow for large dataset.")
            docs = sorted(docs, key=lambda x: len(x[1]), reverse=True)
            docs = pack_docs(docs, tokenizer, args.max_seq_length)

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)

        for i, (size, tokens, roles) in enumerate(docs, start=1):
            total_bytes_processed += size
            token_writer.add_item(tokens)
            role_writer.add_item(roles)

            if i % args.log_interval == 0:
                elapsed = time.time() - proc_start
                mbs = total_bytes_processed/1024/1024/elapsed
                print(f"Processed {i} documents ({i/elapsed} docs/s, {mbs} MB/s).")
        print("Done! Now finalizing.")

    for f in fs:
        f.close()


if __name__ == '__main__':
    main()
