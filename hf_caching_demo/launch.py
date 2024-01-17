# todo: maybe do the token shift inside model to reduce host-to-device io

import argparse

import datasets as hfds

from dataloader import get_tokenizer, get_dataset

hfds.disable_caching()
STEP = 0
BATCH_SIZE = 8
SEQLEN = 512
TEXTCOL = "text"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hfds_identifier", type=str, help="HF datasets identifier")
    parser.add_argument("--hfds_split_name", type=str, help="HF datasets split name")
    parser.add_argument("--gc_project", type=str, help="Google Cloud project")
    parser.add_argument("--gc_storage_uri", type=str, help="Google Cloud storage path")
    args = parser.parse_args()

    print(f"calling get_tokenizer to get fast tokenizer...")
    tokenizer = get_tokenizer("GPT2TokenizerFast", "gpt2")

    print(f"calling get_dataset to get sharded bitwise reproducible dataset...")
    ds = get_dataset(
        hfds_identifier=args.hfds_identifier,
        hfds_config=None,
        hfds_datacol=TEXTCOL,
        hfds_buffer_size=1024,
        hftr_tokenizer=tokenizer,
        split_name=args.hfds_split_name,
        batch_size=BATCH_SIZE,
        sequence_len=SEQLEN,
        step=STEP,
    )
    print(next(ds))


if __name__ == "__main__":
    main()
