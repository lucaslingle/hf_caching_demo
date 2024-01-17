# todo: shift inside model to reduce io

from typing import Optional
import argparse

import jax
import datasets as hfds
import transformers as hftr
import numpy as np

hfds.disable_caching()
STEP = 0
BATCH_SIZE = 8
SEQLEN = 512
TEXTCOL = "text"
SPLITS = ["train", "validation", "test"]


def get_tokenizer(
    cls_name: str,
    short_name: Optional[str] = None,
    pad_token: Optional[str] = None,
) -> hftr.PreTrainedTokenizerFast:
    # get class
    cls = getattr(hftr, cls_name)
    # instantiate class
    kwargs = dict(pad_token=pad_token) if pad_token is not None else dict()
    if short_name is not None:
        obj = cls.from_pretrained(short_name, **kwargs)
    else:
        try:
            short_name, *_ = cls_name.lower().split("tokenizer")
            obj = cls.from_pretrained(short_name, **kwargs)
        except Exception as e:
            raise NotImplementedError(f"Got exception {e}.")
    # grab eos token, for consistency of data pipeline always use it for padding
    if pad_token is None:
        assert obj.eos_token_id is not None
        obj = get_tokenizer(cls_name, short_name, pad_token=obj.eos_token)
    assert obj.is_fast
    return obj


def get_dataset(
    hfds_identifier: str,
    hfds_config: Optional[str],
    hfds_datacol: str,
    hfds_buffer_size: int,
    hftr_tokenizer: hftr.PreTrainedTokenizerFast,
    split_name: str,
    batch_size: int,
    sequence_len: int,
    step: int,
) -> hfds.Dataset:
    hfds.disable_caching()

    # get tokenizer info
    assert hftr_tokenizer.is_fast
    bos_id = hftr_tokenizer.bos_token_id

    # determine the dataset split to use
    hfds_splits_set = set(hfds.get_dataset_split_names(hfds_identifier))
    if hfds_splits_set != set(SPLITS):
        if split_name == "train":
            split_name = "train[0%:90%]"
        elif split_name == "validation":
            split_name = "train[90%:95%]"
        elif split_name == "test":
            split_name = "train[95%:100%]"
        else:
            raise NotImplementedError
    else:
        assert split_name in hfds_splits_set

    # load the dataset
    ds = hfds.load_dataset(
        path=hfds_identifier,
        config=hfds_config,
        split=split_name,
        keep_in_memory=True,  # on tpu vm, disk < cpu ram. todo: whatif ram too small?
    )

    # shard by host, drop remainder
    pcount = jax.process_count()
    pindex = jax.process_index()
    full_len = len(ds)
    shard_len = full_len // pcount
    ds = ds.select(range(pindex * shard_len, (pindex + 1) * shard_len))

    # skip to current batch for reproducibility
    # note we are currently not manually shuffling each epoch.
    # this should be fine if there is only one epoch.
    offset = (step * batch_size) % shard_len
    ds = ds.select(range(offset, shard_len))

    # tokenize only the relevant stuff
    def tokenize(examples):
        targets = hftr_tokenizer(
            examples[hfds_datacol],
            padding="max_length",
            truncation=True,
            max_length=sequence_len,
        )["input_ids"]
        inputs = [[bos_id, *e[0:-1]] for e in targets]
        return {"inputs": inputs, "targets": targets}

    ds = ds.map(
        tokenize,
        batched=True,
        batch_size=hfds_buffer_size,
        remove_columns=list(ds.column_names),
    )
    return ds


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

    # convert to iterator, batch examples to the desired batch size per host.
    print(f"calling Dataset.iter to make iterator of batches...")
    ds_iter = ds.iter(batch_size=BATCH_SIZE, drop_last_batch=True)
    print(f"calling map(ds_iter) to get numpy arrays...")
    ds_iter = map(
        lambda r: {
            "inputs": np.array(r["inputs"], dtype=np.int32),
            "targets": np.array(r["targets"], dtype=np.int32),
            "loss_mask": np.cumprod(
                np.pad(
                    np.not_equal(r["inputs"], tokenizer.eos_token_id)[:, 1:],
                    pad_width=((0, 0), (1, 0)),
                    constant_values=True,
                ).astype(np.int32),
                axis=-1,
            ),  # mask out every timestep once the input is eos, except at seq start
        },
        ds_iter,
    )
    return ds_iter


if __name__ == "__main__":
    batch = next(main())
    print(batch)
