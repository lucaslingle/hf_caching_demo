# todo: maybe do the token shift inside model to reduce host-to-device io

from typing import Optional, Iterator, Dict

import datasets as hfds
import transformers as hftr
import numpy as np
# import jax
import math
import logging

hfds.disable_caching()
SPLITS = ["train", "validation", "test"]
PINDEX = 0
PCOUNT = 1


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
) -> Iterator[Dict[str, np.ndarray]]:
    hfds.disable_caching()

    # get tokenizer info
    assert hftr_tokenizer.is_fast
    bos_id = hftr_tokenizer.bos_token_id

    # we'll always use a custom split of the training set for out train/validation/test
    source_split = "train"
    hfds_split_name = source_split

    # load the dataset
    ds = hfds.load_dataset(
        path=hfds_identifier,
        config=hfds_config,
        split=hfds_split_name,
        # keep_in_memory=True,  # on tpu vm, disk < cpu ram. todo: whatif ram too small?
        streaming=True,
    )

    # shard by host, drop remainder
    # pcount = jax.process_count()
    # pindex = jax.process_index()
    pcount = PCOUNT
    pindex = PINDEX
    full_len = ds.info.splits.get(source_split).num_examples  # todo: edit
    val_len = math.floor(0.05 * full_len)
    if split_name == "validation":
        ds = ds.take(val_len)
    elif split_name == "test":
        ds = ds.skip(val_len).take(val_len)
    elif split_name == "train":
        ds = ds.skip(2 * val_len)
    else:
        raise ValueError("Unrecognized split name")

    split_len = full_len - 2 * val_len if split_name == "train" else val_len
    assert split_len > 0
    assert split_len > batch_size
    shard_len = split_len // pcount  # shard across hosts
    shard_len = (shard_len // batch_size) * batch_size  # drop any partial batch
    ds = ds.skip(pindex * shard_len).take(shard_len)  # skip to shard, drop rest

    # skip to current batch within shard, for reproducibility
    # note we are currently not manually shuffling each epoch.
    # this should be fine if there is only one epoch.
    offset = (step * batch_size) % shard_len
    ds = ds.skip(offset)

    # tokenize only the relevant stuff
    def tokenize(examples):
        targets = hftr_tokenizer(
            examples[hfds_datacol],
            padding="max_length",
            truncation=True,
            max_length=sequence_len,
        )["input_ids"]
        # inputs = [[bos_id, *e[0:-1]] for e in targets]
        # return {"inputs": inputs, "targets": targets}
        return {"targets": targets}

    ds = ds.map(
        tokenize,
        batched=True,
        batch_size=hfds_buffer_size,
        remove_columns=list(ds.column_names),
    )

    # convert to iterator, batch examples to the desired batch size per host.
    logging.info(f"calling Dataset.iter to make iterator of batches...")
    ds = ds.iter(batch_size=batch_size, drop_last_batch=True)
    logging.info(f"calling map(ds_iter) to get numpy arrays...")
    ds = map(
        lambda r: {
            # "inputs": np.array(r["inputs"], dtype=np.int32),
            "targets": np.array(r["targets"], dtype=np.int32),
            # "loss_mask": np.cumprod(
            #     np.pad(
            #         np.not_equal(r["inputs"], hftr_tokenizer.eos_token_id)[:, 1:],
            #         pad_width=((0, 0), (1, 0)),
            #         constant_values=True,
            #     ).astype(np.int32),
            #     axis=-1,
            # ),  # mask out every timestep once the input is eos, except at seq start
        },
        ds,
    )
    return ds
