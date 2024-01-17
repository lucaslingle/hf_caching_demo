# todo: maybe do the token shift inside model to reduce host-to-device io

from typing import Optional, Iterator, Dict

import datasets as hfds
import transformers as hftr
import numpy as np
import jax

hfds.disable_caching()
SPLITS = ["train", "validation", "test"]
# PINDEX = 0
# PCOUNT = 1


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
    unittesting_shard_size: Optional[int] = None,  # small non-None for fast tests
) -> Iterator[Dict[str, np.ndarray]]:
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

    if unittesting_shard_size is not None:
        assert unittesting_shard_size % batch_size == 0
        split_name = f"train[0:{unittesting_shard_size}]"

    # load the dataset
    ds = hfds.load_dataset(
        path=hfds_identifier,
        config=hfds_config,
        split=split_name,
        keep_in_memory=True,  # on tpu vm, disk < cpu ram. todo: whatif ram too small?
    )

    # shard by host, drop remainder
    pcount = jax.process_count()  # PCOUNT
    pindex = jax.process_index()  # PINDEX
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

    # convert to iterator, batch examples to the desired batch size per host.
    print(f"calling Dataset.iter to make iterator of batches...")
    ds = ds.iter(batch_size=batch_size, drop_last_batch=True)
    print(f"calling map(ds_iter) to get numpy arrays...")
    ds = map(
        lambda r: {
            "inputs": np.array(r["inputs"], dtype=np.int32),
            "targets": np.array(r["targets"], dtype=np.int32),
            "loss_mask": np.cumprod(
                np.pad(
                    np.not_equal(r["inputs"], hftr_tokenizer.eos_token_id)[:, 1:],
                    pad_width=((0, 0), (1, 0)),
                    constant_values=True,
                ).astype(np.int32),
                axis=-1,
            ),  # mask out every timestep once the input is eos, except at seq start
        },
        ds,
    )
    return ds
