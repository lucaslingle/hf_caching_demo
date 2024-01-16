# todo: shard by host, somehow

from typing import Optional
import argparse

import gcsfs
import datasets as hfds
import posixpath
import transformers as hftr
# import jax
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
    sequence_len: int,
) -> hfds.Dataset:
    # get shard info
    # pcount = jax.process_count()
    # pindex = jax.process_index()

    # get tokenizer info
    assert hftr_tokenizer.is_fast
    bos_id = hftr_tokenizer.bos_token_id

    # load dataset
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

    ds = hfds.load_dataset(
        hfds_identifier,
        hfds_config,
        split=split_name,
        streaming=False,
    )
    # shard by host, then tokenize the host's shard only
    assert "content_" not in set(ds.column_names)

    def shard_by_host(examples):
        # examples = examples[hfds_datacol]
        # examples = [e for i, e in enumerate(examples) if i % pcount == pindex]
        return {"content_": examples[hfds_datacol]}

    def tokenize(examples):
        targets = hftr_tokenizer(
            examples["content_"],
            padding="max_length",
            truncation=True,
            max_length=sequence_len,
        )["input_ids"]
        inputs = [[bos_id, *e[0:-1]] for e in targets]
        return {"inputs": inputs, "targets": targets}

    ds = ds.map(
        shard_by_host,
        batched=True,
        batch_size=hfds_buffer_size,  # * jax.process_count(),
        remove_columns=list(ds.column_names),
    )
    ds = ds.map(
        tokenize,
        batched=True,
        batch_size=hfds_buffer_size,
    )
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hfds_identifier", type=str, help="HF datasets identifier")
    parser.add_argument("--hfds_split_name", type=str, help="HF datasets split name")
    parser.add_argument("--gc_project", type=str, help="Google Cloud project")
    parser.add_argument("--gc_storage_uri", type=str, help="Google Cloud storage path")
    args = parser.parse_args()
    storage_options = dict(project=args.gc_project)
    fs = gcsfs.GCSFileSystem(**storage_options)

    tokenizer = get_tokenizer("GPT2TokenizerFast", "gpt2")
    for s in SPLITS:
        path_s = posixpath.join(args.gc_storage_uri, s)
        if not fs.exists(path_s):
            print(f"calling get_dataset for split {s}...")
            ds = get_dataset(
                hfds_identifier=args.hfds_identifier,
                hfds_config=None,
                hfds_datacol=TEXTCOL,
                hfds_buffer_size=1024,
                hftr_tokenizer=get_tokenizer("GPT2TokenizerFast", "gpt2"),
                split_name=s,
                sequence_len=SEQLEN,
            )
            ds.save_to_disk(path_s, storage_options=storage_options)

    print(f"calling hfds.load_from_disk for split {args.hfds_split_name}...")
    ds = hfds.load_from_disk(
        dataset_path=posixpath.join(args.gc_storage_uri, args.hfds_split_name),
        storage_options=storage_options,
    )
    print(f"calling Dataset.select to slice...")
    # https://discuss.huggingface.co/t/efficiently-slicing-dataset/28067
    ds = ds.select(range(STEP * BATCH_SIZE, len(ds)))

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
