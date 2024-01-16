import argparse
import gcsfs
import datasets
import posixpath
import transformers

SEQLEN = 512
TEXTCOL = "text"


# to make an api key, follow instructions at
# https://developers.google.com/workspace/guides/create-credentials
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hfds_identifier", type=str, help="HF datasets identifier")
    parser.add_argument("--hfds_split_name", type=str, help="HF datasets split name")
    parser.add_argument("--gc_project", type=str, help="Google Cloud project")
    parser.add_argument("--gc_secret_key", type=str, help="Google Cloud api key")
    parser.add_argument("--gc_storage_uri", type=str, help="Google Cloud storage path")
    args = parser.parse_args()
    fs = gcsfs.GCSFileSystem(project=args.gc_project, token=None)
    storage_options = dict(project=args.gc_project, key=args.gc_secret_key)
    ds_all_splits = datasets.load_dataset(
        args.hfds_identifier,
        cache_dir=posixpath.join(args.gc_storage_uri, "staging"),
        storage_options=storage_options,
    )
    tokenizer_ = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

    def tokenize_fast(examples):
        return tokenizer_(
            examples[TEXTCOL],
            padding="max_length",
            truncation=True,
            max_length=SEQLEN,
        )["input_ids"]

    for s in ds_all_splits.keys():
        path_s = posixpath.join(args.gc_storage_uri, s)
        if not fs.exists(path_s):
            ds = ds_all_splits.get(s)
            ds = ds.map(tokenize_fast, batch_size=1024)
            ds.save_to_disk(path_s, storage_options=storage_options)

    return datasets.load_from_disk(
        dataset_path=posixpath.join(args.gc_storage_uri, args.hfds_split_name),
        storage_options=storage_options,
    )


if __name__ == "__main__":
    main()
