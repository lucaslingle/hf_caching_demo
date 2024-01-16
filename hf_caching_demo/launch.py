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
    parser.add_argument("--gc_storage_uri", type=str, help="Google Cloud storage path")
    args = parser.parse_args()
    storage_options = dict(project=args.gc_project)
    fs = gcsfs.GCSFileSystem(**storage_options)
    ds_all_splits = datasets.load_dataset(args.hfds_identifier)
    tokenizer_ = transformers.GPT2TokenizerFast.from_pretrained(
        pretrained_model_name_or_path="gpt2",
        pad_token=transformers.GPT2TokenizerFast.from_pretrained("gpt2").eos_token,
    )

    def tokenize_fast(examples):
        return tokenizer_(
            examples[TEXTCOL],
            padding="max_length",
            truncation=True,
            max_length=SEQLEN,
        )

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
