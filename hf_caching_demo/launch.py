import argparse
import gcsfs
import datasets
from etils import epath


# to make an api key, follow instructions at
# https://developers.google.com/workspace/guides/create-credentials
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hfds_identifier", type=str, help="HF datasets identifier")
    parser.add_argument("--gc_project", type=str, help="Google Cloud project")
    parser.add_argument("--gc_secret_key", type=str, help="Google Cloud api key")
    parser.add_argument("--gc_storage_uri", type=str, help="Google Cloud storage path")
    args = parser.parse_args()
    storage_options = dict(
        project=args.gc_project,
        key=args.gc_secret_key,
    )
    fs = gcsfs.GCSFileSystem(**storage_options)  # todo: do we need this line?

    # maybe cache the built dataset
    output_dir = (epath.Path(args.gc_storage_uri) / "staging").as_uri()
    builder = datasets.load_dataset_builder(args.hfds_identifier)
    builder.download_and_prepare(
        output_dir=output_dir,
        storage_options=storage_options,
    )
    # and maybe process further
    # todo: this
    # and maybe save the processed dataset
    # encoded_dataset.save_to_disk(
    #     "gcs://my-private-datasets/imdb/train",
    #     storage_options=storage_options,
    # )

    # load the processed dataset


if __name__ == "__main__":
    main()
