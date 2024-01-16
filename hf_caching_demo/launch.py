import argparse
import gcsfs
import datasets
from etils import epath


# to make an api key, follow instructions at
# https://developers.google.com/workspace/guides/create-credentials
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hfds_identifier")
    parser.add_argument("gc_project")
    parser.add_argument("gc_secret_key")
    parser.add_argument("gc_storage_uri")
    args = parser.parse_args()
    storage_options = dict(
        project=args.gc_project,
        token=args.gc_secret_key,
    )
    fs = gcsfs.GCSFileSystem(**storage_options)  # todo: do we need this line?
    builder = datasets.load_dataset_builder(args.hfds_identifier)
    builder.download_and_prepare(
        (epath.Path(args.gc_storage_uri) / "staging").as_uri(),
        storage_options=storage_options,
        file_format="parquet",
    )

    # #
    # encoded_dataset.save_to_disk(
    #     "gcs://my-private-datasets/imdb/train",
    #     storage_options=storage_options,
    # )


if __name__ == "__main__":
    main()
