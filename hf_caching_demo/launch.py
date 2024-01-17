# todo: maybe do the token shift inside model to reduce host-to-device io

import datasets as hfds
from absl import app
from absl import flags
from absl import logging

from dataloader import get_tokenizer, get_dataset

hfds.disable_caching()
STEP = 0
BATCH_SIZE = 8
SEQLEN = 512
TEXTCOL = "text"

FLAGS = flags.FLAGS
flags.DEFINE_string("hfds_identifier", None, "HF datasets identifier")
flags.DEFINE_string("hfds_split_name", None, "HF datasets split name")
flags.mark_flags_as_required(["hfds_identifier", "hfds_split_name"])


def main(argv):
    del argv

    logging.info(f"calling get_tokenizer to get fast tokenizer...")
    tokenizer = get_tokenizer("GPT2TokenizerFast", "gpt2")

    logging.info(f"calling get_dataset to get sharded bitwise reproducible dataset...")
    ds = get_dataset(
        hfds_identifier=FLAGS.hfds_identifier,
        hfds_config=None,
        hfds_datacol=TEXTCOL,
        hfds_buffer_size=1024,
        hftr_tokenizer=tokenizer,
        split_name=FLAGS.hfds_split_name,
        batch_size=BATCH_SIZE,
        sequence_len=SEQLEN,
        step=STEP,
    )
    print(next(ds))


if __name__ == "__main__":
    app.run(main)
