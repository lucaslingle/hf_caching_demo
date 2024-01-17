import numpy as np

from hf_caching_demo.dataloader import get_tokenizer, get_dataset


def test_get_tokenizer():
    tokenizer = get_tokenizer("GPT2TokenizerFast", "gpt2")
    text0 = "Hello world."
    text1 = "The quick brown fox jumped over the lazy dog."
    text2 = "".join([chr(i) for i in range(256) if chr(i) != " "])  # broken w/ space?!
    texts = [text0, text1, text2]
    for text in texts:
        encoded = tokenizer(text)["input_ids"]
        output = tokenizer.decode(encoded)
        for i in range(min(len(text), len(output))):
            assert text[i] == output[i]
        assert len(text) == len(output)
        assert text == output


def test_get_dataset():
    batch_size = 8
    seq_len = 512
    buffer_size = 1024
    unittesting_shard_size = batch_size * 10

    ds_iter_original = get_dataset(
        hfds_identifier="roneneldan/TinyStories",  # todo: needs internet connection
        hfds_config=None,
        hfds_datacol="text",
        hfds_buffer_size=buffer_size,
        hftr_tokenizer=get_tokenizer("GPT2TokenizerFast", "gpt2"),
        split_name="train",
        batch_size=batch_size,
        sequence_len=seq_len,
        step=0,
        unittesting_shard_size=unittesting_shard_size,
    )
    batch0 = next(ds_iter_original)  # step 0
    batch1 = next(ds_iter_original)  # step 1
    batch2 = next(ds_iter_original)  # step 2
    expected_batch3 = next(ds_iter_original)

    ds_iter_resumed = get_dataset(
        hfds_identifier="roneneldan/TinyStories",  # todo: needs internet connection
        hfds_config=None,
        hfds_datacol="text",
        hfds_buffer_size=buffer_size,
        hftr_tokenizer=get_tokenizer("GPT2TokenizerFast", "gpt2"),
        split_name="train",
        batch_size=batch_size,
        sequence_len=seq_len,
        step=3,
        unittesting_shard_size=unittesting_shard_size,
    )
    actual_batch3 = next(ds_iter_resumed)
    assert expected_batch3.keys() == actual_batch3.keys()
    for k in expected_batch3:
        np.testing.assert_allclose(actual=actual_batch3[k], desired=expected_batch3[k])
