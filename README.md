# hf_caching_demo
Self-demo of HF dataset caching (work in progress)

### Installation

Install Pipx and Poetry. Then run
```
git clone https://github.com/lucaslingle/hf_caching_demo.git;
cd hf_caching_demo;
poetry install --with tpu --without cpu;  # or other way around
```

### Usage

To make a Google Cloud API key, follow instructions [here](https://developers.google.com/workspace/guides/create-credentials).  
To run the script, do something like this
```
poetry run python3 hf_caching_demo/launch.py \
    --hfds_identifier=Skylion007/openwebtext \
    --hfds_split_name=train \
    --gc_project=someproject \
    --gc_storage_uri=gs://somewhere/
```


### Todo

- How fast to resume midway through a large dataset?
- How much VM disk space do we need to write large datasets to GCS?
- How much VM disk space do we need to read large datasets from GCS?
- How to shard reasonably before writing to GCS? Perhaps just make num files a multiple of 8 (since there are eight cores per TPU VM)?
- How to load with minimal network IO, minimal host-to-device IO? 
- How much does any of this matter with async dispatch? 
