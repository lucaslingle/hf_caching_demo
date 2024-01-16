# hf_caching_demo
Self-demo of HF dataset caching (work in progress)

### Installation

Install Pipx and Poetry. Then run
```
git clone https://github.com/lucaslingle/hf_caching_demo.git;
cd hf_caching_demo;
poetry install --with GROUP;
```
where the dependency group ```GROUP``` is ```tpu``` for TPU VMs, and ```cpu``` otherwise. 

### Usage

To make a Google Cloud API key, follow instructions [here](https://developers.google.com/workspace/guides/create-credentials).  
To run the script, do something like this
```
poetry run python3 hf_caching_demo/launch.py \
    --hfds_identifier=Skylion007/openwebtext \
    --gc_project=someproject \
    --gc_secret_key=SOMETHING \
    --gc_storage_uri=gs://somewhere/
```
