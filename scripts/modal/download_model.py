"""Download dormant-model-1 to the shared volume cache."""
from huggingface_hub import snapshot_download
import os

model_id = "jane-street/dormant-model-1"
cache_dir = os.environ.get("HF_HOME", "/vol/hf_cache")

print(f"Downloading {model_id} to {cache_dir}...")
snapshot_download(model_id, cache_dir=cache_dir)
print("Done.")
