import os
from huggingface_hub import snapshot_download

download_list = [
    "google/gemma-2b-it",
]

folder = "../pretrained"

if not os.path.exists(folder):
    os.makedirs(folder)

for repo in download_list:
    snapshot_download(
        repo_id=repo,
        local_dir=os.path.join(folder,repo),
        cache_dir=os.path.join(folder,repo),
        local_dir_use_symlinks=False,
        # ignore_patterns=["*.pth"],
        token="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"
    )