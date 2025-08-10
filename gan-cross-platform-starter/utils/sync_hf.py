import os
from huggingface_hub import HfApi, snapshot_download, create_repo

def hf_login_and_check(token=None):
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("Provide HF token (arg or HF_TOKEN env).")
    api = HfApi(token=token)
    me = api.whoami(token=token)
    return api, me

def hf_snapshot_upload(repo_id, local_dir="checkpoints", token=None, private=True):
    api, _ = hf_login_and_check(token)
    try:
        create_repo(repo_id, private=private, exist_ok=True, token=api.token)
    except Exception:
        pass
    from huggingface_hub import snapshot_upload
    snapshot_upload(repo_id=repo_id, local_dir=local_dir, token=api.token, commit_message="auto snapshot")
    return True

def hf_snapshot_download(repo_id, local_dir="downloaded", token=None):
    api, _ = hf_login_and_check(token)
    path = snapshot_download(repo_id=repo_id, local_dir=local_dir, token=api.token)
    return path
