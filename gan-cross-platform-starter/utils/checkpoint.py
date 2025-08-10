import os, glob, torch

def latest_checkpoint(path="checkpoints", pattern="ckpt_*.pt"):
    os.makedirs(path, exist_ok=True)
    files = sorted(glob.glob(os.path.join(path, pattern)))
    return files[-1] if files else None

def save_checkpoint(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)
