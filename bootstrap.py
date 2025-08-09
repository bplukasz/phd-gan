# BOOTSTRAP: odtwórz repo GAN Kaggle ⇄ Colab ⇄ RunPod/PLGrid w bieżącym środowisku
import os, json, textwrap, pathlib

BASE = pathlib.Path("gan-cross-platform-starter")
(BASE / "utils").mkdir(parents=True, exist_ok=True)
(BASE / "notebooks").mkdir(parents=True, exist_ok=True)

# .gitignore
open(BASE/".gitignore","w").write(textwrap.dedent("""
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
ENV/
.ipynb_checkpoints
data/
checkpoints/
logs/
wandb/
kaggle.json
.DS_Store
build/
dist/
*.egg-info/
""").strip()+"\n")

# requirements.txt (bez pinowania torch/torchvision)
open(BASE/"requirements.txt","w").write(textwrap.dedent("""
kaggle>=1.6.17
huggingface_hub>=0.23.0
wandb>=0.17.0
tqdm>=4.66.0
omegaconf>=2.3.0
""").strip()+"\n")

# README.md
open(BASE/"README.md","w").write(textwrap.dedent("""
# GAN Cross-Platform Starter (Kaggle ⇄ Colab ⇄ RunPod/PLGrid)

Start na Kaggle → kontynuacja na Colab → wznowienie gdziekolwiek (RunPod/PLGrid).
W pakiecie: mini-DCGAN + checkpointy + sync (Kaggle Datasets / HF Hub / W&B).

**Odpal:** notebooks/01_kaggle_colab_runpod.ipynb
""").strip()+"\n")

# utils/checkpoint.py
open(BASE/"utils/checkpoint.py","w").write(textwrap.dedent("""
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
""").strip()+"\n")

# utils/sync_kaggle.py
open(BASE/"utils/sync_kaggle.py","w").write(textwrap.dedent(r'''
import os, subprocess, shutil
from pathlib import Path

def _ensure_kaggle_config():
    cfg1 = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(cfg1) and not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        raise RuntimeError("Missing Kaggle credentials. Provide ~/.kaggle/kaggle.json or env KAGGLE_USERNAME/KAGGLE_KEY.")

def kaggle_dataset_push(dataset_slug, folder="checkpoints", title=None, is_public=False):
    """
    Create or version a Kaggle Dataset from a local folder.
    dataset_slug: 'user/your-dataset-slug' OR 'your-dataset-slug' if user is implied by kaggle config.
    """
    _ensure_kaggle_config()
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    meta_dir = Path("kaggle_meta"); meta_dir.mkdir(exist_ok=True)
    ds_slug_only = dataset_slug.split("/")[-1]
    title = title or ds_slug_only.replace("-", " ").title()
    (meta_dir / "dataset-metadata.json").write_text(
        f'{{"title":"{title}","id":"{dataset_slug}","licenses":[{{"name":"CC0-1.0"}}]}}', encoding="utf-8"
    )
    # zip folder
    zip_base = meta_dir / "payload"
    if (meta_dir / "payload.zip").exists(): (meta_dir / "payload.zip").unlink()
    shutil.make_archive(str(zip_base), 'zip', folder)
    try:
        subprocess.check_call([
            "kaggle","datasets","create","-p",str(meta_dir),"-r","zip","-o",
            "--public" if is_public else "--private"
        ])
    except subprocess.CalledProcessError:
        subprocess.check_call([
            "kaggle","datasets","version","-p",str(meta_dir),"-r","zip","-m","auto snapshot"
        ])
    return True

def kaggle_dataset_pull(dataset_slug, out_dir="downloaded"):
    _ensure_kaggle_config()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["kaggle","datasets","download","-d",dataset_slug,"-p",str(out),"-q"])
    for f in out.glob("*.zip"):
        shutil.unpack_archive(str(f), out)
    return str(out.resolve())
''').strip()+"\n")

# utils/sync_hf.py
open(BASE/"utils/sync_hf.py","w").write(textwrap.dedent("""
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
""").strip()+"\n")

# Notebook (skrócony; wszystkie kluczowe komórki)
nb = {
 "cells": [
  {"cell_type":"markdown","metadata":{},"source":[
    "# Kaggle → Colab → Anywhere: DCGAN Starter\n",
    "Szybki przepływ z checkpointami i sync (Kaggle Datasets / HF Hub / W&B)."
  ]},
  {"cell_type":"code","metadata":{"id":"setup"},"source":[
    "import os, sys, subprocess\n",
    "def pip_install(pkgs): subprocess.check_call([sys.executable,'-m','pip','install','--quiet']+pkgs)\n",
    "try:\n",
    "    import torch, torchvision  # noqa\n",
    "except Exception:\n",
    "    pip_install(['torch','torchvision'])\n",
    "pip_install(['tqdm','omegaconf','wandb','kaggle','huggingface_hub'])\n",
    "print('Setup OK')\n"
  ],"execution_count":None,"outputs":[]},
  {"cell_type":"code","metadata":{"id":"folders"},"source":[
    "import pathlib, platform, torch\n",
    "BASE=pathlib.Path('.')\n",
    "CKPT=(BASE/'checkpoints'); LOGS=(BASE/'logs'); DATA=(BASE/'data')\n",
    "for d in [CKPT,LOGS,DATA]: d.mkdir(parents=True, exist_ok=True)\n",
    "print('Python', platform.python_version(), 'Torch', torch.__version__, 'CUDA', torch.cuda.is_available())\n"
  ],"execution_count":None,"outputs":[]},
  {"cell_type":"code","metadata":{"id":"helpers"},"source":[
    "import sys; sys.path.append(str((BASE/'..'/'..'/'gan-cross-platform-starter').resolve()))\n",
    "sys.path.append('gan-cross-platform-starter')\n",
    "from utils.checkpoint import latest_checkpoint, save_checkpoint, load_checkpoint\n",
    "from utils.sync_kaggle import kaggle_dataset_push, kaggle_dataset_pull\n",
    "from utils.sync_hf import hf_snapshot_upload, hf_snapshot_download\n",
    "print('Helpers OK')\n"
  ],"execution_count":None,"outputs":[]},
  {"cell_type":"code","metadata":{"id":"data"},"source":[
    "from torchvision import datasets, transforms\n",
    "from torch.utils.data import DataLoader\n",
    "transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])\n",
    "train_set=datasets.MNIST(root='data', train=True, download=True, transform=transform)\n",
    "loader=DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)\n",
    "len(train_set), len(loader)\n"
  ],"execution_count":None,"outputs":[]},
  {"cell_type":"code","metadata":{"id":"model"},"source":[
    "import torch, torch.nn as nn\n",
    "nz=64\n",
    "class G(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__(); self.net=nn.Sequential(\n",
    "            nn.ConvTranspose2d(nz,256,4,1,0,bias=False), nn.BatchNorm2d(256), nn.ReLU(True),\n",
    "            nn.ConvTranspose2d(256,128,4,2,1,bias=False), nn.BatchNorm2d(128), nn.ReLU(True),\n",
    "            nn.ConvTranspose2d(128,64,4,2,1,bias=False), nn.BatchNorm2d(64), nn.ReLU(True),\n",
    "            nn.ConvTranspose2d(64,1,4,2,1,bias=False), nn.Tanh())\n",
    "    def forward(self,z): return self.net(z)\n",
    "class D(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__(); self.net=nn.Sequential(\n",
    "            nn.Conv2d(1,64,4,2,1,bias=False), nn.LeakyReLU(0.2, inplace=True),\n",
    "            nn.Conv2d(64,128,4,2,1,bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),\n",
    "            nn.Conv2d(128,256,4,2,1,bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),\n",
    "            nn.Conv2d(256,1,4,1,0,bias=False))\n",
    "    def forward(self,x): return self.net(x).view(-1)\n",
    "device='cuda' if torch.cuda.is_available() else 'cpu'\n",
    "Gnet, Dnet = G().to(device), D().to(device)\n",
    "optG=torch.optim.Adam(Gnet.parameters(), lr=2e-4, betas=(0.5,0.999))\n",
    "optD=torch.optim.Adam(Dnet.parameters(), lr=2e-4, betas=(0.5,0.999))\n",
    "criterion=nn.BCEWithLogitsLoss()\n",
    "print('Model ready on', device)\n"
  ],"execution_count":None,"outputs":[]},
  {"cell_type":"code","metadata":{"id":"resume"},"source":[
    "ckpt=latest_checkpoint('checkpoints')\n",
    "global_step=0\n",
    "if ckpt:\n",
    "    s=load_checkpoint(ckpt,'cpu')\n",
    "    Gnet.load_state_dict(s['G']); Dnet.load_state_dict(s['D'])\n",
    "    optG.load_state_dict(s['optG']); optD.load_state_dict(s['optD'])\n",
    "    global_step=int(s.get('step',0)); print('Resumed from', ckpt, 'step', global_step)\n",
    "else:\n",
    "    print('No checkpoint – start fresh')\n"
  ],"execution_count":None,"outputs":[]},
  {"cell_type":"code","metadata":{"id":"train"},"source":[
    "from tqdm import tqdm; import wandb, os\n",
    "use_wandb=bool(os.getenv('WANDB_API_KEY'))\n",
    "if use_wandb: wandb.init(project=os.getenv('WANDB_PROJECT','gan-starter'), reinit=True)\n",
    "epochs=1; save_every=500; nz=64; device=device\n",
    "for epoch in range(epochs):\n",
    "    pbar=tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', ncols=100)\n",
    "    for real,_ in pbar:\n",
    "        real=real.to(device); bs=real.size(0)\n",
    "        z=torch.randn(bs,nz,1,1,device=device); fake=Gnet(z).detach()\n",
    "        lossD=criterion(Dnet(real), torch.ones(bs, device=device)) + \\\n",
    "              criterion(Dnet(fake), torch.zeros(bs, device=device))\n",
    "        optD.zero_grad(); lossD.backward(); optD.step()\n",
    "        z=torch.randn(bs,nz,1,1,device=device); fake=Gnet(z)\n",
    "        lossG=criterion(Dnet(fake), torch.ones(bs, device=device))\n",
    "        optG.zero_grad(); lossG.backward(); optG.step()\n",
    "        global_step+=1\n",
    "        if use_wandb: wandb.log({'lossD':lossD.item(), 'lossG':lossG.item(), 'step':global_step})\n",
    "        if global_step % save_every == 0:\n",
    "            import pathlib\n",
    "            p=pathlib.Path('checkpoints')/f'ckpt_{global_step:07d}.pt'\n",
    "            save_checkpoint(str(p), {'step':global_step,'G':Gnet.state_dict(),'D':Dnet.state_dict(),\n",
    "                                    'optG':optG.state_dict(),'optD':optD.state_dict(),'cfg':{'nz':nz}})\n",
    "print('Done. Last step =', global_step)\n"
  ],"execution_count":None,"outputs":[]},
  {"cell_type":"markdown","metadata":{},"source":["### Push/Pull – Kaggle Datasets"]},
  {"cell_type":"code","metadata":{"id":"kaggle-sync"},"source":[
    "import os\n",
    "slug=os.getenv('KAGGLE_DATASET_SLUG','your_kaggle_username/gan-checkpoints')\n",
    "try:\n",
    "    kaggle_dataset_push(slug, folder='checkpoints', title='GAN Checkpoints', is_public=False)\n",
    "    print('Pushed to Kaggle:', slug)\n",
    "except Exception as e:\n",
    "    print('Kaggle push skipped/error:', e)\n",
    "try:\n",
    "    out=kaggle_dataset_pull(slug, out_dir='downloaded')\n",
    "    print('Pulled to:', out)\n",
    "except Exception as e:\n",
    "    print('Kaggle pull skipped/error:', e)\n"
  ],"execution_count":None,"outputs":[]},
  {"cell_type":"markdown","metadata":{},"source":["### Push/Pull – Hugging Face Hub"]},
  {"cell_type":"code","metadata":{"id":"hf-sync"},"source":[
    "HF_TOKEN=os.getenv('HF_TOKEN'); HF_REPO=os.getenv('HF_REPO','username/gan-checkpoints')\n",
    "if HF_TOKEN:\n",
    "    try:\n",
    "        hf_snapshot_upload(HF_REPO, local_dir='checkpoints', token=HF_TOKEN, private=True)\n",
    "        print('Pushed to HF:', HF_REPO)\n",
    "    except Exception as e:\n",
    "        print('HF upload error:', e)\n",
    "    try:\n",
    "        path=hf_snapshot_download(HF_REPO, local_dir='downloaded_hf', token=HF_TOKEN)\n",
    "        print('Pulled from HF to:', path)\n",
    "    except Exception as e:\n",
    "        print('HF download error:', e)\n",
    "else:\n",
    "    print('Set HF_TOKEN to use HF sync')\n"
  ],"execution_count":None,"outputs":[]},
 ],
 "metadata": {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},
 "nbformat":4, "nbformat_minor":5
}
open(BASE/"notebooks/01_kaggle_colab_runpod.ipynb","w",encoding="utf-8").write(json.dumps(nb, ensure_ascii=False, indent=2))

print(f"✅ Utworzono projekt w: {BASE.resolve()}")
print("Otwórz teraz notebooks/01_kaggle_colab_runpod.ipynb i uruchamiaj kolejne komórki.")
