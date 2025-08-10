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
