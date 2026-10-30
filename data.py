"""TrashNet download, reproducible stratified split, and data loaders."""
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from model import CLASSES, eval_transform, train_transform

TRASHNET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _find_root(dest):
    for candidate in [dest, *dest.rglob("*")]:
        if candidate.is_dir() and "__MACOSX" not in candidate.parts and all((candidate / c).is_dir() for c in CLASSES):
            return candidate
    return None


def download_trashnet(dest):
    """Download and unzip TrashNet into dest. Returns the folder holding the 6 class folders."""
    dest = Path(dest)
    root = _find_root(dest) if dest.exists() else None
    if root is not None:
        return root
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "dataset-resized.zip"
    urllib.request.urlretrieve(TRASHNET_URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    root = _find_root(dest)
    if root is None:
        raise RuntimeError(f"Could not find the class folders {CLASSES} after unzipping into {dest}")
    return root


def make_splits(root, out_csv, seed=42):
    """Stratified 70/15/15 split. An existing out_csv is loaded unchanged so every run uses the same images."""
    root, out_csv = Path(root), Path(out_csv)
    if out_csv.exists():
        return pd.read_csv(out_csv)

    paths, labels = [], []
    for label, name in enumerate(CLASSES):
        for f in sorted((root / name).iterdir()):
            if f.suffix.lower() in _IMAGE_SUFFIXES and not f.name.startswith("."):
                paths.append(f.relative_to(root).as_posix())
                labels.append(label)

    train_p, rest_p, train_y, rest_y = train_test_split(
        paths, labels, train_size=0.70, stratify=labels, random_state=seed)
    val_p, test_p, val_y, test_y = train_test_split(
        rest_p, rest_y, test_size=0.50, stratify=rest_y, random_state=seed)

    df = pd.DataFrame({
        "path": train_p + val_p + test_p,
        "label": train_y + val_y + test_y,
        "split": ["train"] * len(train_p) + ["val"] * len(val_p) + ["test"] * len(test_p),
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


class TrashDataset(Dataset):
    def __init__(self, df, root, transform):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.root / row.path) as img:
            x = self.transform(img.convert("RGB"))
        return x, int(row.label)


def make_loaders(df, root, batch_size=32, num_workers=2):
    loaders = {}
    for split in ["train", "val", "test"]:
        is_train = split == "train"
        ds = TrashDataset(df[df.split == split], root, train_transform() if is_train else eval_transform())
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)
    return loaders
