from pathlib import Path

import pytest
import torch
from PIL import Image

from data import make_loaders, make_splits
from model import CLASSES


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "dataset-resized"
    # create folders in reverse order so disk order != CLASSES order
    for c in reversed(CLASSES):
        (root / c).mkdir(parents=True)
        for i in range(10 if c == "trash" else 20):
            Image.new("RGB", (64, 48), (i * 10, 50, 100)).save(root / c / f"{c}{i}.jpg")
    (root / "__MACOSX" / "glass").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(root / "__MACOSX" / "glass" / "x.jpg")
    (root / "glass" / ".DS_Store").write_bytes(b"\0")
    return root


def test_split_proportions_and_stratified(fake_root, tmp_path):
    df = make_splits(fake_root, tmp_path / "s.csv")
    assert len(df) == 110
    frac = df.split.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.03 and abs(frac["test"] - 0.15) < 0.03
    for split in ["train", "val", "test"]:
        assert set(df[df.split == split].label) == set(range(6))


def test_split_reproducible_and_cached(fake_root, tmp_path):
    a = make_splits(fake_root, tmp_path / "a.csv")
    b = make_splits(fake_root, tmp_path / "b.csv")
    assert a.equals(b)
    assert make_splits(fake_root, tmp_path / "a.csv").equals(a)


def test_labels_follow_CLASSES_not_disk_order(fake_root, tmp_path):
    df = make_splits(fake_root, tmp_path / "s.csv")
    for _, r in df.iterrows():
        assert Path(r.path).parent.name == CLASSES[r.label] and not Path(r.path).is_absolute()
    assert not df.path.str.contains("__MACOSX|DS_Store").any()


def test_loader_batch_shape(fake_root, tmp_path):
    loaders = make_loaders(make_splits(fake_root, tmp_path / "s.csv"), fake_root, batch_size=4, num_workers=0)
    x, y = next(iter(loaders["train"]))
    assert x.shape == (4, 3, 224, 224) and y.dtype == torch.int64
