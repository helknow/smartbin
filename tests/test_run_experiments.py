import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
import torch
from PIL import Image

from model import CLASSES, build_model, save_checkpoint
from run_experiments import copy_examples, overfit_check, run_all, select_best
from data import make_loaders, make_splits

COLUMNS = ["exp_id", "test_acc", "macro_f1", "trash_f1", "train_time_s", "trainable_params", "best_val_acc"]


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "data"
    for c in CLASSES:
        (root / c).mkdir(parents=True)
        for i in range(10):
            Image.new("RGB", (64, 48), (i * 20, 50, 100)).save(root / c / f"{c}{i}.jpg")
    return root


def test_run_all_writes_outputs_and_resumes(fake_root, tmp_path, monkeypatch):
    out = tmp_path / "project"
    kwargs = dict(epochs=1, device="cpu", pretrained=False, batch_size=8, num_workers=0)
    run_all(fake_root, out, ["E5"], **kwargs)
    results = pd.read_csv(out / "results" / "results.csv")
    assert list(results.columns) == COLUMNS and list(results.exp_id) == ["E5"]
    for name in ["results/curves_E5.png", "results/confusion_E5.png", "weights/E5.pt", "splits.csv"]:
        assert (out / name).exists(), name

    # A rerun must not train E5 again or add a second row
    monkeypatch.setattr("run_experiments.fit", lambda *a, **k: pytest.fail("E5 was retrained"))
    run_all(fake_root, out, ["E5"], **kwargs)
    assert len(pd.read_csv(out / "results" / "results.csv")) == 1


def test_select_best_copies_highest_test_acc(tmp_path):
    (tmp_path / "results").mkdir()
    pd.DataFrame({"exp_id": ["E5", "E2"], "test_acc": [0.9, 0.8]}).to_csv(tmp_path / "results" / "results.csv", index=False)
    save_checkpoint(build_model("E5", pretrained=False), "E5", 0.9, tmp_path / "weights" / "E5.pt")
    assert select_best(tmp_path) == "E5"
    assert torch.load(tmp_path / "weights" / "best.pt", weights_only=True)["exp_id"] == "E5"


def test_overfit_check_returns_accuracy(fake_root, tmp_path):
    loaders = make_loaders(make_splits(fake_root, tmp_path / "s.csv"), fake_root, batch_size=4, num_workers=0)
    acc = overfit_check("E5", loaders, "cpu", pretrained=False, steps=2)
    assert 0.0 <= acc <= 1.0


def test_copy_examples_one_per_class_from_test_split(fake_root, tmp_path):
    df = make_splits(fake_root, tmp_path / "s.csv")
    out = tmp_path / "examples"
    copy_examples(df, fake_root, out)
    assert sorted(p.stem for p in out.iterdir()) == sorted(CLASSES)
    test_names = {p.split("/")[-1] for p in df[df.split == "test"].path}
    for c in CLASSES:
        assert (out / f"{c}.jpg").stat().st_size == (fake_root / c / next(n for n in sorted(test_names) if n.startswith(c))).stat().st_size


def test_select_best_explains_missing_checkpoint(tmp_path):
    (tmp_path / "results").mkdir()
    pd.DataFrame({"exp_id": ["E1"], "test_acc": [0.9]}).to_csv(tmp_path / "results" / "results.csv", index=False)
    with pytest.raises(SystemExit, match="results.csv"):
        select_best(tmp_path)
