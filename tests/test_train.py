import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import build_model
from train import evaluate, fit, make_optimizer, pending_experiments


def tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 6))


@pytest.fixture
def tiny_loaders():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(12, 3, 8, 8, generator=g)
    y = torch.arange(12) % 6
    ds = TensorDataset(x, y)
    return {s: DataLoader(ds, batch_size=4, shuffle=(s == "train")) for s in ["train", "val", "test"]}


def test_e4_param_groups():
    opt = make_optimizer("E4", build_model("E4", pretrained=False))
    assert sorted(g["lr"] for g in opt.param_groups) == [1e-5, 1e-4]


@pytest.mark.parametrize("exp_id,cls,lr", [("E1", torch.optim.SGD, 1e-3),
    ("E2", torch.optim.SGD, 1e-3), ("E3", torch.optim.Adam, 1e-4), ("E5", torch.optim.Adam, 1e-4)])
def test_optimizer_choice(exp_id, cls, lr):
    opt = make_optimizer(exp_id, build_model(exp_id, pretrained=False))
    assert isinstance(opt, cls) and opt.param_groups[0]["lr"] == lr


def test_optimizer_skips_frozen_params():
    m = build_model("E2", pretrained=False)
    opt = make_optimizer("E2", m)
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_fit_overfits_tiny_batch(tiny_loaders):
    model = tiny_model()
    model, hist = fit(model, torch.optim.Adam(model.parameters(), 1e-2), tiny_loaders, epochs=30, device="cpu")
    assert len(hist) == 30 and hist[-1]["train_acc"] == 1.0


def test_fit_returns_best_val_weights(tiny_loaders):
    model = tiny_model()
    model, hist = fit(model, torch.optim.Adam(model.parameters(), 1e-2), tiny_loaders, epochs=10, device="cpu")
    assert evaluate(model, tiny_loaders["val"], "cpu")["acc"] == max(h["val_acc"] for h in hist)


def test_evaluate_keys_and_confusion_shape(tiny_loaders):
    r = evaluate(tiny_model(), tiny_loaders["test"], "cpu")
    assert {"acc", "macro_f1", "report", "confusion"} <= set(r)
    assert len(r["confusion"]) == 6 and all(len(row) == 6 for row in r["confusion"])


def test_pending_experiments(tmp_path):
    csv = tmp_path / "results.csv"
    assert pending_experiments(csv, ["E1", "E2"]) == ["E1", "E2"]
    pd.DataFrame({"exp_id": ["E1"]}).to_csv(csv, index=False)
    assert pending_experiments(csv, ["E1", "E2", "E3"]) == ["E2", "E3"]
