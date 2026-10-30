import matplotlib

matplotlib.use("Agg")

import pandas as pd

from report import plot_comparison, plot_confusion, plot_curves


def _ok(path):
    return path.exists() and path.stat().st_size > 1000


def test_plot_comparison(tmp_path):
    df = pd.DataFrame({"exp_id": ["E1", "E2"], "test_acc": [0.8, 0.9], "macro_f1": [0.75, 0.88]})
    out = tmp_path / "c.png"
    plot_comparison(df, out)
    assert _ok(out)


def test_plot_curves(tmp_path):
    hist = [{"epoch": e, "train_loss": 1 / e, "train_acc": 0.5 + e / 40, "val_loss": 1.2 / e, "val_acc": 0.5 + e / 50}
            for e in range(1, 11)]
    out = tmp_path / "curves.png"
    plot_curves(hist, "E1", out)
    assert _ok(out)


def test_plot_confusion(tmp_path):
    cm = [[5 if i == j else 1 for j in range(6)] for i in range(6)]
    out = tmp_path / "cm.png"
    plot_confusion(cm, "E1", out)
    assert _ok(out)
