"""Charts for the defence: experiment comparison, training curves, confusion matrix."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from model import CLASSES


def _save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_comparison(results, out):
    x = np.arange(len(results))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_acc = ax.bar(x - width / 2, results["test_acc"], width, label="Test accuracy", color="#2563eb")
    bars_f1 = ax.bar(x + width / 2, results["macro_f1"], width, label="Macro F1", color="#16a34a")
    ax.bar_label(bars_acc, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars_f1, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, results["exp_id"])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Score")
    ax.set_title("SmartBin experiment comparison (test set)")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out)


def plot_curves(history, exp_id, out):
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key, title in [(ax_loss, "loss", "Loss"), (ax_acc, "acc", "Accuracy")]:
        ax.plot(epochs, [h[f"train_{key}"] for h in history], label="Train", color="#2563eb")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], label="Validation", color="#ea580c")
        ax.set_xlabel("Epoch")
        ax.set_title(f"{exp_id}: {title}")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out)


def plot_confusion(confusion, exp_id, out):
    cm = np.array(confusion)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{exp_id}: confusion matrix (test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, out)
