"""Optimisers per experiment, training loop with best-val checkpoint, evaluation, resume helper."""
import copy
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn

from model import CLASSES


def make_optimizer(exp_id, model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if exp_id in ("E1", "E2"):
        return torch.optim.SGD(trainable, lr=1e-3, momentum=0.9)
    if exp_id in ("E3", "E5"):
        return torch.optim.Adam(trainable, lr=1e-4)
    if exp_id == "E4":
        return torch.optim.Adam([
            {"params": model.features[24:].parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ])
    raise ValueError(f"Unknown experiment: {exp_id}")


def _run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
    return total_loss / seen, correct / seen


def fit(model, optimizer, loaders, epochs, device):
    """Train and return the model with its best-validation weights, plus the per-epoch history."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    history, best_acc, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc = _run_epoch(model, loaders["val"], criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch}/{epochs}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history


def evaluate(model, loader, device):
    model.to(device).eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            y_pred += model(x.to(device)).argmax(1).cpu().tolist()
            y_true += y.tolist()
    labels = list(range(len(CLASSES)))
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "report": classification_report(y_true, y_pred, labels=labels, target_names=CLASSES,
                                        output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def pending_experiments(results_csv, wanted):
    """Experiments in `wanted` that have no row in results_csv yet (so a rerun can resume)."""
    results_csv = Path(results_csv)
    if not results_csv.exists():
        return list(wanted)
    done = set(pd.read_csv(results_csv)["exp_id"])
    return [e for e in wanted if e not in done]
