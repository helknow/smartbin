"""Run the SmartBin experiments end to end. Used by the Colab notebook and from the command line.

    python run_experiments.py                  # all experiments, then pick the best
    python run_experiments.py --experiments E1 E2
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from data import download_trashnet, make_loaders, make_splits
from model import CLASSES, EXPERIMENTS, build_model, count_trainable, save_checkpoint
from report import plot_comparison, plot_confusion, plot_curves
from train import evaluate, fit, make_optimizer, pending_experiments, set_train_mode

SEED = 42
RESULT_COLUMNS = ["exp_id", "test_acc", "macro_f1", "trash_f1", "train_time_s", "trainable_params", "best_val_acc"]


def overfit_check(exp_id, loaders, device, pretrained=True, steps=30):
    """Train on a single batch; a healthy training loop should nearly memorise it. Returns its accuracy."""
    torch.manual_seed(SEED)
    model = build_model(exp_id, pretrained=pretrained).to(device)
    optimizer = make_optimizer(exp_id, model)
    criterion = nn.CrossEntropyLoss()
    x, y = next(iter(loaders["train"]))
    x, y = x.to(device), y.to(device)
    set_train_mode(model)
    for _ in range(steps):
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def run_all(root, out_dir, experiments, epochs=10, device="cpu", pretrained=True, batch_size=32, num_workers=2):
    """Train every experiment not yet in results.csv, saving each result right after its run."""
    root, out_dir = Path(root), Path(out_dir)
    results_csv = out_dir / "results" / "results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df = make_splits(root, out_dir / "splits.csv")
    loaders = make_loaders(df, root, batch_size=batch_size, num_workers=num_workers)

    for exp_id in pending_experiments(results_csv, experiments):
        print(f"\n=== {exp_id} ===")
        torch.manual_seed(SEED)
        model = build_model(exp_id, pretrained=pretrained)
        trainable = count_trainable(model)
        optimizer = make_optimizer(exp_id, model)
        start = time.perf_counter()
        model, history = fit(model, optimizer, loaders, epochs, device)
        train_time = time.perf_counter() - start
        metrics = evaluate(model, loaders["test"], device)

        plot_curves(history, exp_id, out_dir / "results" / f"curves_{exp_id}.png")
        plot_confusion(metrics["confusion"], exp_id, out_dir / "results" / f"confusion_{exp_id}.png")
        (out_dir / "results" / f"report_{exp_id}.json").write_text(json.dumps(metrics["report"], indent=2))
        save_checkpoint(model, exp_id, metrics["acc"], out_dir / "weights" / f"{exp_id}.pt")

        row = {"exp_id": exp_id, "test_acc": metrics["acc"], "macro_f1": metrics["macro_f1"],
               "trash_f1": metrics["report"]["trash"]["f1-score"], "train_time_s": round(train_time, 1),
               "trainable_params": trainable, "best_val_acc": max(h["val_acc"] for h in history)}
        pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(
            results_csv, mode="a", header=not results_csv.exists(), index=False)
        print(f"{exp_id}: test_acc={metrics['acc']:.3f} macro_f1={metrics['macro_f1']:.3f} time={train_time:.0f}s")
    return df


def select_best(out_dir):
    """Copy the checkpoint with the highest test accuracy to weights/best.pt. Returns its exp_id."""
    out_dir = Path(out_dir)
    results = pd.read_csv(out_dir / "results" / "results.csv")
    best = results.loc[results["test_acc"].idxmax(), "exp_id"]
    checkpoint = out_dir / "weights" / f"{best}.pt"
    if not checkpoint.exists():
        raise SystemExit(
            f"{checkpoint} is missing, so nothing was trained: results/results.csv already lists these experiments. "
            "Delete results/results.csv to retrain, or copy the checkpoints into weights/.")
    shutil.copyfile(checkpoint, out_dir / "weights" / "best.pt")
    return best


def copy_examples(df, root, out_dir):
    """Copy one test-split photo per class, so the app's examples were never seen in training."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test = df[df.split == "test"].sort_values("path")
    for label, name in enumerate(CLASSES):
        first = test[test.label == label].iloc[0]
        shutil.copyfile(Path(root) / first.path, out_dir / f"{name}.jpg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENTS, choices=EXPERIMENTS)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default=".")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    root = download_trashnet(Path(args.data))
    df = make_splits(root, Path(args.out) / "splits.csv")
    print(f"images: {len(df)}  {df.split.value_counts().to_dict()}")

    acc = overfit_check("E2", make_loaders(df, root), device)
    print(f"overfit check (E2, one batch): {acc:.2f}")
    if acc < 0.8:
        raise SystemExit("Overfit check failed: the training loop can't memorise one batch. Stopping.")

    df = run_all(root, args.out, args.experiments, epochs=args.epochs, device=device)
    results = pd.read_csv(Path(args.out) / "results" / "results.csv")
    plot_comparison(results, Path(args.out) / "results" / "comparison.png")
    best = select_best(args.out)
    copy_examples(df, root, Path(args.out) / "examples")
    print(results.to_string(index=False))
    print(f"\nBest experiment: {best} (copied to weights/best.pt)")


if __name__ == "__main__":
    main()
