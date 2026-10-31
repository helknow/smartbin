"""Export the 30-epoch E5 model to ONNX for the website, and check it scores the same as the PyTorch model.

    python export_web_model.py
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image

from model import build_model, eval_transform

CHECKPOINT = Path("longer_training/weights/E5.pt")
OUTPUT = Path("web/model.onnx")
SPLITS = Path("splits.csv")
DATA_ROOT = Path("data/dataset-resized")


def main():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = build_model("E5", pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), OUTPUT, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)

    session = ort.InferenceSession(OUTPUT, providers=["CPUExecutionProvider"])
    df = pd.read_csv(SPLITS)
    test = df[df.split == "test"]
    transform = eval_transform()
    correct, biggest_gap = 0, 0.0
    for row in test.itertuples():
        x = transform(Image.open(DATA_ROOT / row.path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            torch_logits = model(x).numpy()
        onnx_logits = session.run(None, {"image": x.numpy()})[0]
        biggest_gap = max(biggest_gap, float(np.abs(torch_logits - onnx_logits).max()))
        correct += int(onnx_logits.argmax() == row.label)
    print(f"file: {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")
    print(f"test accuracy: {correct}/{len(test)} = {correct / len(test):.4f}  (checkpoint says {checkpoint['test_acc']:.4f})")
    print(f"biggest PyTorch vs ONNX difference: {biggest_gap:.6f}")


if __name__ == "__main__":
    main()
