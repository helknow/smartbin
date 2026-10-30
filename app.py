"""SmartBin Gradio app: upload a photo of one waste item and see which bin it belongs in."""
import os
import sys
from pathlib import Path

import gradio as gr
import torch
from PIL import ImageOps

from model import BIN_INFO, CLASSES, build_model, eval_transform, save_checkpoint  # noqa: F401

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights" / "best.pt"
EXAMPLES = HERE / "examples"
COMPARISON = HERE / "results" / "comparison.png"
LOW_CONFIDENCE = 0.5
LOW_CONFIDENCE_MESSAGE = "Not sure. Try a clearer photo of one item on a plain background"


def load(path, device="cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "weights/best.pt not found. Run notebooks/smartbin_training.ipynb first and copy best.pt into weights/")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["exp_id"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, {k: ckpt[k] for k in ("exp_id", "classes", "test_acc")}


def predict(model, image):
    if image is None:
        return {"error": "Please upload a photo of one waste item."}
    image = ImageOps.exif_transpose(image).convert("RGB")
    device = next(model.parameters()).device
    x = eval_transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()
    ranked = sorted(zip(CLASSES, probs), key=lambda pair: pair[1], reverse=True)
    best_class, best_prob = ranked[0]
    info = BIN_INFO[best_class]
    return {
        "top3": ranked[:3],
        "probs": dict(zip(CLASSES, probs)),
        "bin": info["bin"],
        "color": info["color"],
        "tip": info["tip"],
        "warning": LOW_CONFIDENCE_MESSAGE if best_prob < LOW_CONFIDENCE else None,
    }


def _bin_card(result):
    if "error" in result:
        return f"<p>{result['error']}</p>"
    heading = f"Best guess: {result['bin']}" if result["warning"] else result["bin"]
    warning = f"<p style='margin:0 0 8px;font-weight:600'>{result['warning']}</p>" if result["warning"] else ""
    return (
        f"<div style='background:{result['color']};color:#fff;border-radius:12px;padding:20px'>"
        f"{warning}<div style='font-size:1.5rem;font-weight:700'>{heading}</div>"
        f"<p style='margin:8px 0 0'>{result['tip']}</p></div>"
    )


def build_ui(model, meta):
    def classify(image):
        result = predict(model, image)
        return result.get("probs", {}), _bin_card(result)

    with gr.Blocks(title="SmartBin", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# SmartBin: Which bin does it go in?\nUpload a photo of one waste item.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Photo of the item")
                button = gr.Button("Classify", variant="primary")
            with gr.Column():
                card = gr.HTML()
                label = gr.Label(num_top_classes=3, label="Top 3 classes")
        examples = sorted(str(p) for p in EXAMPLES.glob("*.jpg")) if EXAMPLES.exists() else []
        if examples:
            gr.Examples(examples=examples, inputs=image)
        with gr.Accordion("About the model", open=False):
            gr.Markdown(f"Experiment **{meta['exp_id']}**, test accuracy **{meta['test_acc']:.1%}** on TrashNet.")
            if COMPARISON.exists():
                gr.Image(str(COMPARISON), show_label=False, interactive=False)
        button.click(classify, inputs=image, outputs=[label, card])
    return demo


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        loaded_model, loaded_meta = load(WEIGHTS, device)
    except FileNotFoundError as err:
        sys.exit(str(err))
    server_name = "0.0.0.0" if os.environ.get("IN_DOCKER") == "1" else None
    build_ui(loaded_model, loaded_meta).launch(server_name=server_name)
