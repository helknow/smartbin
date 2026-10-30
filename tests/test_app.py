import io

import pytest
import torch
from PIL import Image

import app
from app import load, predict, save_checkpoint
from model import BIN_INFO, EXPERIMENTS, build_model


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_model("E5", pretrained=False).eval()


@pytest.mark.parametrize("exp_id", EXPERIMENTS)
def test_checkpoint_roundtrip(exp_id, tmp_path):
    m = build_model(exp_id, pretrained=False).eval()
    save_checkpoint(m, exp_id, 0.9, tmp_path / "b.pt")
    m2, meta = load(tmp_path / "b.pt")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert meta["exp_id"] == exp_id and torch.allclose(m(x), m2(x))


def test_missing_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run notebooks"):
        load(tmp_path / "none.pt")


@pytest.mark.parametrize("img", [Image.new("RGB", (512, 384)), Image.new("RGBA", (300, 300)),
    Image.new("L", (200, 200)), Image.new("RGB", (40, 40)), Image.new("RGB", (4000, 3000))])
def test_predict_any_image(model, img):
    r = predict(model, img)
    assert len(r["top3"]) == 3 and abs(sum(r["probs"].values()) - 1) < 1e-4
    assert r["bin"] == BIN_INFO[r["top3"][0][0]]["bin"]


def test_predict_none(model):
    assert "error" in predict(model, None)


def test_low_confidence_warning(model):
    model.classifier[1].weight.data.zero_(); model.classifier[1].bias.data.zero_()  # -> uniform 1/6
    assert predict(model, Image.new("RGB", (224, 224)))["warning"].startswith("Not sure")


def test_confident_no_warning(model):
    model.classifier[1].weight.data.zero_(); model.classifier[1].bias.data[:] = torch.tensor([9., 0, 0, 0, 0, 0])
    r = predict(model, Image.new("RGB", (224, 224)))
    assert r["warning"] is None and r["top3"][0][0] == "cardboard"


def test_exif_rotation_applied(model, monkeypatch):
    buf = io.BytesIO(); exif = Image.Exif(); exif[0x0112] = 6   # "rotate 90 deg CW to display"
    Image.new("RGB", (400, 200)).save(buf, "JPEG", exif=exif)
    img = Image.open(io.BytesIO(buf.getvalue()))
    seen = {}
    real = app.eval_transform()
    monkeypatch.setattr(app, "eval_transform", lambda: (lambda im: (seen.setdefault("size", im.size), real(im))[1]))
    predict(model, img)
    assert seen["size"] == (200, 400)   # portrait after transpose
