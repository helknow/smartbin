import pytest
import torch
from PIL import Image

from model import BIN_INFO, CLASSES, EXPERIMENTS, build_model, count_trainable, eval_transform


@pytest.mark.parametrize("exp_id", EXPERIMENTS)
def test_output_shape(exp_id):
    m = build_model(exp_id, pretrained=False).eval()
    assert m(torch.randn(2, 3, 224, 224)).shape == (2, 6)


@pytest.mark.parametrize("exp_id,expected", [
    ("E1", 119_570_438), ("E2", 25_697_286), ("E3", 25_697_286),
    ("E4", 32_776_710), ("E5", 7_686)])
def test_trainable_params(exp_id, expected):
    assert count_trainable(build_model(exp_id, pretrained=False)) == expected


def test_e4_unfreezes_only_last_block():
    m = build_model("E4", pretrained=False)
    assert all(not p.requires_grad for p in m.features[:24].parameters())
    assert all(p.requires_grad for p in m.features[24:].parameters())


def test_unknown_experiment():
    with pytest.raises(ValueError):
        build_model("E9", pretrained=False)


def test_bin_info_covers_classes():
    assert list(BIN_INFO) == CLASSES
    assert all({"bin", "color", "tip"} <= set(v) for v in BIN_INFO.values())


def test_eval_transform_output():
    x = eval_transform()(Image.new("RGB", (512, 384)))
    assert x.shape == (3, 224, 224)
