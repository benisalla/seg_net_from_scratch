"""Dataset checks. No network: Mapillary is faked with a few files on disk."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from segnet.config import DataConfig
from segnet.data import IGNORE_INDEX, ShapesDataset, build_datasets, joint_transform, mapillary_info


def test_shapes_are_deterministic_and_labelled():
    ds = ShapesDataset(length=4, size=48, seed=3)
    (x1, m1), (x2, m2) = ds[2], ds[2]
    torch.testing.assert_close(x1, x2)
    assert torch.equal(m1, m2)
    assert x1.shape == (3, 48, 48) and m1.shape == (48, 48) and m1.dtype == torch.long
    assert set(m1.unique().tolist()) <= {0, 1, 2, 3} and (m1 > 0).any()


def test_shapes_train_and_val_differ():
    train, val, info = build_datasets(DataConfig(dataset="shapes", shapes_train_size=4, shapes_val_size=4))
    assert info.num_classes == 4
    assert not torch.equal(train[0][0], val[0][0])


def test_joint_flip_keeps_image_and_mask_aligned():
    # left half red / class 1, right half blue / class 2: after any flip they must still match
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :4, 0] = 255
    image[:, 4:, 2] = 255
    mask = np.ones((8, 8), dtype=np.uint8)
    mask[:, 4:] = 2
    for _ in range(20):
        x, m = joint_transform(Image.fromarray(image), mask, 8, train=True)
        red = x[0] > x[2]
        assert torch.equal(red, m == 1)


def make_fake_mapillary(root, n=5, names=("road", "car", "object--traffic-sign--front", "Unlabeled")):
    labels = [{"name": n_, "color": [i * 40, 0, 0]} for i, n_ in enumerate(names)]
    (root / "config_v1.2.json").write_text(json.dumps({"labels": labels}))
    images, masks = root / "validation" / "images", root / "validation" / "v1.2" / "labels"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    palette = [c for lab in labels for c in lab["color"]]
    for i in range(n):
        Image.fromarray(np.full((20, 30, 3), i * 20, dtype=np.uint8)).save(images / f"img{i}.jpg")
        label = Image.fromarray(np.arange(20 * 30).reshape(20, 30).astype(np.uint8) % len(names), mode="P")
        label.putpalette(palette)  # palette PNG: pixel value = class id, like the real data
        label.save(masks / f"img{i}.png")


def test_mapillary_all_classes(tmp_path):
    make_fake_mapillary(tmp_path)
    train, val, info = build_datasets(DataConfig(dataset="mapillary", root=str(tmp_path), image_size=16))
    assert info.class_names == ["road", "car", "object--traffic-sign--front"]  # unlabeled dropped
    assert len(train) + len(val) == 5 and len(val) >= 1
    x, m = train[0]
    assert x.shape == (3, 16, 16)
    assert set(m.unique().tolist()) == {0, 1, 2, IGNORE_INDEX}


def test_mapillary_keep_traffic_signs(tmp_path):
    make_fake_mapillary(tmp_path)
    cfg = DataConfig(
        dataset="mapillary", root=str(tmp_path), image_size=16, classes=["object--traffic-sign--front"]
    )
    _, val, info = build_datasets(cfg)
    assert info.class_names == ["background", "object--traffic-sign--front"]
    assert set(val[0][1].unique().tolist()) == {0, 1}


def test_mapillary_unknown_class_is_an_error(tmp_path):
    make_fake_mapillary(tmp_path)
    with pytest.raises(ValueError, match="unknown"):
        mapillary_info(tmp_path, ["not-a-class"])


def test_mapillary_without_config_uses_v12_ids(tmp_path):
    info, lut = mapillary_info(tmp_path, None)
    assert info.num_classes == 65 and lut[64] == 64 and lut[65] == IGNORE_INDEX
    with pytest.raises(FileNotFoundError):
        mapillary_info(tmp_path, ["object--traffic-sign--front"])
