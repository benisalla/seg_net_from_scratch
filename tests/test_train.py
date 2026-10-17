"""End-to-end training checks on tiny synthetic data. CPU only, about a minute."""

import json

import numpy as np
import torch
from PIL import Image

from segnet.config import DataConfig, ModelConfig, TrainConfig
from segnet.predict import main as predict_main
from segnet.train import build_parser, configs_from_args, load_model, lr_lambda, train


def tiny(tmp_path, image_size=32, **train_kw):
    data = DataConfig(
        dataset="shapes", image_size=image_size, shapes_train_size=16, shapes_val_size=8, num_workers=0
    )
    model = ModelConfig(base_channels=8)
    train_cfg = TrainConfig(
        batch_size=8,
        lr=3e-3,
        warmup_epochs=0,
        log_every=0,
        device="cpu",
        output_dir=str(tmp_path / "run"),
        **train_kw,
    )
    return data, model, train_cfg


def test_overfit_a_few_samples(tmp_path):
    """The classic sanity check: a model that cannot memorise 4 images is broken."""
    result = train(*tiny(tmp_path, image_size=64, epochs=150, overfit=4))
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert history[-1]["train_loss"] < 0.25 * history[0]["train_loss"]
    assert result["pixel_acc"] > 0.9


def test_outputs_resume_and_predict(tmp_path):
    data, model, train_cfg = tiny(tmp_path, epochs=2)
    train(data, model, train_cfg)
    run = tmp_path / "run"
    for name in ("best.pt", "last.pt", "history.json", "history.png", "predictions.png"):
        assert (run / name).exists(), name

    train_cfg.epochs = 3
    train(data, model, train_cfg, resume=str(run / "last.pt"))
    assert [h["epoch"] for h in json.loads((run / "history.json").read_text())] == [1, 2, 3]

    net, info, configs = load_model(run / "last.pt")
    assert info.num_classes == 4 and configs["data"]["image_size"] == 32
    assert net(torch.randn(1, 3, 32, 32)).shape == (1, 4, 32, 32)

    Image.fromarray(np.zeros((40, 50, 3), dtype=np.uint8)).save(tmp_path / "photo.jpg")
    predict_main(
        [
            "--checkpoint",
            str(run / "best.pt"),
            "--images",
            str(tmp_path / "photo.jpg"),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
        ]
    )
    assert Image.open(tmp_path / "out" / "photo_mask.png").size == (50, 40)


def test_cli_flags_reach_configs():
    args = build_parser().parse_args(
        ["--dataset", "mapillary", "--classes", "a", "b", "--base-channels", "32", "--no-amp", "--seed", "7"]
    )
    data, model, train_cfg = configs_from_args(args)
    assert data.dataset == "mapillary" and data.classes == ["a", "b"] and data.seed == 7
    assert model.base_channels == 32 and train_cfg.amp is False and train_cfg.seed == 7


def test_lr_schedule():
    assert lr_lambda(0, 10, 100) == 0.1
    assert lr_lambda(9, 10, 100) == 1.0
    assert abs(lr_lambda(100, 10, 100) - 0.01) < 1e-9
