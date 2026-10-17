"""All settings, grouped in three dataclasses. CLI flags in train.py map 1:1 to these fields."""

from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset: str = "shapes"  # shapes | pets | mapillary
    root: str = "data"  # download / dataset folder (pets, mapillary)
    image_size: int = 128
    val_fraction: float = 0.1  # mapillary with a single split only
    classes: list[str] | None = None  # mapillary: keep only these classes (others -> background)
    shapes_train_size: int = 2000
    shapes_val_size: int = 200
    num_workers: int = 2
    seed: int = 0


@dataclass
class ModelConfig:
    base_channels: int = 64  # 64 = original SegNet (29.5M params); 16/32 for small runs
    dropout: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    dice_weight: float = 0.0  # loss = cross-entropy + dice_weight * soft dice
    grad_clip: float = 1.0
    amp: bool = True  # mixed precision on CUDA
    device: str = "auto"
    output_dir: str = "checkpoints/run"
    overfit: int = 0  # >0: train and validate on this many training samples (sanity check)
    log_every: int = 50
    seed: int = 0
