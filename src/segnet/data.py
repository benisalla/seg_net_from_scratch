"""Datasets: every item is (image float tensor (3, H, W), mask long tensor (H, W)).

- ``shapes``    synthetic circles / squares / triangles, generated on the fly
                (no download; used by the tests and the quick sanity checks)
- ``pets``      Oxford-IIIT Pet, pet vs background (simple real data, auto-download)
- ``mapillary`` Mapillary Vistas v1.2 street scenes, 66 classes (the real target)

Pixels whose mask value is ``IGNORE_INDEX`` do not count in the loss or metrics.
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import Dataset, Subset

from .config import DataConfig

IGNORE_INDEX = 255
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass
class DatasetInfo:
    class_names: list[str]
    colors: list[tuple[int, int, int]]  # one RGB color per class, for drawing masks

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


# --- shared image / mask transforms ------------------------------------------------


def to_tensor(image: Image.Image) -> Tensor:
    """PIL RGB image -> normalized float tensor (3, H, W)."""
    x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x - mean) / std


def denormalize(x: Tensor) -> Tensor:
    """Inverse of to_tensor's normalization, clamped to [0, 1] (for plotting)."""
    mean = torch.tensor(MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def joint_transform(image: Image.Image, mask: np.ndarray, size: int, train: bool) -> tuple[Tensor, Tensor]:
    """Resize both to size x size (bilinear image, nearest mask) and, when
    training, flip both with the *same* random draw so they stay aligned."""
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    mask_img = Image.fromarray(mask.astype(np.uint8)).resize((size, size), Image.NEAREST)
    if train and random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)
    return to_tensor(image), torch.from_numpy(np.array(mask_img, dtype=np.int64))


# --- synthetic shapes -------------------------------------------------------------

SHAPES_INFO = DatasetInfo(
    class_names=["background", "circle", "square", "triangle"],
    colors=[(0, 0, 0), (230, 25, 75), (60, 180, 75), (0, 130, 200)],
)


class ShapesDataset(Dataset):
    """Random colored shapes on a noisy background. Item i is always the same
    image (seeded by seed + i), so train and validation sets never overlap
    when they use different seeds."""

    def __init__(self, length: int, size: int = 64, seed: int = 0, max_shapes: int = 3) -> None:
        self.length = length
        self.size = size
        self.seed = seed
        self.max_shapes = max_shapes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        rng = random.Random(self.seed * 1_000_003 + index)
        s = self.size
        background = tuple(rng.randint(0, 255) for _ in range(3))
        image = Image.new("RGB", (s, s), background)
        mask = Image.new("L", (s, s), 0)
        draw_img, draw_mask = ImageDraw.Draw(image), ImageDraw.Draw(mask)
        for _ in range(rng.randint(1, self.max_shapes)):
            cls = rng.randint(1, 3)
            r = rng.uniform(0.1, 0.22) * s
            cx, cy = rng.uniform(r, s - r), rng.uniform(r, s - r)
            color = tuple(rng.randint(0, 255) for _ in range(3))
            if cls == 1:
                shape = [cx - r, cy - r, cx + r, cy + r]
                draw_img.ellipse(shape, fill=color)
                draw_mask.ellipse(shape, fill=cls)
            elif cls == 2:
                shape = [cx - r, cy - r, cx + r, cy + r]
                draw_img.rectangle(shape, fill=color)
                draw_mask.rectangle(shape, fill=cls)
            else:
                a = rng.uniform(0, 2 * math.pi)
                shape = [
                    (cx + r * math.cos(a + k * 2 * math.pi / 3), cy + r * math.sin(a + k * 2 * math.pi / 3))
                    for k in range(3)
                ]
                draw_img.polygon(shape, fill=color)
                draw_mask.polygon(shape, fill=cls)
        noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 12, (s, s, 3))
        pixels = np.clip(np.asarray(image, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
        return to_tensor(Image.fromarray(pixels)), torch.from_numpy(np.array(mask, dtype=np.int64))


# --- Oxford-IIIT Pet --------------------------------------------------------------

PETS_INFO = DatasetInfo(class_names=["background", "pet"], colors=[(0, 0, 0), (255, 200, 0)])


class PetsDataset(Dataset):
    """Oxford-IIIT Pet trimaps: 1 = pet, 2 = background, 3 = border (ignored)."""

    def __init__(self, root: str, split: str, size: int, train: bool, download: bool = True) -> None:
        from torchvision.datasets import OxfordIIITPet

        self.base = OxfordIIITPet(root, split=split, target_types="segmentation", download=download)
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, trimap = self.base[index]
        trimap = np.array(trimap)
        mask = np.full(trimap.shape, IGNORE_INDEX, dtype=np.uint8)
        mask[trimap == 1] = 1
        mask[trimap == 2] = 0
        return joint_transform(image, mask, self.size, self.train)


# --- Mapillary Vistas ---------------------------------------------------------------

MAPILLARY_V12_CLASSES = 65  # + 1 "unlabeled" id


def mapillary_info(root: Path, keep: list[str] | None) -> tuple[DatasetInfo, np.ndarray]:
    """Class names / colors from Mapillary's config_v1.2.json, and a lookup table
    raw label id -> training id.

    Without ``keep`` all 65 real classes are used and "unlabeled" is ignored.
    With ``keep`` (e.g. ["object--traffic-sign--front"]) those classes become
    1..k and everything else is background 0: e.g. traffic-sign segmentation.
    Without the config file, generic names and colors are used.
    """
    config = next(
        (
            p
            for p in [root / "config_v1.2.json", root.parent / "config_v1.2.json", root / "config.json"]
            if p.exists()
        ),
        None,
    )
    if config is not None:
        labels = json.loads(config.read_text())["labels"]
        names = [lab["name"] for lab in labels]
        colors = [tuple(lab["color"]) for lab in labels]
    elif keep:
        raise FileNotFoundError(f"--classes needs config_v1.2.json (class names) in {root} or its parent")
    else:  # v1.2 layout without its config: ids 0..64 are classes, 65 is unlabeled
        names = [f"class_{i}" for i in range(MAPILLARY_V12_CLASSES)] + ["unlabeled"]
        rng = np.random.default_rng(0)
        colors = [tuple(int(c) for c in rng.integers(0, 256, 3)) for _ in names]
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)

    if keep:
        unknown = set(keep) - set(names)
        if unknown:
            raise ValueError(f"unknown Mapillary classes: {sorted(unknown)}")
        lut[: len(names)] = 0
        for new_id, name in enumerate(keep, start=1):
            lut[names.index(name)] = new_id
        info = DatasetInfo(["background", *keep], [(0, 0, 0)] + [colors[names.index(n)] for n in keep])
        return info, lut

    kept = [i for i, n in enumerate(names) if "unlabeled" not in n.lower()]
    for new_id, old_id in enumerate(kept):
        lut[old_id] = new_id
    return DatasetInfo([names[i] for i in kept], [colors[i] for i in kept]), lut


def find_mapillary_split(root: Path, split: str) -> tuple[Path, Path] | None:
    """(images dir, labels dir) for a split, accepting the usual layouts:
    <root>/<split>/images + <root>/<split>/v1.2/labels (or <split>/labels)."""
    images = root / split / "images"
    for labels in (root / split / "v1.2" / "labels", root / split / "labels"):
        if images.is_dir() and labels.is_dir():
            return images, labels
    return None


class MapillaryDataset(Dataset):
    """Pairs <images>/<id>.jpg with <labels>/<id>.png. The label PNGs are
    palette images whose pixel *index* is the class id, so they are read with
    PIL in "P" mode (never converted to RGB or grayscale)."""

    def __init__(self, images_dir: Path, labels_dir: Path, lut: np.ndarray, size: int, train: bool) -> None:
        labels = {p.stem: p for p in labels_dir.glob("*.png")}
        self.pairs = [(p, labels[p.stem]) for p in sorted(images_dir.glob("*.jpg")) if p.stem in labels]
        if not self.pairs:
            raise FileNotFoundError(f"no (jpg, png) pairs in {images_dir} / {labels_dir}")
        self.lut = lut
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path, label_path = self.pairs[index]
        raw = np.array(Image.open(label_path))
        if raw.ndim != 2:
            raise ValueError(f"{label_path}: expected a palette (class-id) PNG, got shape {raw.shape}")
        return joint_transform(Image.open(image_path), self.lut[raw], self.size, self.train)


# --- entry point ---------------------------------------------------------------------


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    order = list(range(n))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(n * val_fraction))
    return order[n_val:], order[:n_val]


def build_datasets(cfg: DataConfig) -> tuple[Dataset, Dataset, DatasetInfo]:
    """Train set, validation set and class info for cfg.dataset."""
    if cfg.dataset == "shapes":
        train = ShapesDataset(cfg.shapes_train_size, cfg.image_size, seed=cfg.seed)
        val = ShapesDataset(cfg.shapes_val_size, cfg.image_size, seed=cfg.seed + 1)
        return train, val, SHAPES_INFO

    if cfg.dataset == "pets":
        train = PetsDataset(cfg.root, "trainval", cfg.image_size, train=True)
        val = PetsDataset(cfg.root, "test", cfg.image_size, train=False)
        return train, val, PETS_INFO

    if cfg.dataset == "mapillary":
        root = Path(cfg.root)
        info, lut = mapillary_info(root, cfg.classes)
        tr, va = find_mapillary_split(root, "training"), find_mapillary_split(root, "validation")
        if tr and va:
            train = MapillaryDataset(*tr, lut, cfg.image_size, train=True)
            val = MapillaryDataset(*va, lut, cfg.image_size, train=False)
            return train, val, info
        only = tr or va
        if only is None:
            raise FileNotFoundError(f"no training/ or validation/ split found under {root}")
        # a single split (e.g. the validation-only Kaggle copy): hold out val_fraction of it
        train_full = MapillaryDataset(*only, lut, cfg.image_size, train=True)
        val_full = MapillaryDataset(*only, lut, cfg.image_size, train=False)
        tr_idx, va_idx = split_indices(len(train_full), cfg.val_fraction, cfg.seed)
        return Subset(train_full, tr_idx), Subset(val_full, va_idx), info

    raise ValueError(f"unknown dataset {cfg.dataset!r} (shapes, pets, mapillary)")
