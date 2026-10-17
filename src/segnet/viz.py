"""Plotting helpers: colored masks, image | truth | prediction grids, training curves."""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .data import IGNORE_INDEX, DatasetInfo, denormalize


def colorize(mask: Tensor | np.ndarray, colors: list[tuple[int, int, int]]) -> np.ndarray:
    """(H, W) class ids -> (H, W, 3) uint8 RGB. Ignored pixels are drawn white."""
    mask = np.asarray(mask.cpu() if isinstance(mask, Tensor) else mask)
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[: len(colors)] = colors
    palette[IGNORE_INDEX] = (255, 255, 255)
    return palette[mask]


def save_predictions(
    images: Tensor, masks: Tensor, preds: Tensor, info: DatasetInfo, path: str | Path
) -> None:
    """One row per sample: input image, ground-truth mask, predicted mask."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for i in range(n):
        panels = [
            denormalize(images[i]).permute(1, 2, 0).cpu().numpy(),
            colorize(masks[i], info.colors),
            colorize(preds[i], info.colors),
        ]
        for ax, panel, title in zip(axes[i], panels, ["image", "ground truth", "prediction"]):
            ax.imshow(panel)
            ax.set_title(title if i == 0 else "")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_history(history: list[dict], path: str | Path) -> None:
    """Loss and validation mIoU / pixel accuracy per epoch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax1.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax1.legend()
    ax2.plot(epochs, [h["miou"] for h in history], label="mIoU")
    ax2.plot(epochs, [h["pixel_acc"] for h in history], label="pixel accuracy")
    ax2.set(xlabel="epoch", ylim=(0, 1), title="Validation metrics")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


@torch.no_grad()
def predict_batch(model: torch.nn.Module, images: Tensor) -> Tensor:
    model.eval()
    return model(images).argmax(dim=1)
