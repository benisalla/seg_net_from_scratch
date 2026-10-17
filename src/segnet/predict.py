"""Segment images with a trained checkpoint.

    segnet-predict --checkpoint checkpoints/run/best.pt --images street.jpg more/*.jpg

For each image writes <name>_mask.png (colored classes, original resolution)
and <name>_overlay.png (mask blended over the photo) into --output-dir.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import to_tensor
from .train import load_model, resolve_device
from .viz import colorize


@torch.no_grad()
def segment(model: torch.nn.Module, image: Image.Image, size: int, device) -> np.ndarray:
    """(H, W) class ids at the image's own resolution."""
    image = image.convert("RGB")
    x = to_tensor(image.resize((size, size), Image.BILINEAR)).unsqueeze(0).to(device)
    pred = model(x).argmax(1)[0].byte().cpu().numpy()
    return np.array(Image.fromarray(pred).resize(image.size, Image.NEAREST))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Segment images with a trained SegNet.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--image-size", type=int, default=None, help="default: the training size")
    parser.add_argument("--alpha", type=float, default=0.5, help="overlay opacity")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    model, info, configs = load_model(args.checkpoint, device)
    size = args.image_size or configs["data"]["image_size"]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for path in map(Path, args.images):
        image = Image.open(path).convert("RGB")
        mask = colorize(segment(model, image, size, device), info.colors)
        overlay = (1 - args.alpha) * np.asarray(image, dtype=np.float32) + args.alpha * mask
        Image.fromarray(mask).save(out / f"{path.stem}_mask.png")
        Image.fromarray(overlay.astype(np.uint8)).save(out / f"{path.stem}_overlay.png")
        print(f"{path} -> {out / (path.stem + '_mask.png')}")


if __name__ == "__main__":
    main()
