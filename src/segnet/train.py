"""Train SegNet.  Examples:

    segnet-train --dataset shapes --base-channels 16 --epochs 15        # toy data, CPU, minutes
    segnet-train --dataset pets --root data --base-channels 32          # simple real data
    segnet-train --dataset mapillary --root /kaggle/input/mapillaryvistasval
    segnet-train --dataset shapes --overfit 8 --epochs 100              # sanity check

Writes to --output-dir: last.pt, best.pt (best validation mIoU), history.json,
history.png and predictions.png.
"""

import argparse
import json
import math
import random
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .config import DataConfig, ModelConfig, TrainConfig
from .data import DatasetInfo, build_datasets
from .metrics import ConfusionMatrix, SegmentationLoss
from .model import SegNet, count_parameters
from .viz import predict_batch, save_history, save_predictions


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup to 1, then cosine decay to 0.01 (multiplier of the base lr)."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))


def build_model(cfg: ModelConfig, num_classes: int) -> SegNet:
    return SegNet(num_classes, base_channels=cfg.base_channels, dropout=cfg.dropout)


@torch.no_grad()
def evaluate(model, loader, loss_fn, num_classes: int, device) -> dict[str, float]:
    model.eval()
    cm = ConfusionMatrix(num_classes)
    total, count = 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total += loss_fn(logits, masks).item() * len(images)
        count += len(images)
        cm.update(logits.argmax(1), masks)
    return {"val_loss": total / max(1, count), **cm.compute()}


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, best_miou, configs, info) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_miou": best_miou,
            "configs": {k: asdict(v) for k, v in configs.items()},
            "class_names": info.class_names,
            "colors": info.colors,
        },
        path,
    )


def load_model(path: str | Path, device="cpu") -> tuple[SegNet, DatasetInfo, dict]:
    """Rebuild a trained model from a checkpoint written by train()."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    info = DatasetInfo(ckpt["class_names"], [tuple(c) for c in ckpt["colors"]])
    model = build_model(ModelConfig(**ckpt["configs"]["model"]), info.num_classes)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), info, ckpt["configs"]


def train(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    datasets: tuple[Dataset, Dataset, DatasetInfo] | None = None,
    resume: str | None = None,
) -> dict:
    """Full training run. Returns the final history entry plus best_miou.
    ``datasets`` lets tests pass in their own (train, val, info)."""
    seed_everything(train_cfg.seed)
    device = resolve_device(train_cfg.device)
    out = Path(train_cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_set, val_set, info = datasets or build_datasets(data_cfg)
    if train_cfg.overfit > 0:  # memorise a handful of samples: loss must go to ~0
        train_set = val_set = Subset(train_set, range(min(train_cfg.overfit, len(train_set))))
    loader_kw = {
        "num_workers": data_cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": data_cfg.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=len(train_set) > train_cfg.batch_size,
        **loader_kw,
    )
    val_loader = DataLoader(val_set, batch_size=train_cfg.batch_size, **loader_kw)

    model = build_model(model_cfg, info.num_classes).to(device)
    loss_fn = SegmentationLoss(train_cfg.dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = train_cfg.epochs * steps_per_epoch
    warmup = train_cfg.warmup_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, warmup, total_steps))
    use_amp = train_cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    configs = {"data": data_cfg, "model": model_cfg, "train": train_cfg}
    history_path = out / "history.json"
    history, start_epoch, best_miou = [], 1, -1.0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_miou = ckpt["epoch"] + 1, ckpt["best_miou"]
        if history_path.exists():
            history = [h for h in json.loads(history_path.read_text()) if h["epoch"] < start_epoch]

    print(
        f"device={device}  params={count_parameters(model):,}  classes={info.num_classes}  "
        f"train={len(train_set)}  val={len(val_set)}  steps/epoch={steps_per_epoch}"
    )

    for epoch in range(start_epoch, train_cfg.epochs + 1):
        model.train()
        start, running, seen = time.time(), 0.0, 0
        for step, (images, masks) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
            loss = loss_fn(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(images)
            seen += len(images)
            if train_cfg.log_every and step % train_cfg.log_every == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {loss.item():.4f}")

        metrics = evaluate(model, val_loader, loss_fn, info.num_classes, device)
        entry = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            **metrics,
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - start, 1),
        }
        history.append(entry)
        print(
            f"epoch {epoch:3d} | train {entry['train_loss']:.4f} | val {entry['val_loss']:.4f} | "
            f"mIoU {entry['miou']:.3f} | acc {entry['pixel_acc']:.3f} | {entry['seconds']}s"
        )

        if entry["miou"] > best_miou:
            best_miou = entry["miou"]
            save_checkpoint(out / "best.pt", model, optimizer, scheduler, epoch, best_miou, configs, info)
        save_checkpoint(out / "last.pt", model, optimizer, scheduler, epoch, best_miou, configs, info)
        history_path.write_text(json.dumps(history, indent=2))

    if history:
        save_history(history, out / "history.png")
        images, masks = next(iter(DataLoader(val_set, batch_size=4)))
        preds = predict_batch(model, images.to(device)).cpu()
        save_predictions(images, masks, preds, info, out / "predictions.png")
    return {**(history[-1] if history else {}), "best_miou": best_miou}


# --- command line -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """One --flag per config field (underscores become dashes)."""
    parser = argparse.ArgumentParser(
        description="Train SegNet from scratch.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for group_name, cls in (("data", DataConfig), ("model", ModelConfig), ("train", TrainConfig)):
        group = parser.add_argument_group(group_name)
        for f in fields(cls):
            if f.name == "seed" and cls is DataConfig:
                continue  # one --seed sets both
            flag = "--" + f.name.replace("_", "-")
            if isinstance(f.default, bool):
                group.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
            elif f.name == "classes":
                group.add_argument(
                    flag,
                    nargs="+",
                    default=None,
                    help="mapillary: class names to keep, e.g. object--traffic-sign--front",
                )
            else:
                group.add_argument(flag, type=type(f.default), default=f.default)
    parser.add_argument("--resume", default=None, help="continue from a last.pt checkpoint")
    return parser


def configs_from_args(args: argparse.Namespace) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    values = vars(args)
    data = DataConfig(**{f.name: values[f.name] for f in fields(DataConfig) if f.name in values})
    data.seed = args.seed
    model = ModelConfig(**{f.name: values[f.name] for f in fields(ModelConfig)})
    train_cfg = TrainConfig(**{f.name: values[f.name] for f in fields(TrainConfig)})
    return data, model, train_cfg


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data, model, train_cfg = configs_from_args(args)
    result = train(data, model, train_cfg, resume=args.resume)
    print(f"done. best mIoU {result['best_miou']:.3f}. checkpoints in {train_cfg.output_dir}")


if __name__ == "__main__":
    main()
