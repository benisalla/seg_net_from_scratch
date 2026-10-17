# Testing

```bash
pip install -e '.[dev]'
pytest            # CPU only, no network
ruff check src tests
```

| file | what it checks |
|---|---|
| `tests/test_model.py` | output size equals input size (also odd sizes), full-width parameter count matches the paper (29.5M), every parameter gets a gradient, max-unpooling leaves at most one non-zero per 2×2 window, batch items are independent in eval mode |
| `tests/test_data.py` | shapes are deterministic and labelled, train/val differ, image and mask flip together, Mapillary palette labels → class ids (all classes, the `--classes` subset, missing config file) on a fake dataset written to disk |
| `tests/test_metrics.py` | pixel accuracy and IoU computed by hand, ignored pixels count in neither the loss nor the metrics, dice loss range |
| `tests/test_train.py` | **overfit sanity check** (a small SegNet memorises 4 images: loss falls by more than 4× and pixel accuracy goes above 90%), all output files are written, `--resume` continues the history, checkpoint reload and `segnet-predict` work, CLI flags reach the configs, lr schedule values |

`test_overfit_a_few_samples` is the slowest test (under a minute on an idle
CPU). It is also the most important one: a segmentation model that cannot
memorise four images has a bug in its data, loss or architecture.

## Checks on real data

The unit tests cannot show that the model learns real images. That is what
the two simple-data runs in the [README](../README.md#results) are for:
`shapes` (synthetic) and `pets` (Oxford-IIIT Pet). Run them again after
changing the model, loss or data pipeline:

```bash
segnet-train --dataset shapes --image-size 64 --base-channels 16 --shapes-train-size 1000 \
             --epochs 12 --lr 3e-3 --output-dir checkpoints/shapes
segnet-train --dataset pets --root data --image-size 96 --base-channels 16 \
             --epochs 6 --lr 3e-3 --output-dir checkpoints/pets
```
