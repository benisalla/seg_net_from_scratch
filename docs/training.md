# Training

Code: [`src/segnet/train.py`](../src/segnet/train.py).
`segnet-train --help` lists every flag. Each one maps to a field of
[`config.py`](../src/segnet/config.py).

## Datasets

| `--dataset` | what | classes | download |
|---|---|---|---|
| `shapes` | synthetic circles, squares and triangles on noisy backgrounds, generated on the fly | 4 | none |
| `pets` | [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/): pet vs background (border pixels ignored) | 2 | automatic, ~800 MB into `--root` |
| `mapillary` | [Mapillary Vistas v1.2](https://www.mapillary.com/dataset/vistas) street scenes | 65 (or those you pick) | manual (see below) |

Start with `shapes`, then `pets`. If those do not learn, nothing will, so they
are the fastest way to check a change.

### Mapillary Vistas

Download it from Mapillary (free registration), or on Kaggle add a copy of
it to the notebook (for example the validation-only `mapillaryvistasval`).
`--root` must contain:

```
<root>/
├── config_v1.2.json             # class names and colors (optional, see below)
├── training/images/*.jpg
├── training/v1.2/labels/*.png   # or training/labels/
├── validation/images/*.jpg
└── validation/v1.2/labels/*.png
```

- If only one split exists (the Kaggle validation-only copy), `--val-fraction`
  of it (default 10%) is held out for validation.
- The label PNGs are palette images: the pixel value is the class id. The
  "unlabeled" class is ignored.
- Without `config_v1.2.json` the 65 classes get generic names and colors.
- **Traffic signs only.** Keep some classes and turn every other pixel into
  background. This needs the config file:

  ```bash
  segnet-train --dataset mapillary --root data/mapillary \
               --classes object--traffic-sign--front object--traffic-sign--back \
               --dice-weight 0.5
  ```

## Recipes

```bash
# 1. sanity check: memorise 8 images (loss -> ~0, pixel accuracy -> ~1)
segnet-train --dataset shapes --image-size 64 --base-channels 16 --overfit 8 \
             --batch-size 8 --epochs 150 --lr 3e-3 --output-dir checkpoints/overfit

# 2. toy data, CPU, no download
segnet-train --dataset shapes --image-size 64 --base-channels 16 --epochs 20 \
             --lr 3e-3 --output-dir checkpoints/shapes

# 3. simple real data, one GPU
segnet-train --dataset pets --root data --image-size 128 --base-channels 32 \
             --epochs 30 --batch-size 32 --output-dir checkpoints/pets

# 4. street scenes, one GPU (original-size SegNet)
segnet-train --dataset mapillary --root data/mapillary --image-size 256 \
             --base-channels 64 --epochs 60 --batch-size 16 --dice-weight 0.5 \
             --output-dir checkpoints/mapillary

# continue an interrupted run (e.g. a Kaggle session that timed out)
segnet-train <same flags> --resume checkpoints/mapillary/last.pt
```

## What the run writes

In `--output-dir`:

| file | content |
|---|---|
| `best.pt` | weights with the best validation mIoU, plus configs and class names/colors |
| `last.pt` | the latest epoch, also optimizer and scheduler state for `--resume` |
| `history.json` | per epoch: train/val loss, pixel accuracy, mIoU, lr, seconds |
| `history.png` | loss and metric curves |
| `predictions.png` | image / ground truth / prediction for 4 validation images |

## Training details

- **Optimizer**: AdamW (`--lr 1e-3`, `--weight-decay 1e-4`) with gradient
  clipping (`--grad-clip 1.0`).
- **Schedule**: linear warmup for `--warmup-epochs`, then cosine decay to 1% of
  the base lr, updated every step.
- **Mixed precision**: float16 autocast on CUDA (`--no-amp` turns it off).
- **Augmentation**: random horizontal flip, applied to the image and its mask
  with the same random draw.
- **Input size**: images are resized to `--image-size` squared (bilinear for
  images, nearest for masks, so no new class ids appear).

## Kaggle / Colab

[`notebooks/train_kaggle_colab.ipynb`](../notebooks/train_kaggle_colab.ipynb)
installs this package from GitHub, finds the dataset, trains, and shows the
curves and predictions. Pick the dataset in its config cell and press
**Run all**.

## Inference

```bash
segnet-predict --checkpoint checkpoints/pets/best.pt --images cat.jpg dog.jpg
# -> outputs/cat_mask.png, outputs/cat_overlay.png, ...
```

From Python:

```python
from PIL import Image
from segnet.predict import segment
from segnet.train import load_model

model, info, configs = load_model("checkpoints/pets/best.pt")
mask = segment(model, Image.open("cat.jpg"), configs["data"]["image_size"], "cpu")
# mask: (H, W) numpy array of class ids; names in info.class_names
```
