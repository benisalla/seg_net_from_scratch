# legacy/

`3dsf-traffic-signs-segmentation.ipynb` is the original 2023 Kaggle notebook
(TensorFlow / Keras) this project started from. It is kept for reference only;
the maintained code is the PyTorch package in [`src/segnet`](../src/segnet).

## What changed, and why

Problems found in the notebook, and how the package handles them:

| Notebook | Effect | Now |
|---|---|---|
| `train_ds = val_ds.map(...)` overwrote the training set | the model trained on the 10% validation split only | separate train / validation datasets |
| `tf.image.decode_png(mask, channels=1)` on Mapillary label PNGs | the labels are *palette* PNGs (pixel = class id); decoding to grayscale turns ids into brightness values, i.e. wrong labels | read with PIL in palette mode, ids mapped through a lookup table |
| `mask -= 1` on a `uint8` mask | class 0 wraps around to 255 | no shift; "unlabeled" pixels are explicitly ignored |
| one `RandomFlip` layer called separately on image and mask | each call draws its own flip, so image and mask were often flipped differently, so labels no longer matched pixels | one random draw flips both |
| 1 output channel + sigmoid + MSE loss on 66-class labels | regression on class ids; loss stayed between 12,000 and 15,000 | `num_classes` logits + cross-entropy (optional soft dice) |
| "SegNet" model was a plain conv / transposed-conv network | none of SegNet's pooling-index unpooling | real SegNet: VGG16-BN encoder, max-unpooling with the saved indices |
| `lr=1e-5`, `weight_decay=0.1`, 2000 epochs, `.repeat()` dataset with no `steps_per_epoch` | the inner loop never ended, so "epochs" never advanced | AdamW, warmup + cosine schedule, finite epochs, validation each epoch |
| no metrics | no way to tell if it learned | validation pixel accuracy and mean IoU, best checkpoint saved |
