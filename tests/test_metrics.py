import math

import torch

from segnet.data import IGNORE_INDEX
from segnet.metrics import ConfusionMatrix, SegmentationLoss, soft_dice_loss


def test_perfect_prediction_metrics():
    target = torch.tensor([[[0, 1], [2, 2]]])
    cm = ConfusionMatrix(3)
    cm.update(target.clone(), target)
    assert cm.compute() == {"pixel_acc": 1.0, "miou": 1.0}


def test_iou_by_hand():
    target = torch.tensor([[0, 0, 1, 1]])
    pred = torch.tensor([[0, 1, 1, 1]])
    cm = ConfusionMatrix(2)
    cm.update(pred, target)
    iou = cm.iou()
    assert math.isclose(iou[0].item(), 1 / 2) and math.isclose(iou[1].item(), 2 / 3)
    assert math.isclose(cm.compute()["pixel_acc"], 3 / 4)


def test_ignored_pixels_do_not_count():
    target = torch.tensor([[[0, IGNORE_INDEX]]])
    cm = ConfusionMatrix(2)
    cm.update(torch.tensor([[[0, 1]]]), target)
    assert cm.matrix.sum() == 1 and cm.compute()["pixel_acc"] == 1.0

    logits = torch.zeros(1, 2, 1, 2)
    logits[0, 0, 0, 0] = 10  # confident and right on the valid pixel
    assert SegmentationLoss(dice_weight=1.0)(logits, target) < 0.01


def test_dice_loss_range():
    target = torch.randint(0, 3, (2, 8, 8))
    perfect = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float() * 50
    assert soft_dice_loss(perfect, target) < 1e-3
    assert 0 < soft_dice_loss(torch.randn(2, 3, 8, 8), target) < 1
