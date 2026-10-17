"""SegNet shape and gradient checks. CPU only, tiny widths."""

import pytest
import torch

from segnet.model import SegNet, count_parameters


@pytest.mark.parametrize("size", [(32, 32), (64, 64), (45, 71)])
def test_output_has_input_size(size):
    model = SegNet(num_classes=5, base_channels=4)
    assert model(torch.randn(2, 3, *size)).shape == (2, 5, *size)


def test_full_width_matches_paper_size():
    # the SegNet paper reports 29.5M trainable parameters
    assert 29.4e6 < count_parameters(SegNet(num_classes=11, base_channels=64)) < 29.6e6


def test_every_parameter_gets_a_gradient():
    model = SegNet(num_classes=3, base_channels=4)
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not missing


def test_unpooling_puts_values_back_at_max_positions():
    stage_out = {}
    model = SegNet(num_classes=2, base_channels=4).eval()
    model.decoder[-1].unpool.register_forward_hook(lambda m, i, o: stage_out.setdefault("x", o))
    with torch.no_grad():
        model(torch.randn(1, 3, 32, 32))
    # in each 2x2 window at most one value is non-zero after max-unpooling
    windows = stage_out["x"].unfold(2, 2, 2).unfold(3, 2, 2).reshape(-1, 4)
    assert ((windows != 0).sum(1) <= 1).all()


def test_batch_items_are_independent_in_eval():
    model = SegNet(num_classes=3, base_channels=4).eval()
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        full = model(x)
        single = model(x[1:2])
    torch.testing.assert_close(full[1:2], single, atol=1e-5, rtol=1e-4)
