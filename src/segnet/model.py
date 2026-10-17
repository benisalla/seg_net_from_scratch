"""SegNet (Badrinarayanan et al., 2017) written from scratch in PyTorch.

Encoder: the 13 conv layers of VGG16 (conv-BN-ReLU), split into 5 stages, each
ending with a 2x2 max-pool that also returns the *indices* of the max values.
Decoder: the mirror image. Each stage first *unpools* with the indices saved by
the matching encoder stage (the values go back to where the max came from, the
rest stays zero), then densifies the sparse map with conv-BN-ReLU layers.
A final 1x1 conv gives one logit per class and pixel.

Re-using pooling indices instead of learning upsampling (or copying full encoder
feature maps like U-Net) is the defining idea of SegNet: boundaries stay sharp
and the decoder needs very little memory.
"""

from torch import Tensor, nn

# (number of conv layers, channel multiplier of base_channels) per encoder stage, as in VGG16
VGG16_STAGES = ((2, 1), (2, 2), (3, 4), (3, 8), (3, 8))


def conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class EncoderStage(nn.Module):
    """n x (conv-BN-ReLU), then 2x2 max-pool that remembers where the max was."""

    def __init__(self, in_channels: int, out_channels: int, num_convs: int) -> None:
        super().__init__()
        layers = [conv_bn_relu(in_channels, out_channels)]
        layers += [conv_bn_relu(out_channels, out_channels) for _ in range(num_convs - 1)]
        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, tuple[int, int]]:
        x = self.convs(x)
        size = x.shape[-2:]  # kept so unpooling restores odd sizes exactly
        x, indices = self.pool(x)
        return x, indices, size


class DecoderStage(nn.Module):
    """Max-unpool with the encoder's indices, then n x (conv-BN-ReLU)."""

    def __init__(self, in_channels: int, out_channels: int, num_convs: int) -> None:
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        layers = [conv_bn_relu(in_channels, in_channels) for _ in range(num_convs - 1)]
        layers.append(conv_bn_relu(in_channels, out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x: Tensor, indices: Tensor, size: tuple[int, int]) -> Tensor:
        x = self.unpool(x, indices, output_size=size)
        return self.convs(x)


class SegNet(nn.Module):
    """SegNet for semantic segmentation.

    Args:
        num_classes: number of output classes (one logit map each).
        in_channels: image channels (3 for RGB).
        base_channels: width of the first stage. 64 is the original VGG16 SegNet
            (~29M parameters); 16 or 32 give a small model for CPUs and toy data.
        dropout: dropout on the deepest feature map (0 disables it).

    Input height and width can be any size >= 32; output has the same size.
    Returns raw logits of shape (batch, num_classes, H, W).
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        base_channels: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        widths = [base_channels * mult for _, mult in VGG16_STAGES]

        self.encoder = nn.ModuleList()
        prev = in_channels
        for (num_convs, _), width in zip(VGG16_STAGES, widths):
            self.encoder.append(EncoderStage(prev, width, num_convs))
            prev = width

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # decoder stage i mirrors encoder stage i: width[i] in, width[i-1] out
        # (the last decoder stage keeps base_channels; the classifier comes after it)
        self.decoder = nn.ModuleList()
        outs = [widths[0]] + widths[:-1]
        for (num_convs, _), width, out in reversed(list(zip(VGG16_STAGES, widths, outs))):
            self.decoder.append(DecoderStage(width, out, num_convs))

        self.classifier = nn.Conv2d(widths[0], num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        saved = []
        for stage in self.encoder:
            x, indices, size = stage(x)
            saved.append((indices, size))
        x = self.dropout(x)
        for stage, (indices, size) in zip(self.decoder, reversed(saved)):
            x = stage(x, indices, size)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
