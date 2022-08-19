# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from models.decode_head.upernet import ConvModule
from models.decode_head.upernet import PPM


class PSPHead(nn.Module):
    """Pyramid Scene Parsing Network.
    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, in_channels=2048,
                 channels=512,
                 pool_scales=(1, 2, 3, 6),
                 dropout_rate=0.1,
                 num_classes=14,
                 align_corners=False):

        super(PSPHead, self).__init__()
        assert isinstance(pool_scales, (list, tuple))

        self.psp_modules = PPM(
            pool_scales,
            in_channels,
            channels,
            align_corners=align_corners)

        self.bottleneck = ConvModule(
            in_channels + len(pool_scales) * channels,
            channels,
            3,
            padding=1,
        )
        self.cls_seg = nn.Sequential(
            nn.Dropout(dropout_rate) if dropout_rate != 0 else nn.Identity(),
            nn.Conv2d(channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
        )


    def forward(self, inputs):
        """Forward function."""
        psp_outs = [inputs]
        psp_outs.extend(self.psp_modules(inputs))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        output = self.cls_seg(output)
        return output

if __name__ == '__main__':
    module = PSPHead(in_channels=2048, channels=512, )
    x = torch.randn(2, 2048, 32, 32)
    x = module(x)
    print(x.shape)

    print(module)

