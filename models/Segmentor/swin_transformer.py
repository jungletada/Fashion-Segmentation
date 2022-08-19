from models.backbone.swin import swin_tiny_patch4_window7_224, \
    swin_base_patch4_window7_224, swin_small_patch4_window7_224
from models.decode_head.upernet import UPerHead

import torch.nn.functional as F
import torch.nn as nn


class swin(nn.Module):
    def __init__(self, varients='swin_t', num_classes=14):
        super(swin, self).__init__()
        if varients == 'swin_t':
            embed_dim = 96
            self.backbone = swin_tiny_patch4_window7_224(num_classes=num_classes)

        elif varients == 'swin_s':
            embed_dim = 96
            self.backbone = swin_small_patch4_window7_224(num_classes=num_classes)

        elif varients == 'swin_b':
            embed_dim = 128
            self.backbone = swin_base_patch4_window7_224(num_classes=num_classes)

        self.decoder = UPerHead(in_channels=[embed_dim * (2**i) for i in range(4)],
                                channels=512, num_classes=num_classes)

    def forward(self, x):
        size = x.size()[2:]
        features = self.backbone(x)
        out = self.decoder(features)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return {'output':out}


