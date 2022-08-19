import torch
import torch.nn as nn
from torch.nn.functional import upsample
from models.backbone.resnetv1c import Resnet101
from models.decode_head.da_head import DANetHead


class DANet(nn.Module):
    def __init__(self, num_classes):
        super(DANet, self).__init__()
        self.backbone = Resnet101(nn.BatchNorm2d)
        self.decode_head = DANetHead(2048, num_classes, norm_layer=nn.BatchNorm2d)


    def forward(self, x):
        imsize = x.size()[2:]
        _, _, _, c4 = self.backbone(x)

        x = self.decode_head(c4)
        x = list(x)
        x[0] = upsample(x[0], imsize, mode='bilinear', align_corners=False)
        x[1] = upsample(x[1], imsize, mode='bilinear', align_corners=False)
        x[2] = upsample(x[2], imsize, mode='bilinear', align_corners=False)

        outputs = {'output':x[0], 'aux1':x[1], 'aux2':x[2]}
        return tuple(outputs)

