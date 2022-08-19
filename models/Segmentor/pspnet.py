import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.sync_batchnorm import SynchronizedBatchNorm2d
from models.decode_head.psp import PSPHead
from models.decode_head.fcn_head import FCNHead
from models.backbone.resnetv1c import Resnet101


class PSPNet(nn.Module):
    def __init__(self, num_classes=14, BatchNorm=nn.BatchNorm2d):
        super(PSPNet, self).__init__()
        self.backbone = Resnet101(BatchNorm)
        self.decode_head = PSPHead(in_channels=2048, channels=512, num_classes=num_classes)
        self.aux_head = FCNHead(in_channels=1024, channels=256, num_classes=num_classes)

    def forward(self, input):
        c1, c2, c3, c4 = self.backbone(input)
        out = self.decode_head(c4)
        aux_out = self.aux_head(c3)
        out = F.interpolate(out, size=input.size()[2:], mode='bilinear', align_corners=False)
        aux_out = F.interpolate(aux_out, size=input.size()[2:], mode='bilinear', align_corners=False)
        return out, aux_out


if __name__ == "__main__":
    model = PSPNet(num_classes=14)
    model.eval()
    input = torch.rand(2, 3, 224, 224)
    outputs = model(input)
    for output in outputs:
        print(output.size())