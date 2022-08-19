from models.backbone.resnetv1c import ResNetv1c
from models.backbone.resnetv1c import Bottleneck
from models.decode_head.ocr_head import OCRHead
from models.decode_head.fcn_head import FCNHead
import torch.nn.functional as F
import torch.nn as nn


class OCRNet(nn.Module):
    def __init__(self, num_classes=14):
        super(OCRNet, self).__init__()
        self.backbone = ResNetv1c(Bottleneck,
                                  layers=[3, 4, 23, 3],
                                  BatchNorm=nn.BatchNorm2d,
                                  strides=[1, 2, 1, 1])
        self.aux = FCNHead(in_channels=1024, channels=256, num_classes=num_classes)
        self.head = OCRHead(num_class=num_classes,
                            ocr_channels=256,
                            scale=1,
                            in_channels=2048,
                            channels=512)

    def forward(self, inputs):
        out_size = inputs.shape[2:]
        c1, c2, c3, c4 = self.backbone(inputs)
        prev = self.aux(c3)
        output = self.head(c4, prev)
        prev = F.interpolate(prev, size=out_size, mode='bilinear', align_corners=False)
        output = F.interpolate(output, size=out_size, mode='bilinear', align_corners=False)
        return {'output': output, 'aux': prev}


if __name__ == '__main__':
    import torch
    x = torch.randn(4, 3, 224, 224)
    net = OCRNet(num_classes=14)
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("{}".format(params/1e6))

    out_dict = net(x)
    print(out_dict['output'].shape)
    print(out_dict['aux'].shape)
