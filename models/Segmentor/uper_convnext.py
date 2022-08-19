from models.backbone.convnext import ConvNeXt
from models.decode_head.upernet import UPerHead
from models.decode_head.fcn_head import FCNHead
import torch.nn.functional as F
import torch.nn as nn
import torch


class Uper_ConvNeXt(nn.Module):
    def __init__(self, type='small', num_classes=14):
        super(Uper_ConvNeXt, self).__init__()
        if type == 'base':
            self.backbone = ConvNeXt(in_chans=3,
                                     depths=[3, 3, 27, 3],
                                     dims=[128, 256, 512, 1024],
                                     drop_path_rate=0.4,
                                     layer_scale_init_value=1.0,
                                     out_indices=[0, 1, 2, 3])
            self.decoder = UPerHead(in_channels=[128, 256, 512, 1024],
                                    channels=512,
                                    dropout_rate=0.1,
                                    num_classes=num_classes)
            self.aux_head = FCNHead(in_channels=512,
                                    channels=256,
                                    num_convs=1,
                                    num_classes=num_classes)

        elif type == 'small':
            self.backbone = ConvNeXt(in_chans=3,
                                     depths=[3, 3, 27, 3],
                                     dims=[96, 192, 384, 768],
                                     drop_path_rate=0.3,
                                     layer_scale_init_value=1.0,
                                     out_indices=[0, 1, 2, 3])
            self.decoder = UPerHead(in_channels=[96, 192, 384, 768],
                                    channels=512,
                                    dropout_rate=0.1,
                                    num_classes=num_classes)
            self.aux_head = FCNHead(in_channels=384,
                                    channels=256,
                                    num_convs=1,
                                    num_classes=num_classes)
        elif type == 'tiny':
            self.backbone = ConvNeXt(in_chans=3,
                                     depths=[3, 3, 9, 3],
                                     dims=[96, 192, 384, 768],
                                     drop_path_rate=0.4,
                                     layer_scale_init_value=1.0,
                                     out_indices=[0, 1, 2, 3])
            self.decoder = UPerHead(in_channels=[96, 192, 384, 768],
                                    channels=512,
                                    dropout_rate=0.1,
                                    num_classes=num_classes)
            self.aux_head = FCNHead(in_channels=384,
                                    channels=256,
                                    num_convs=1,
                                    num_classes=num_classes)

    def forward(self, x):
        f1,f2,f3,f4 = self.backbone(x)
        out = self.decoder((f1, f2, f3, f4))
        aux = self.aux_head(f3)
        aux = F.interpolate(aux, size=x.size()[2:], mode='bilinear', align_corners=False)
        return out, aux


if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)
    type = 'small'

    model = Uper_ConvNeXt(type=type)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("ConvNeXt-{}'s Totoal params: {:.2f}".format(type, params/1e6))

    with torch.no_grad():
        out, aux = model(x)
        print(out.shape)
        print(aux.shape)