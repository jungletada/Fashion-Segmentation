import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


class SemanticBranchFPN(nn.Module):
    def __init__(self, feature_strides, num_classes):
        """Panoptic Feature Pyramid Networks.
        This head is the implementation of `Semantic FPN
        for the SeMask Semantic Attention Part
        """
        super(SemanticBranchFPN, self).__init__()
        # Semantic Decoder
        self.feature_strides = feature_strides
        self.align_corners = False
        self.cls_scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            cls_scale_head = []
            for k in range(head_length):
                if feature_strides[i] != feature_strides[0]:
                    cls_scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.cls_scale_heads.append(nn.Sequential(*cls_scale_head))

    def forward(self, cls_x):
        # Semantic Decoder
        cls_output = cls_x[0]
        for i in range(1, len(self.feature_strides)):
            # non inplace
            cls_output = cls_output + F.interpolate(
                self.cls_scale_heads[i](cls_x[i]),
                size=cls_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return cls_output


class BranchFPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks.
    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.
    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """
    channels = 256
    def __init__(self, feature_strides, in_channels, num_classes):
        super(BranchFPNHead, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.align_corners = False
        self.cls_seg = nn.Sequential(
            nn.Conv2d(self.channels, num_classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.Dropout2d(p=0.1, inplace=False)
        )
        # Feature Decoder
        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append( # 1. ConvModule
                    ConvModule(
                        in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='ReLU'))
                )
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append( # 2. Upsample for feature map
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

        # Semantic Decoder
        self.cls_scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            cls_scale_head = []
            for k in range(head_length):
                if feature_strides[i] != feature_strides[0]:
                    cls_scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.cls_scale_heads.append(nn.Sequential(*cls_scale_head))

    def forward(self, x, cls_x):
        # Fearure FPN Decoder
        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + F.interpolate(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        output = self.cls_seg(output)

        # Semantic Decoder
        cls_output = cls_x[0]
        for i in range(1, len(self.feature_strides)):
            # non inplace
            cls_output = cls_output + F.interpolate(
                self.cls_scale_heads[i](cls_x[i]),
                size=cls_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        return output, cls_output


if __name__ == '__main__':
    feature_strides = [4, 8, 16, 32]
    in_channels = [96*(2**i) for i in range(4)]
    net = BranchFPNHead(feature_strides, in_channels, num_classes=14)
    print(net)
    import torch
    x = (torch.ones(4, 96, 56, 56),
         torch.ones(4,192, 28, 28),
         torch.ones(4,384,14, 14),
         torch.ones(4,768,7, 7))
    cls_x = [torch.ones(4, 14, 56, 56),
             torch.ones(4, 14, 28, 28),
             torch.ones(4, 14, 14, 14),
             torch.ones(4, 14, 7, 7)]
    output, cls_output = net(x, cls_x)

    print(output.shape)
    print(cls_output.shape)
