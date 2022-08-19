import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernal_size, stride=1, padding=0):
        super(ConvModule, self).__init__()
        self.unit = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernal_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.unit(x)


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
    """
    def __init__(self, pool_scales, in_channels, channels, align_corners=False):
        super(PPM, self).__init__()
        for pool_scale in pool_scales:
            self.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(in_channels,channels, 1,)
            ))
        self.align_corners = align_corners

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,size=x.size()[2:],mode='bilinear',align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)

        return ppm_outs


class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.
    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                 in_channels=[64,128,256,512],
                 channels=256,
                 pool_scales=(1, 2, 3, 6),
                 dropout_rate=0.1,
                 num_classes=14):

        super(UPerHead, self).__init__()
        # input channels: C1, C2, C3, C4
        self.psp_modules = PPM(
            pool_scales,
            in_channels[-1],
            channels)
        # last feature through PPM
        in_dim = in_channels[-1] + len(pool_scales) * channels
        # bottleneck for PPM features combination
        self.bottleneck = ConvModule(in_dim, channels, 3, padding=1)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channel in in_channels[:-1]:  # skip the top layer
            # lateral conv for each feature
            l_conv = ConvModule(in_channel,channels,kernal_size=1)

            # feature pyramid network for lateral features after l_conv
            fpn_conv = ConvModule(channels,channels,kernal_size=3,padding=1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # concatenate all conv lateral features
        self.fpn_bottleneck = ConvModule(len(in_channels) * channels, channels, 3, padding=1,)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate != 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)


    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""
        # inputs: C1, C2, C3, C4

        # build laterals for each C1, C2, C3 (except last one: C4)
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # add the C4 PPM result
        laterals.append(self.psp_forward(inputs))

        # build top-down path from stage 4 to stage 1
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape,
                mode='bilinear', align_corners=True)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)
        ]

        # append psp feature
        fpn_outs.append(laterals[-1])

        # upsample each fpn_out to 1/4 size of original
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:],
                mode='bilinear',align_corners=True)

        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.dropout(output)
        output = self.conv_seg(output)
        # output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=True)
        return output


if __name__ == '__main__':
    import torch
    B = 2
    c1, c2, c3, c4 = torch.ones(B, 128, 56, 56), torch.ones(B, 256, 28, 28), \
                     torch.ones(B, 512, 14, 14), torch.ones(B, 1024, 7, 7)

    # ppm = PPM(in_channels=512, channels=256, pool_scales=(1,2,3,6))
    uperhead = UPerHead(in_channels=[128,256,512,1024],channels=512,pool_scales=(1,2,3,6))
    with torch.no_grad():
        out = uperhead((c1, c2, c3, c4))
    print(out.shape)