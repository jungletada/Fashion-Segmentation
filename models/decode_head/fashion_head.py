import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath
from models.backbone.convnext import LayerNorm
from models.decode_head.upernet import ConvModule
from models.decode_head.psp import PPM


class ChannelAttention(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(ChannelAttention, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        self.norm = LayerNorm(normalized_shape=in_dim, data_format='channels_first')

    def forward(self,x):
        H, W = x.size()[2:]
        x = self.norm(x)
        proj_query = rearrange(x, 'B C H W->B C (H W)', H=H, W=W)
        proj_key = rearrange(x, 'B C H W->B (H W) C', H=H, W=W)
        
        energy = torch.bmm(proj_query, proj_key) # -> B C C'
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = rearrange(x, 'B C H W->B C (H W)', H=H, W=W)
        out = torch.bmm(attention, proj_value)
        out = rearrange(out, 'B C (H W)->B C H W', H=H, W=W)
        out = self.gamma * out
        return out


class ConvNextBlock(nn.Module):
    def __init__(self, dim, drop_path=0.1, scale_value=1.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(scale_value * torch.ones((dim)),
                                  requires_grad=True) if scale_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.dwconv(x)
        x = rearrange(x, 'B C H W -> B H W C')
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = rearrange(x, 'B H W C -> B C H W')
        x = self.drop_path(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Upsample, self).__init__()
        self.proj = nn.Conv2d(in_dim, 4 * out_dim, kernel_size=1, stride=1)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.pixelshuffle(x)
        x = self.norm(x)
        return x


class FashionHead(nn.Module):
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

        super(FashionHead, self).__init__()

        self.psp_modules = PPM(pool_scales,in_channels[-1],channels)
        # last feature through PPM
        last_in_dim = in_channels[-1] + len(pool_scales) * channels
        # bottleneck for PPM features combination
        self.bottleneck = ConvModule(last_in_dim, channels, 3, padding=1)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channel in in_channels[:-1]:  # skip the top layer
            # lateral conv for each feature
            l_conv = ConvModule(in_channel,channels,kernal_size=1)
            # feature pyramid network for lateral features after l_conv
            fpn_conv = SpadeConvModule(channels, num_cls=num_classes)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # concatenate all conv lateral features
        self.fpn_bottleneck = ConvModule(len(in_channels) * channels, channels, 3, padding=1,)

        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate != 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)


    def forward_last(self, x):
        """Forward function of PSP module."""
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def forward(self, inputs, segmap):
        """Forward function."""
        # inputs: C1, C2, C3, C4
        # build laterals for each C1, C2, C3 (except last one: C4)
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # add the C4 PPM result
        laterals.append(self.forward_last(inputs[-1]))
        # build top-down path from stage 4 to stage 1
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape,
                mode='bilinear', align_corners=True)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i].forward(laterals[i], segmap) for i in range(used_backbone_levels - 1)
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
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=True)
        return output


if __name__ == '__main__':
    module = FashionHead()
    # module = SpadeConvModule(dim=384, num_cls=14)
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print("Total parameters: {:.2f}".format(params / 1e6))
    # x = (torch.randn(3, 64, 56, 56),
    #      torch.randn(3, 128, 28, 28),
    #      torch.randn(3, 256, 14, 14),
    #      torch.randn(3, 512, 7, 7),)
    #
    # segmap = torch.ones(3, 14, 224, 224)
    # out_feature = module.forward(x, segmap)
    # print(out_feature.shape)
    # net1 = ConvNextBlock(dim=256, drop_path=0.1, scale_value=1)
    # net2 = ConvModule(in_dim=256,out_dim=256, kernal_size=3,padding=1)
    # params1 = sum(p.numel() for p in net1.parameters() if p.requires_grad)
    # print("Total parameters1: {:.2f}".format(params1 / 1e6))
    # params2 = sum(p.numel() for p in net2.parameters() if p.requires_grad)
    # print("Total parameters1: {:.2f}".format(params2 / 1e6))