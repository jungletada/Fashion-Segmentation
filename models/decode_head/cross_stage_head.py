from models.backbone.fashion_modules import CrossAttention
from models.decode_head.upernet import ConvModule, PPM
from timm.models.layers import to_2tuple
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import torch.nn as nn
import torch


class SegModule(nn.Module):
    def __init__(self, dim, num_cls, emb_dim=128):
        super(SegModule, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

        self.proj_segmap = nn.Sequential(
            nn.Conv2d(num_cls, emb_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.gamma = nn.Conv2d(emb_dim, dim, kernel_size=3, stride=1, padding=1)
        self.beta = nn.Conv2d(emb_dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, segmap):
        identity = x
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        segmap = self.proj_segmap(segmap)
        x = self.conv(x)
        gamma = self.gamma(segmap)
        beta = self.beta(segmap)
        x = self.norm(x) * (1 + gamma) + beta
        return self.act(x) + identity


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super(AttentionModule, self).__init__()
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1)
        self.gamma = nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,groups=dim//2)
        self.beta = nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,groups=dim//2)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(self.conv1x1(x))
        a = x
        x = self.conv(x)
        gamma = self.gamma(a)
        beta = self.beta(a)
        x = x * (1 + gamma) + beta
        return self.act(x)


class SpadeMask(nn.Module):
    def __init__(self, num_classes, in_dims, dim=512, out_size=(56, 56)):
        super(SpadeMask, self).__init__()
        self.out_size = out_size
        self.stages = 4
        self.linears = nn.ModuleList(
            [nn.Conv2d(in_dims[0], dim, kernel_size=1),
             nn.Conv2d(in_dims[1], dim, kernel_size=1),
             nn.Conv2d(in_dims[2], dim, kernel_size=1),
             nn.Conv2d(in_dims[3], dim, kernel_size=1)]
        )
        self.conv_modules = nn.ModuleList(
            [AttentionModule(dim),
             AttentionModule(dim),
             AttentionModule(dim),
             AttentionModule(dim),]
        )
        self.fuse = ConvModule(dim * 4, dim, kernal_size=1)
        self.seg_module = SegModule(dim=dim, emb_dim=128, num_cls=num_classes)

    def forward(self, inputs, segmap):
        c0, c1, c2, c3 = inputs
        segmap = F.interpolate(segmap, size=c0.size()[2:], mode='nearest')

        c3 = self.linears[3](c3)
        c3 = self.conv_modules[3](c3)
        c3 = F.interpolate(c3, size=c0.size()[2:], mode='bilinear', align_corners=True)

        c2 = self.linears[2](c2)
        c2 = self.conv_modules[2](c2)
        c2 = F.interpolate(c2, size=c0.size()[2:], mode='bilinear', align_corners=True)

        c1 = self.linears[1](c1)
        c1 = self.conv_modules[1](c1)
        c1 = F.interpolate(c1, size=c0.size()[2:], mode='bilinear', align_corners=True)

        c0 = self.linears[0](c0)
        c0 = self.conv_modules[0](c0)

        output = self.fuse(torch.cat((c0, c1, c2, c3), dim=1))
        output = self.seg_module(output, segmap)
        return output


class CrossStageFashion(nn.Module):
    def __init__(self, emb_dims, num_heads, num_classes=14, img_size=224,
                 mask_dim=256, pool_scales=(1, 2, 3, 6)):
        super(CrossStageFashion, self).__init__()
        self.attention1_2 = CrossAttention(emb_dims[0], emb_dims[1], num_heads[0])
        self.attention2_3 = CrossAttention(emb_dims[1], emb_dims[2], num_heads[1])
        self.attention3_4 = CrossAttention(emb_dims[2], emb_dims[3], num_heads[2])
        self.ppm_module = PPM(pool_scales=pool_scales, in_channels=emb_dims[3], channels=emb_dims[3])
        self.out_mask = SpadeMask(num_classes, emb_dims, dim=mask_dim, out_size=to_2tuple(img_size // 4))
        dim = (len(pool_scales) + 1) * emb_dims[3]
        self.bottleneck = ConvModule(dim, emb_dims[3], 3, padding=1)
        self.img_size = img_size
        self.act = nn.GELU()

        self.norms = nn.ModuleList([
            nn.LayerNorm(emb_dims[0]),
            nn.LayerNorm(emb_dims[1]),
            nn.LayerNorm(emb_dims[2]),
        ])

    def psp_forward(self, x4):
        """Forward function of PSP module."""
        x4 = rearrange(x4, 'B (H W) C -> B C H W', H=self.img_size // 32, W=self.img_size // 32)
        psp_outs = [x4]
        psp_outs.extend(self.ppm_module(x4))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def forward(self, inputs, seg_map):
        c1, c2, c3, c4 = inputs
        x1 = self.attention1_2(c1, c2, H=self.img_size // 4, W=self.img_size // 4)
        x1 = self.act(self.norms[0](x1))
        x1 = rearrange(x1, 'B (H W) C -> B C H W', H=self.img_size // 4, W=self.img_size // 4)

        x2 = self.attention2_3(c2, c3, H=self.img_size // 8, W=self.img_size // 8)
        x2 = self.act(self.norms[1](x2))
        x2 = rearrange(x2, 'B (H W) C -> B C H W', H=self.img_size // 8, W=self.img_size // 8)

        x3 = self.attention3_4(c3, c4, H=self.img_size // 16, W=self.img_size // 16)
        x3 = self.act(self.norms[2](x3))
        x3 = rearrange(x3, 'B (H W) C -> B C H W', H=self.img_size // 16, W=self.img_size // 16)

        x4 = self.psp_forward(c4)

        out = self.out_mask([x1, x2, x3, x4], seg_map)

        return out


if __name__ == '__main__':
    # c1 = torch.randn(4, 56*56, 64)
    # c2 = torch.randn(4, 28*28, 128)
    # c3 = torch.randn(4, 14*14, 256)
    # c4 = torch.randn(4, 7*7, 512)
    # seg_map = torch.randn(4, 14, 56, 56)
    # inputs = (c1, c2, c3, c4)
    # module = CrossStageFashion(emb_dims=(64, 128, 256, 512),
    #                            num_heads=(4, 8, 16, 32), mask_dim=384,
    #                            img_size=224)
    # out = module.forward(inputs, seg_map)
    # for items in out:
    #     print(items.shape)
    x = torch.randn(4, 64, 56, 56)
    # params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    # print("Params: {:.2f}M".format(params / 1e6))