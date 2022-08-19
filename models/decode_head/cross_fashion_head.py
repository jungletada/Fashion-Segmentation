from models.decode_head.upernet import PPM, ConvModule
from models.backbone.fashion_modules import CrossAttention
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn
import torch
import numpy as np


class LargeKernelConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=9, dilation=2):
        super(LargeKernelConv, self).__init__()
        kernel_dw = int(2 * dilation - 1)
        padding_dw = kernel_dw //2
        kernel_dw_d = int(np.ceil(kernel_size / dilation))
        padding_dw_d = ((kernel_dw_d  - 1) * dilation + 1) // 2

        self.mlp_prev = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.dw_conv = nn.Conv2d(out_dim, out_dim, kernel_size=kernel_dw,
                            stride=1,padding=padding_dw, groups=out_dim)
        self.dw_d_conv = nn.Conv2d(out_dim, out_dim, kernel_size=kernel_dw_d,
                            stride=1, padding=padding_dw_d, dilation=dilation, groups=out_dim)
        self.mlp = nn.Conv2d(out_dim, out_dim, kernel_size=1)

    def forward(self, x):
        x = self.mlp_prev(x)
        x = self.dw_conv(x)
        x = self.dw_d_conv(x)
        x = self.mlp(x)
        return x


class LKSpadeNorm(nn.Module):
    def __init__(self, in_dim, num_cls, emb_dim=128, kernel_size=7, dliation=3):
        super(LKSpadeNorm, self).__init__()
        self.conv_feat = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(in_dim)
        self.act = nn.GELU()
        self.mlp_segmap = nn.Sequential(
            LargeKernelConv(num_cls, emb_dim, kernel_size=kernel_size, dilation=dliation),
            nn.GELU())
        # self.mlp_scale = LargeKernelConv(emb_dim, in_dim, kernel_size=kernel_size, dilation=dliation)
        # self.mlp_bias = LargeKernelConv(emb_dim, in_dim, kernel_size=kernel_size, dilation=dliation)
        self.mlp_scale = nn.Conv2d(emb_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.mlp_bias = nn.Conv2d(emb_dim, in_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, segmap):
        shortcut = x
        x = self.bn(self.conv_feat(x))
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        segmap = self.mlp_segmap(segmap)
        x = self.act(x * (1 + self.mlp_scale(segmap)) + self.mlp_bias(segmap)) + shortcut
        return x


class SpadeNorm(nn.Module):
    def __init__(self, in_dim, num_cls, emb_dim=128):
        super(SpadeNorm, self).__init__()
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(num_cls, emb_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(emb_dim, in_dim, kernel_size=3,stride=1,padding=1)
        self.mlp_beta = nn.Conv2d(emb_dim, in_dim, kernel_size=3,stride=1,padding=1)
        self.param_free_norm = nn.BatchNorm2d(in_dim)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        h, w = x.size()[2:]
        segmap = F.interpolate(segmap, size=(2*h, 2*w), mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out


class SpadeConvModule(nn.Module):
    def __init__(self, dim, num_cls):
        super(SpadeConvModule, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.norm = SpadeNorm(in_dim=dim, num_cls=num_cls, emb_dim=128)
        self.act = nn.GELU()

    def forward(self, x, segmap):
        identity = x
        x = self.conv(x)
        x = self.norm(x, segmap)
        x = self.act(x)
        x = x + identity
        return x


class FashionMask(nn.Module):
    def __init__(self, num_classes, in_dims, dim=512, out_size=(56, 56)):
        super(FashionMask, self).__init__()
        self.out_size = out_size
        self.stages = 4
        self.linears = nn.ModuleList(
            [nn.Conv2d(in_dims[0], dim, kernel_size=1),
             nn.Conv2d(in_dims[1], dim, kernel_size=1),
             nn.Conv2d(in_dims[2], dim, kernel_size=1),
             nn.Conv2d(in_dims[3], dim, kernel_size=1)]
        )
        self.conv_modules = nn.ModuleList(
            [SpadeConvModule(dim, num_classes),
             SpadeConvModule(dim, num_classes),
             SpadeConvModule(dim, num_classes),
             SpadeConvModule(dim, num_classes)]
        )
        self.fuse = ConvModule(dim * 4, dim, kernal_size=1)

    def forward(self, inputs, seg_map):
        c0, c1, c2, c3 = inputs
        c3 = self.linears[3](c3)
        c3 = self.conv_modules[3](c3, seg_map)
        c3 = F.interpolate(c3, size=c0.size()[2:], mode='bilinear', align_corners=True)
        c2 = self.linears[2](c2)
        c2 = self.conv_modules[2](c2, seg_map)
        c2 = F.interpolate(c2, size=c0.size()[2:], mode='bilinear', align_corners=True)
        c1 = self.linears[1](c1)
        c1 = self.conv_modules[1](c1, seg_map)
        c1 = F.interpolate(c1, size=c0.size()[2:], mode='bilinear', align_corners=True)
        c0 = self.linears[0](c0)
        c0 = self.conv_modules[0](c0, seg_map)
        output = self.fuse(torch.cat((c0, c1, c2, c3), dim=1))
        return output


class CrossFashion(nn.Module):
    channels = 512
    def __init__(self, emb_dims, num_heads, num_classes=14, img_size=224, mask_dim=256, pool_scales=[1,2,3,6]):
        super(CrossFashion, self).__init__()
        self.attention1_2 = CrossAttention(emb_dims[0], emb_dims[1], num_heads[0])
        self.attention2_3 = CrossAttention(emb_dims[1], emb_dims[2], num_heads[1])
        self.attention3_4 = CrossAttention(emb_dims[2], emb_dims[3], num_heads[2])
        self.ppm = PPM(pool_scales, emb_dims[-1], self.channels)
        self.out_mask = FashionMask(num_classes, emb_dims, dim=mask_dim, out_size=to_2tuple(img_size//4))
        in_dim = emb_dims[-1] + len(pool_scales) * self.channels
        self.bottleneck = ConvModule(in_dim, self.channels, 3, padding=1)
        self.img_size = img_size

        self.norms = nn.ModuleList([
            nn.LayerNorm(emb_dims[0]),
            nn.LayerNorm(emb_dims[1]),
            nn.LayerNorm(emb_dims[2]),
        ])

    def psp_forward(self, x):
        """Forward function of PSP module."""
        x = rearrange(x, 'B (H W) C -> B C H W', H=self.img_size//32, W=self.img_size//32)
        psp_outs = [x]
        psp_outs.extend(self.ppm(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output


    def forward(self, inputs, seg_map):
        c1, c2, c3, c4 = inputs
        x1 = self.attention1_2(c1, c2, H=self.img_size//4, W=self.img_size//4)
        x1 = self.norms[0](x1)
        x1 = rearrange(x1, 'B (H W) C -> B C H W', H=self.img_size//4, W=self.img_size//4)

        x2 = self.attention2_3(c2, c3, H=self.img_size//8, W=self.img_size//8)
        x2 = self.norms[1](x2)
        x2 = rearrange(x2, 'B (H W) C -> B C H W', H=self.img_size // 8, W=self.img_size // 8)

        x3 = self.attention3_4(c3, c4, H=self.img_size//16, W=self.img_size//16)
        x3 = self.norms[2](x3)
        x3 = rearrange(x3, 'B (H W) C -> B C H W', H=self.img_size // 16, W=self.img_size // 16)

        x4 = self.psp_forward(c4)
        out = self.out_mask([x1, x2, x3, x4], seg_map)
        return out


if __name__ == '__main__':
    c1 = torch.randn(4, 56*56, 64)
    c2 = torch.randn(4, 28*28, 128)
    c3 = torch.randn(4, 14*14, 256)
    c4 = torch.randn(4, 7*7, 512)
    seg_map = torch.randn(4, 14, 56, 56)
    inputs = (c1, c2, c3, c4)
    module = CrossFashion(emb_dims=(64,128,256,512), num_heads=(4,8,16), img_size=224)
    out = module.forward(inputs, seg_map)
    # for items in out:
    #     print(items.shape)
    print(out.shape)
    params = sum(p.numel() for p in module.parameters())
    print(params/1e6)
    # f = torch.zeros(4, 128, 28, 28)
    # convm = LargeKernelConv(in_dim=128, out_dim=128, kernel_size=14, dilation=3)
    # print(convm)
    # out = convm(f)
    # print(out.size())
