from models.decode_head.upernet import PPM, ConvModule
from models.backbone.fashion_modules import CrossAttention
from models.decode_head.fashion_head import SpadeConvModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn
import torch


class WaveNorm(nn.Module):
    """
    WaveNorm
    modified on 2022/03/21
    for fashion0321
    """
    def __init__(self, dim, num_cls):
        super(WaveNorm, self).__init__()
        self.dim = dim
        self.th_n = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.th_c = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.amplitude_seg = nn.Conv2d(num_cls, dim, 3, 1, 1)
        self.phase_seg = nn.Conv2d(num_cls, dim, 3, 1, 1)
        self.proj_channel = nn.Conv2d(num_cls, dim, 1, 1)
        self.reweight = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(dim*2, dim*2, 1),
            nn.Dropout2d(0.1),
        )

    def forward(self, x, segmap):
        shortcut = x
        segmap = F.interpolate(segmap,
                               size=x.size()[2:],
                               mode='nearest')
        s = self.amplitude_seg(segmap) + x
        beta = self.phase_seg(segmap) + x
        channel = self.proj_channel(segmap) + x
        c = self.th_c(channel)
        spatial = s * torch.cos(beta)
        n = self.th_n(spatial)
        weight = F.adaptive_avg_pool2d(spatial + channel, output_size=(1, 1))
        weight = self.reweight(weight)
        weight = rearrange(weight, 'B (d C) H W -> d B C H W', d=2, C=self.dim)
        weight = weight.softmax(dim=0)
        x = n * weight[0] + c * weight[1]
        return self.act(self.norm(x)) + shortcut


class WaveMask(nn.Module):
    """
    WaveMask
    modified on 2022/03/23
    for fashion0323
    """
    def __init__(self, dim, hidden_dim=32, num_cls=14):
        super(WaveMask, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv_x = nn.Conv2d(dim, dim, kernel_size=3, stride=1,padding=1)
        self.share = nn.Sequential(
            nn.Conv2d(num_cls, hidden_dim, kernel_size=3, stride=1,padding=1),
            nn.GELU())
        self.beta = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.gamma = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        self.fc_n = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,padding=1)
        self.fc_c = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1)
        # restricts the FC laters only connect tokens within a local window
        self.tfc_n = nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.theta_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU())

        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

        self.reweight = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, 1),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 1),
            nn.Dropout2d(0.1),
        )

    def forward(self, x, segmap):
        shortcut = x
        segmap = F.interpolate(segmap,
                               size=x.size()[2:],
                               mode='nearest')
        s = self.share(segmap)
        theta = self.theta_conv(s)
        n = self.fc_n(s)
        n = torch.cat([n * torch.cos(theta), n * torch.sin(theta)], dim=1)
        n = self.tfc_n(n)
        c = self.fc_c(s)
        a = F.adaptive_avg_pool2d(n + c, output_size=(1, 1))
        a = self.reweight(a)
        a = rearrange(a, 'B (D N) H W -> D B N H W', D=2, N=self.hidden_dim)
        a = a.softmax(dim=0)
        s = s * a[0] + s * a[1]
        gamma = self.gamma(s)
        beta = self.beta(s)
        x = self.conv_x(x)
        x = self.norm(x) * (1 + gamma) + beta
        return self.act(x) + shortcut


class FashionWaveMask(nn.Module):
    def __init__(self, num_classes, in_dims, dim=512, out_size=(56, 56)):
        super(FashionWaveMask, self).__init__()
        self.out_size = out_size
        self.stages = 4
        self.linears = nn.ModuleList(
            [nn.Conv2d(in_dims[0], dim, kernel_size=1),
             nn.Conv2d(in_dims[1], dim, kernel_size=1),
             nn.Conv2d(in_dims[2], dim, kernel_size=1),
             nn.Conv2d(in_dims[3], dim, kernel_size=1)]
        )
        """ 
        0321: Wavenorm
        0323: WaveMask
        """
        self.fuse = ConvModule(dim * 4, dim, kernal_size=1)
        self.wavemask = WaveMask(dim=dim, hidden_dim=dim, num_cls=num_classes)

    def forward(self, inputs, segmap):
        c0, c1, c2, c3 = inputs

        c3 = self.linears[3](c3)
        # c3 = self.conv_modules[3](c3, seg_map)
        c3 = F.interpolate(c3, size=c0.size()[2:], mode='bilinear', align_corners=True)

        c2 = self.linears[2](c2)
        # c2 = self.conv_modules[2](c2, seg_map)
        c2 = F.interpolate(c2, size=c0.size()[2:], mode='bilinear', align_corners=True)

        c1 = self.linears[1](c1)
        # c1 = self.conv_modules[1](c1, seg_map)
        c1 = F.interpolate(c1, size=c0.size()[2:], mode='bilinear', align_corners=True)

        c0 = self.linears[0](c0)
        #  c0 = self.conv_modules[0](c0, seg_map)

        output = self.fuse(torch.cat((c0, c1, c2, c3), dim=1))
        output = self.wavemask(output, segmap)

        return output


class CrossWaveFashion(nn.Module):
    channels = 512
    def __init__(self, emb_dims, num_heads, num_classes=14, img_size=224, mask_dim=256, pool_scales=[1, 2, 3, 6]):
        super(CrossWaveFashion, self).__init__()
        self.attention1_2 = CrossAttention(emb_dims[0], emb_dims[1], num_heads[0])
        self.attention2_3 = CrossAttention(emb_dims[1], emb_dims[2], num_heads[1])
        self.attention3_4 = CrossAttention(emb_dims[2], emb_dims[3], num_heads[2])
        self.ppm = PPM(pool_scales, emb_dims[-1], self.channels)
        self.out_mask = FashionWaveMask(num_classes, emb_dims, dim=mask_dim, out_size=to_2tuple(img_size // 4))
        in_dim = emb_dims[-1] + len(pool_scales) * self.channels
        self.bottleneck = ConvModule(in_dim, self.channels, 3, padding=1)
        self.img_size = img_size
        self.norms = nn.ModuleList([
            nn.LayerNorm(emb_dims[0]),
            nn.LayerNorm(emb_dims[1]),
            nn.LayerNorm(emb_dims[2]),
        ])
        self.act = nn.GELU()

    def psp_forward(self, x):
        """Forward function of PSP module."""
        x = rearrange(x, 'B (H W) C -> B C H W', H=self.img_size // 32, W=self.img_size // 32)
        psp_outs = [x]
        psp_outs.extend(self.ppm(x))
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
    module = CrossWaveFashion(emb_dims=(64,128,256,512), num_heads=(4,8,16), mask_dim=512, img_size=224)
    # out = module.forward(inputs, seg_map)
    # for items in out:
    #     print(items.shape)
    # module = WaveMask(dim=384, num_cls=14)
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print("Params: {:.3f}M".format(params / 1e6))
