from models.backbone.fashion_modules import AttentionBlock as SABlk
from models.backbone.convnext import Block as ConvNextBlk
from models.decode_head.upernet import UPerHead
from models.decode_head.fashion_head import FashionHead
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial

import math
import torch
import torch.nn as nn
from einops import rearrange


class ResnetPatchEmbed(nn.Module):
    def __init__(self, channel=64):
        super(ResnetPatchEmbed, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channel // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel//2),
            nn.ReLU(),
            nn.Conv2d(channel // 2, channel // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel//2),
            nn.ReLU(),
            nn.Conv2d(channel // 2, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.conv = nn.Conv2d(channel, channel, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(channel, eps=1e-6)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        identity = x
        x = self.stem(x)
        x = self.conv(x)
        h, w = x.size()[2:]
        x = rearrange(x, 'B C H W->B (H W) C', H=h, W=w)
        x = self.norm(x)
        x = self.dropout(x)
        return x, h, w


class PatchEmbed(nn.Module):
    "Vanilla Patch embedding"
    def __init__(self, in_channel, channel, stride=2):
        super(PatchEmbed, self).__init__()
        self.stride = stride
        self.down = nn.Conv2d(in_channel, channel, kernel_size=stride, stride=stride)
        self.norm = nn.LayerNorm(channel,eps=1e-6)

    def forward(self, x):
        H, W = x.size()[2:]
        h, w = H // self.stride, W // self.stride
        x = self.down(x)
        x = rearrange(x, 'B C H W->B (H W) C',H=h, W=w)
        x = self.norm(x)
        return x, h, w


class FashionFormer(nn.Module):
    def __init__(self, num_classes=14, embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=False, qk_scale=None, 
                 drop_rate=0.1,attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[3, 3, 9, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, decode='Upernet'):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.embed_dims = embed_dims

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = ResnetPatchEmbed(embed_dims[i])
            else:
                patch_embed = PatchEmbed(embed_dims[i-1], embed_dims[i], stride=2)

            block = nn.ModuleList([SABlk(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio = mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[i])
                for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        if decode == 'Upernet':
            self.head = UPerHead(in_channels=embed_dims, channels=512, pool_scales=(1, 2, 3, 6))
        elif decode == 'fashion':
            self.head = FashionHead(in_channels=embed_dims, channels=512, pool_scales=(1, 2, 3, 6))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B, _, H, W = x.shape
        features = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            identity = x
            for blk in block:
                x = blk(x, H, W)    # Self-Attention Block
            x = x + identity
            x = norm(x)
            x = rearrange(x, 'B (H W) C->B C H W', H=H, W=W)
            features.append(x)

        return features

    def forward(self, x):
        out = self.forward_features(x)
        out = self.head(out)
        return out


@register_model
def Fashion_t(decode='Upernet'):
    model = FashionFormer(
        embed_dims=[64, 128, 320, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=False, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[1, 2, 5, 1], 
        sr_ratios=[8, 4, 2, 1],
        decode=decode)
    model.default_cfg = _cfg()

    return model


@register_model
def Fashion_s(decode='Upernet'):
    model = FashionFormer(
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        depths=[2, 3, 13, 1],
        sr_ratios=[8, 4, 2, 1],
        decode=decode)
    model.default_cfg = _cfg()

    return model


@register_model
def Fashion_b(decode='Upernet'):
    model = FashionFormer(
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16], 
        mlp_ratios=[8, 8, 4, 4], 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        depths=[2, 4, 24, 1],
        sr_ratios=[8, 4, 2, 1],
        decode=decode,
        )
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    """
        output stride: [4,8,16,32]
        Stages: Shunted Attention Block 
        Stage-wise long skip connection
        patch_embed1.requires_grad = True & Resnet.patch_embed1
        v = v + local_conv for sr_ration=1
    """
    module = Fashion_s(decode='fashion')
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(params/1e6)
    x = torch.ones(4,3,224,224)
    out = module.forward(x)
    print(out.shape)

