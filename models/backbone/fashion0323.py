from models.backbone.fashion_modules import OverlapPatchEmbed, ResnetPatchEmbed
from models.backbone.fashion_modules import AttentionBlock, SemanticBlock
from models.decode_head.wavefashion_head import CrossWaveFashion
from models.decode_head.cross_stage_head import CrossStageFashion
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FashionFormer(nn.Module):
    def __init__(self, img_size=224, num_classes=14, embed_dims=[64, 128, 256, 512], mask_dim=360,
                 num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[3, 3, 9, 3], sr_ratios=[8, 4, 2, 1],
                 num_stages=4):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.embed_dims = embed_dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = ResnetPatchEmbed(embed_dims[i])
            else:
                patch_embed = OverlapPatchEmbed(
                    img_size // (2 ** (i + 1)),
                    patch_size=3,
                    stride=2,
                    in_chans=embed_dims[i - 1],
                    embed_dim=embed_dims[i]
                )

            block = nn.ModuleList([AttentionBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[i])
                for j in range(depths[i])])

            seblock = SemanticBlock(
                dim=embed_dims[i],
                num_cls=num_classes,
                drop_path=0
            )
            norm = norm_layer(embed_dims[i])
            cur += depths[i]
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"seblock{i + 1}", seblock)
            setattr(self, f"norm{i + 1}", norm)

        # fashion0323
        self.head = CrossWaveFashion(
            emb_dims=embed_dims,
            num_heads=num_heads[1:],
            img_size=img_size,
            mask_dim=mask_dim,
            pool_scales=(1, 2, 3, 6))
        self.out_cls = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(mask_dim, num_classes, kernel_size=1))

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

    def side_semantic_maps(self, feats):
        segmap = feats[0]
        size = feats[0].size()[2:]
        for i in range(self.num_stages - 1):
            segmap = segmap + F.interpolate(
                feats[i + 1],
                size=size,
                mode='bilinear',
                align_corners=True)
        return segmap

    def forward_features(self, x):
        B, _, H, W = x.shape
        features = []
        maps = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            seblock = getattr(self, f"seblock{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W) # Self-Attention Block
            x, map = seblock(x, H, W)
            x = norm(x)
            features.append(x)
            maps.append(map)
            x = rearrange(x, 'B (H W) C->B C H W', H=H, W=W)

        return features, maps

    def forward(self, x):
        out_size = x.size()[2:]
        feats, maps = self.forward_features(x)
        aux = self.side_semantic_maps(maps)
        aux = F.interpolate(aux, size=out_size, mode='bilinear', align_corners=True)

        output = self.head(feats, aux)
        output = self.out_cls(output)
        output = F.interpolate(output, size=out_size, mode='bilinear', align_corners=True)
        return {'output': output, 'aux': aux}


@register_model
def Fashion_t():
    model = FashionFormer(
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        mask_dim=320,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[1, 3, 6, 1],
        sr_ratios=[8, 4, 2, 1])
    model.default_cfg = _cfg()
    return model


@register_model
def Fashion_s():
    model = FashionFormer(
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mask_dim=360,
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 5, 15, 1],
        sr_ratios=[8, 4, 2, 1])
    model.default_cfg = _cfg()
    return model


@register_model
def Fashion_b():
    model = FashionFormer(
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        mask_dim=512,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 4, 24, 1],
        sr_ratios=[8, 4, 2, 1],)
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    """
        epochs = 120 (160k iterations)
        side output for auxilary
        multi-tasks loss
    """
    module = Fashion_s()
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(params/1e6)
    x = torch.ones(4, 3, 224, 224)
    out_dict = module.forward(x)
    for k, v in out_dict.items():
        print(v.shape)
