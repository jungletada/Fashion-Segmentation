import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, reduce, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from models.decode_head.upernet import UPerHead
import math

# Inverted Bottleneck
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.act = nn.GELU()
            assert sr_ratio % 2 == 0, f"sr_ratio {sr_ratio} should be (1,2,4,8)."
            self.sr1 = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm1 = nn.LayerNorm(dim)
            self.sr2 = nn.Conv2d(dim, dim, kernel_size=sr_ratio//2, stride=sr_ratio//2)
            self.norm2 = nn.LayerNorm(dim)
            self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.local_conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
            self.local_conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)

        else:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
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

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = rearrange(self.q(x), 'B N (d c) -> B d N c', B=B, N=N, c=C//self.num_heads, d=self.num_heads)
        if self.sr_ratio > 1:
            sr1 = self.sr_ratio
            sr2 = self.sr_ratio // 2
            x_ = rearrange(x, 'B (H W) C -> B C H W', B=B, C=C, H=H, W=W)

            x_1 = rearrange(self.sr1(x_), 'B C h w -> B (h w) C', B=B, C=C, h=H//sr1, w=W//sr1)
            x_1 = self.act(self.norm1(x_1))

            x_2 = rearrange(self.sr2(x_), 'B C h w -> B (h w) C', B=B, C=C, h=H//sr2, w=W//sr2)
            x_2 = self.act(self.norm2(x_2))

            kv1 = rearrange(self.kv1(x_1), 'B n (a d c) -> a B d n c',
                            a=2, B=B, n=H*W//(sr1**2), d=self.num_heads//2, c=C//self.num_heads)
            kv2 = rearrange(self.kv2(x_2), 'B n (a d c) -> a B d n c',
                            a=2, B=B, n=H * W // (sr2 ** 2), d=self.num_heads//2, c=C//self.num_heads)

            k1, v1 = kv1[0], kv1[1]     # B d N C
            k2, v2 = kv2[0], kv2[1]     # B d N C
            ################ shunted 1 ################
            attn1 = (q[:, :self.num_heads // 2] @ rearrange(k1,'B d N C-> B d C N')) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)

            local_v1 = rearrange(v1, 'B d (h w) (a c) -> B (a c d) h w',
                                 a=2, B=B, d=self.num_heads//2, c=C//self.num_heads//2, h=H//sr1, w=W//sr1)
            local_v1 = rearrange(self.local_conv1(local_v1), 'B (a c d) h w -> B d (h w) (a c)',
                                 a=2, B=B, d=self.num_heads//2, c=C//self.num_heads//2, h=H//sr1, w=W//sr1)
            v1 = v1 + local_v1
            x1 = rearrange((attn1 @ v1), 'B d N c->B N (d c)',
                           B=B,d=self.num_heads//2,N=H*W,c=C//self.num_heads)
            ################ shunted 2 ################
            attn2 = (q[:, self.num_heads // 2:] @ rearrange(k2,'B d N C-> B d C N')) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            local_v2 = rearrange(v2, 'B d (h w) (a c) -> B (a c d) h w',
                                 a=2, B=B, d=self.num_heads//2, c=C//self.num_heads//2, h=H//sr2, w=W//sr2)
            local_v2 = rearrange(self.local_conv2(local_v2), 'B (a c d) h w -> B d (h w) (a c)',
                                a=2, B=B, d=self.num_heads//2, c=C//self.num_heads//2, h=H//sr2, w=W//sr2)
            v2 = v2 + local_v2
            x2 = rearrange((attn2 @ v2), 'B d N c->B N (d c)',
                           B=B,d=self.num_heads//2,N=H*W,c=C//self.num_heads)
            x = torch.cat([x1, x2], dim=-1)

        else:
            kv = rearrange(self.kv(x),'B N (a c d)-> a B d N c',
                           B=B,N=H*W,a=2, c=C//self.num_heads, d=self.num_heads)
            k, v = kv[0], kv[1]
            attn = (q @ rearrange(k,'B d N C-> B d C N')) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            local_v = rearrange(v, 'B d (h w) c -> B (c d) h w',
                                d=self.num_heads, c=C//self.num_heads, h=H, w=W)
            local_v = rearrange(self.local_conv(local_v), 'B (c d) h w -> B d (h w) c',
                      d=self.num_heads, c=C // self.num_heads, h=H, w=W)
            v = v + local_v
            x = rearrange((attn @ v), 'B d N c->B N (d c)', d=self.num_heads, N=H*W, c=C//self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Transformer Block
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.proj_img = nn.Conv2d(3, dim,kernel_size=3,stride=1,padding=1,bias=False)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

    def forward(self, x, H, W):
        # x: feature from last layer
        identity1 = x
        x = self.attn(self.norm1(x), H, W)
        x = identity1 + self.drop_path(x)
        identity2 = x
        x = self.mlp(self.norm2(x), H, W)
        x = identity2 + self.drop_path(x)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.gamma = nn.Conv2d(3, embed_dim, kernel_size=1, stride=1, bias=False)
        self.beta = nn.Conv2d(3, embed_dim, kernel_size=1, stride=1, bias=False)
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

    def forward(self, x, img):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x * (1 + self.gamma(img)) + self.beta(img)
        x = rearrange(x, 'B C H W->B (H W) C')
        x = self.norm(x)
        return x, H, W


class Head(nn.Module):
    def __init__(self, num):
        super(Head, self).__init__()
        stem = [nn.Conv2d(3, 64, 7, 2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True)]
        for i in range(num):
            stem.append(nn.Conv2d(64, 64, 3, 1, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(64))
            stem.append(nn.ReLU(True))
        stem.append(nn.Conv2d(64, 64, kernel_size=2, stride=2))
        self.conv = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(64)

        self.gamma = nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False)
        self.beta = nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False)
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

    def forward(self, x, img):
        x = self.conv(x)
        _, _, H, W = x.shape
        x = x * (1 + self.gamma(img)) + self.beta(img)
        x = rearrange(x, 'B C H W->B (H W) C')
        x = self.norm(x)
        return x, H, W


class FashionFormer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, num_conv=0, decode='Upernet'):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.embed_dims = embed_dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = Head(num_conv)
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size // (2 ** (i + 1)),
                                                patch_size=3,
                                                stride=2,
                                                in_chans=embed_dims[i - 1],
                                                embed_dim=embed_dims[i])

            proj_img = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"proj_img{i + 1}", proj_img)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.head = UPerHead(in_channels=embed_dims, channels=512, pool_scales=(1, 2, 3, 6))

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
        imgs = [
            F.interpolate(x, size=(H // 4, W // 4), mode='nearest'),
            F.interpolate(x, size=(H // 8, W // 8), mode='nearest'),
            F.interpolate(x, size=(H // 16, W // 16), mode='nearest'),
            F.interpolate(x, size=(H // 32, W // 32), mode='nearest'),
        ]
        features = []
        for i in range(self.num_stages):
            # proj_img = getattr(self, f"proj_img{i + 1}")
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            # img = proj_img(imgs[i])
            x, H, W = patch_embed(x, imgs[i])

            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            features.append(x)

        return features

    def forward(self, x):
        out = self.forward_features(x)
        out = self.head(out)
        return out


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def Fashion_t(decode='Upernet', **kwargs):
    model = FashionFormer(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 2, 4, 1], sr_ratios=[8, 4, 2, 1], num_conv=0,
        decode=decode, ** kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def Fashion_s(decode='Upernet', **kwargs):
    model = FashionFormer(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 4, 12, 1], sr_ratios=[8, 4, 2, 1], num_conv=1,
        decode=decode,**kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def Fashion_b(decode='Upernet', **kwargs):
    model = FashionFormer(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 24, 2], sr_ratios=[8, 4, 2, 1], num_conv=2,
        decode=decode,
        **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    """
        output stride: [4,8,16,32] 
        # output stride: [4,8,16,16] may be better
        Add image projection to patch embedding by interpolation
        patch_embed1.requires_grad = False
        v = v + local_conv for sr_ration=1
        small: val/mIoU: 0.5881162745348993
    """
    module = Fashion_s()
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(params/1e6)
    x = torch.ones(4,3,224,224)
    out = module.forward(x)
    print(out.shape)
    # module = Attention(dim=512, num_heads=1,sr_ratio=1)
    # x = torch.ones(4, 7*7, 512)
    # out = module.forward(x, H=7,W=7)
    # print(out.shape)

