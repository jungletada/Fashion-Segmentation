import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, reduce, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from models.decode_head.upernet import UPerHead, ConvModule
from models.backbone.convnext import Block
import math


def SR_Module(in_dim, out_dim, sr_ratio):
    """SR_Module with small overlap"""
    if sr_ratio > 2:
        ks1 = sr_ratio + 1
        ks2 = sr_ratio // 2 + 1
        return {'sr1':nn.Conv2d(in_dim,out_dim,kernel_size=ks1, stride=sr_ratio, padding=1),
                'norm1':nn.LayerNorm(out_dim),
                'sr2': nn.Conv2d(in_dim, out_dim, kernel_size=ks2, stride=sr_ratio//2, padding=1),
                'norm2': nn.LayerNorm(out_dim),
                }
    elif sr_ratio == 2:
        return {'sr1': nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, padding=0, bias=False),
                'norm1': nn.LayerNorm(out_dim),
                'sr2': nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
                'norm2': nn.LayerNorm(out_dim),
                }

    """Ablation Study"""
    # if sr_ratio > 2:
    #     ks1 = sr_ratio
    #     ks2 = sr_ratio // 2
    #     return {'sr1':nn.Conv2d(in_dim, out_dim, kernel_size=ks1, stride=sr_ratio, padding=0),
    #             'norm1':nn.LayerNorm(out_dim),
    #             'sr2': nn.Conv2d(in_dim, out_dim, kernel_size=ks2, stride=sr_ratio//2, padding=0),
    #             'norm2': nn.LayerNorm(out_dim),
    #             }
    # elif sr_ratio == 2:
    #     return {'sr1': nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, padding=0, bias=False),
    #             'norm1': nn.LayerNorm(out_dim),
    #             'sr2': nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
    #             'norm2': nn.LayerNorm(out_dim),
    #             }


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = rearrange(x,'B (H W) C->B C H W', H=H, W=W)
        x = self.dwconv(x)
        x = rearrange(x,'B C H W->B (H W) C', H=H, W=W)
        return x


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
            sr_module = SR_Module(dim, dim,sr_ratio=sr_ratio)
            self.sr1 = sr_module['sr1']
            self.norm1 = sr_module['norm1']
            self.sr2 = sr_module['sr2']
            self.norm2 = sr_module['norm2']
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
        q = rearrange(self.q(x), 'B N (d c) -> B d N c', c=C//self.num_heads, d=self.num_heads)
        if self.sr_ratio > 1:
            sr1 = self.sr_ratio
            sr2 = self.sr_ratio // 2
            x_ = rearrange(x, 'B (H W) C -> B C H W', H=H, W=W)

            x_1 = rearrange(self.sr1(x_), 'B C h w -> B (h w) C', h=H//sr1, w=W//sr1)
            x_1 = self.act(self.norm1(x_1))

            x_2 = rearrange(self.sr2(x_), 'B C h w -> B (h w) C', h=H//sr2, w=W//sr2)
            x_2 = self.act(self.norm2(x_2))

            kv1 = rearrange(self.kv1(x_1), 'B n (a d c) -> a B d n c',
                            a=2, d=self.num_heads//2, c=C//self.num_heads)
            kv2 = rearrange(self.kv2(x_2), 'B n (a d c) -> a B d n c',
                            a=2, d=self.num_heads//2, c=C//self.num_heads)

            k1, v1 = kv1[0], kv1[1]     # B d N C
            k2, v2 = kv2[0], kv2[1]     # B d N C
            ################ shunted 1 ################
            attn1 = (q[:, :self.num_heads // 2] @ rearrange(k1,'B d N C-> B d C N')) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)

            local_v1 = rearrange(v1, 'B d (h w) (a c) -> B (a c d) h w',
                                 a=2, d=self.num_heads//2, c=C//self.num_heads//2, h=H//sr1, w=W//sr1)
            local_v1 = rearrange(self.local_conv1(local_v1), 'B (a c d) h w -> B d (h w) (a c)',
                                 a=2, d=self.num_heads//2, c=C//self.num_heads//2, h=H//sr1, w=W//sr1)
            v1 = v1 + local_v1
            x1 = rearrange((attn1 @ v1), 'B d N c->B N (d c)',
                            d=self.num_heads//2,N=H*W,c=C//self.num_heads)
            ################ shunted 2 ################
            attn2 = (q[:, self.num_heads // 2:] @ rearrange(k2,'B d N C-> B d C N')) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            local_v2 = rearrange(v2, 'B d (h w) (a c) -> B (a c d) h w',
                                 a=2, d=self.num_heads//2, c=C//self.num_heads//2, h=H//sr2, w=W//sr2)
            local_v2 = rearrange(self.local_conv2(local_v2), 'B (a c d) h w -> B d (h w) (a c)',
                                a=2, d=self.num_heads//2, c=C//self.num_heads//2, h=H//sr2, w=W//sr2)
            v2 = v2 + local_v2
            x2 = rearrange((attn2 @ v2), 'B d N c->B N (d c)',
                            d=self.num_heads//2, c=C//self.num_heads)
            x = torch.cat([x1, x2], dim=-1)

        else:
            kv = rearrange(self.kv(x),'B N (a c d)-> a B d N c',
                           a=2, c=C//self.num_heads, d=self.num_heads)
            k, v = kv[0], kv[1]
            attn = (q @ rearrange(k,'B d N C-> B d C N')) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = rearrange((attn @ v), 'B d N c->B N (d c)', d=self.num_heads, c=C//self.num_heads)
            local_v = rearrange(v, 'B d (h w) c -> B (c d) h w',
                                d=self.num_heads, c=C//self.num_heads, h=H, w=W)
            local_v = rearrange(self.local_conv(local_v), 'B C h w -> B (h w) C')
            x = x + local_v

        x = self.proj(x)
        x = self.proj_drop(x)
        # q = rearrange(q, 'B d N c->B N (d c)', c=C // self.num_heads, d=self.num_heads)
        # x = q + x
        return x


# Transformer Block
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        # self.proj_img = nn.Conv2d(3, dim,kernel_size=3,stride=1,padding=1,bias=False)
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
    def __init__(self, in_size=56, patch_size=3, stride=2, in_chans=64, embed_dim=128):
        super().__init__()
        in_size = to_2tuple(in_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = in_size
        self.patch_size = patch_size
        self.H, self.W = in_size[0] // patch_size[0], in_size[1] // patch_size[1]
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

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
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


class ResnetPatchEmbed(nn.Module):
    def __init__(self, channel=64):
        super(ResnetPatchEmbed, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channel // 2, kernel_size=7, stride=2, padding=3, bias=False),
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

    def forward(self, x):
        x = self.stem(x)
        x = self.conv(x)
        h, w = x.size()[2:]
        x = rearrange(x, 'B C H W->B (H W) C', H=h, W=w)
        x = self.norm(x)
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


class SemanticBlock(nn.Module):
    def __init__(self, dim, num_cls, drop_path=0.):
        super(SemanticBlock, self).__init__()
        self.skip_connect = nn.Linear(dim, num_cls)
        self.norm1 = nn.LayerNorm(dim)
        self.skip_norm = nn.LayerNorm(num_cls)
        self.q = nn.Linear(dim, num_cls)
        self.k = nn.Linear(dim, num_cls)
        self.v = nn.Linear(dim, num_cls)
        self.act = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)
        
        self.cls = nn.Sequential(
            ConvModule(in_dim=num_cls,out_dim=128,kernal_size=3,stride=1,padding=1),
            nn.Conv2d(128, num_cls,kernel_size=1, stride=1)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(num_cls)
        self.conv_gamma = nn.Conv2d(num_cls, dim, kernel_size=3, stride=1, padding=1)
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x, H, W):
        identity = rearrange(x, 'B (H W) C->B C H W', H=H, W=W)
        x = self.norm1(x)
        skip1 = self.skip_norm(self.skip_connect(x))

        q = self.q(x)
        k = rearrange(self.k(x),'B N K->B K N')
        v = self.v(x)

        attention = self.softmax(q @ k)
        attention = self.drop_path(attention @ v) + skip1
        semantic_map = rearrange(attention, 'B (H W) K->B K H W', H=H, W=W)
        semantic_map = self.cls(semantic_map)

        seg_feat = self.conv_gamma(semantic_map)
        out = identity + seg_feat * self.gamma
        out = rearrange(out, 'B C H W->B (H W) C', H=H, W=W)
        return out, semantic_map


class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_k, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim = min(dim_k, dim_q)
        assert self.dim % num_heads == 0

        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_k = nn.LayerNorm(dim_k)
        self.q = nn.Linear(dim_q, self.dim)
        self.kv = nn.Linear(dim_k, 2 * self.dim)
        self.scale = (self.dim // num_heads) ** -0.5
        self.local_conv = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, stride=1, groups=self.dim)


    def forward(self, query, key, H, W):
        N1 = query.shape[1] # B x N1 x C1
        N2 = key.shape[1]   # B x N2 x C2
        query = self.q(self.norm_q(query)) # project query
        # skip connection of query
        skip_q = rearrange(query, 'B (H W) C -> B C H W', H=H, W=W)
        skip_q = rearrange(self.local_conv(skip_q),
                           'B C H W -> B (H W) C', H=H, W=W)
        # reshape query
        query = rearrange(query, 'B N (d c) -> B d N c',
                          N=N1, c=self.dim//self.num_heads, d=self.num_heads)
        # project key and value
        key_value = rearrange(self.kv(self.norm_k(key)), 'B N (a d c) -> a B d N c',
                            a=2, N=N2, d=self.num_heads, c=self.dim//self.num_heads)

        key = rearrange(key_value[0], 'B d N c->B d c N') # transpose key
        value = key_value[1]                # value
        attn = (query @ key) * self.scale   # attention
        attn = attn.softmax(dim=-1)
        x = rearrange((attn @ value),
                      'B d N c->B N (d c)', d=self.num_heads, N=N1, c=self.dim // self.num_heads)
        x = x + skip_q
        return x


if __name__ == '__main__':
    # query = torch.randn(4, 56*56, 64)
    # key = torch.randn(4, 28*28, 128)
    # module = CrossAttention(dim_q=64, dim_k=128, num_heads=4)
    # outs = module(query, key, H=56, W=56)
    # print(outs.shape)
    # print(module) # Params: 43k
    # channels = 512
    module = AttentionBlock(dim=channels,num_heads=4,mlp_ratio=4,qkv_bias=True)
    # module2 = Block(dim=channels)
    #
    # params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    # print("Params: {:.3f}k".format(params / 1e3))
    #
    # params2 = sum(p.numel() for p in module2.parameters() if p.requires_grad)
    # print("Params: {:.3f}k".format(params2 / 1e3))


