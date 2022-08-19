import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.decode_head import Pyramid_decoder, Conv_decoder, MLP_decoder, upernet
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def _init_weights(m):
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


class DWConv(nn.Module):
    """
    input x -> B N C
    x = depthwise Conv(x) -> B N C
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SpaPosEmb(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(SpaPosEmb, self).__init__()
        self.DWConv = nn.Conv2d(in_nc,in_nc,3,1,1,bias=False,groups=in_nc)
        self.bn1 = nn.BatchNorm2d(in_nc)
        self.act = nn.ReLU()
        self.PWConv = nn.Conv2d(in_nc,out_nc,1,1,0,bias=False)
        self.bn2 = nn.BatchNorm2d(out_nc)

    def forward(self, x, H, W):
        x = F.interpolate(x, size=(H, W), mode='nearest')
        residual = x
        x = self.DWConv(x)
        x = self.bn1(x)
        x = self.act(x)
        x += residual
        x = self.PWConv(x)
        x = self.bn2(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        self.apply(_init_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AttentionBlk(nn.Module):
    def __init__(self, dim, out_dim, isPoolq=False, num_heads=3, qkv_bias=False, kv_ratio=4, pool_op='conv',
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        self.scale =  (dim // num_heads)** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        # self.spa_emb = SpaPosEmb(in_nc=3, out_nc=dim)

        k, s = kv_ratio + 1, kv_ratio
        self.kr_ratio = kv_ratio

        if isPoolq: # Use Query pooling
            if pool_op == 'conv':
                self.poolq = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)

            elif pool_op == 'pool':
                self.poolq = nn.MaxPool2d(kernel_size=(2, 2))

            self.q_ratio = 2
            self.kr_ratio = 1
            self.poolk = nn.Identity()# Key-Value pooling
            self.poolv = nn.Identity()# Key-Value pooling

        else: # NO Query pooling
            self.poolq = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.q_ratio = 1
            if pool_op == 'conv': # Key-Value pooling
                if s >= 2:
                    self.poolk = nn.Conv2d(dim, dim, kernel_size=k, stride=s, padding=1)
                    self.poolv = nn.Conv2d(dim, dim, kernel_size=k, stride=s, padding=1)
                elif s == 1:
                    self.poolk = nn.Conv2d(dim, dim, kernel_size=1)
                    self.poolv = nn.Conv2d(dim, dim, kernel_size=1)

            elif pool_op == 'pool':
                self.poolk = nn.MaxPool2d(kernel_size=(s, s))
                self.poolv = nn.MaxPool2d(kernel_size=(s, s))

        self.attn_drop = nn.Dropout(attn_drop)
        self.linear_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(proj_drop) if proj_drop > 0. else nn.Identity()

        self.mlp = Mlp(in_features=dim, hidden_features=dim*4,
                       out_features=out_dim, drop=proj_drop)

        if dim != out_dim:
            self.res_linear = nn.Linear(dim, out_dim)
        else:
            self.res_linear = nn.Identity()

        self.apply(_init_weights)

    def forward(self, x, H, W):
        """
        :param x: input features
        :param H: input features height
        :param W: input features width
        :param P: Image for Spatial positional embedding
        :return: out fearures, X, H, W
        """
        # LayerNorm
        B, N, C = x.shape
        assert H * W == N, "input height and width error"
        residual = self.poolq(rearrange(x, 'B (H W) C -> B C H W', H=H, W=W))
        x = self.norm1(x)
        # q,k,v by linear
        qkv = rearrange(self.qkv(x), 'B N (n C) -> n B N C', n=3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q,k,v pooling
        q = self.poolq(rearrange(q, 'B (H W) C -> B C H W', H=H, W=W))
        k = self.poolk(rearrange(k, 'B (H W) C -> B C H W', H=H, W=W))
        v = self.poolv(rearrange(v, 'B (H W) C -> B C H W', H=H, W=W))
        # multi-heads
        h1, w1 = H // self.q_ratio, W // self.q_ratio
        # p = self.spa_emb(P, h1, w1)
        # p = rearrange(p, 'B (d c) H W -> B d (H W) c', d=self.num_heads, H=h1, W=w1)
        q = rearrange(q, 'B (d c) H W -> B d (H W) c', d=self.num_heads, H=h1, W=w1)
        # q = p + q
        k = rearrange(k, 'B (d c) H W -> B d c (H W)', d=self.num_heads, H=H//self.kr_ratio, W=W // self.kr_ratio)
        v = rearrange(v, 'B (d c) H W -> B d (H W) c', d=self.num_heads, H=H//self.kr_ratio, W=W // self.kr_ratio)
        # attention
        attn = self.softmax((q @ k) * self.scale)
        attn = self.attn_drop(attn)
        attn = attn @ v
        attn = attn + q
        x = rearrange(attn, 'B d N c -> B N (d c)', d=self.num_heads)
        x = self.linear_proj(x)
        x = self.drop_path(x)
        x = x + rearrange(residual, 'B C h1 w1 ->B (h1 w1) C', h1=h1,w1=w1)
        r = self.res_linear(x)
        x = r + self.mlp(self.norm2(x))
        return x, h1, w1


class Blocks(nn.Module):
    def __init__(self, dim, out_dim, num_heads, num_blocks, pool=True, qkv_bias=False, drop=0., kv_ratio=4, attn_drop=0.,drop_path=0.):
        super(Blocks, self).__init__()

        if num_blocks == 1:
            self.blocks = nn.ModuleList([AttentionBlk(dim, out_dim, isPoolq=pool, num_heads=num_heads,
                                                      qkv_bias=qkv_bias, kv_ratio=kv_ratio, attn_drop=attn_drop, proj_drop=drop)])

        elif num_blocks == 2:
            self.blocks = nn.ModuleList([
                AttentionBlk(dim, dim, isPoolq=pool, num_heads=num_heads,
                             qkv_bias=qkv_bias, kv_ratio=2, attn_drop=attn_drop, proj_drop=drop),
                AttentionBlk(dim, out_dim, isPoolq=False, num_heads=num_heads,
                             qkv_bias=qkv_bias, kv_ratio=kv_ratio, attn_drop=attn_drop, proj_drop=drop)
            ])

        else: # num_blocks > 2
            self.blocks = nn.ModuleList([
                AttentionBlk(dim, dim, isPoolq=pool, num_heads=num_heads,
                             qkv_bias=qkv_bias, kv_ratio=kv_ratio, attn_drop=attn_drop, proj_drop=drop),

                *[AttentionBlk(dim, dim, isPoolq=False, num_heads=num_heads,
                               qkv_bias=qkv_bias, kv_ratio=kv_ratio, attn_drop=attn_drop, proj_drop=drop)
                  for _ in range(num_blocks - 2)],

                AttentionBlk(dim, out_dim, isPoolq=False, num_heads=num_heads,
                             qkv_bias=qkv_bias, kv_ratio=kv_ratio, attn_drop=attn_drop, proj_drop=drop)
            ])


    def forward(self, x, H, W):

        for blk in self.blocks:
            x, H, W = blk(x, H, W)

        return x, H, W


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=5, stride=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))

        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        self.apply(_init_weights)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'B C H W -> B (H W) C',H=self.H, W=self.W)
        x = self.act(x)
        x = self.norm(x)

        return x


class MViT(nn.Module):
    def __init__(self,
                 img_size=224,
                 emb_dim=[96,192,384,768],
                 num_blocks=[2,2,20,2],
                 heads=[3,6,12,24],
                 kv_ratios=[4,2,2,1],
                 qkv_bias=False,
                 scale = 4,
                 num_cls=14,
                 decode='Pyramid'):
        super(MViT, self).__init__()
        self.H, self.W = img_size, img_size
        self.patch_scale = scale
        self.patch_emb = OverlapPatchEmbed(img_size=img_size,patch_size=5, stride=scale, in_chans=3, embed_dim=emb_dim[0])

        self.blocks1 = Blocks(dim=emb_dim[0], out_dim=emb_dim[0], num_heads=heads[0], num_blocks=num_blocks[0], pool=False, kv_ratio=kv_ratios[0])

        self.blocks2 = Blocks(dim=emb_dim[0], out_dim=emb_dim[1], num_heads=heads[1], num_blocks=num_blocks[1], pool=True, kv_ratio=kv_ratios[1])

        self.blocks3 = Blocks(dim=emb_dim[1], out_dim=emb_dim[2], num_heads=heads[2], num_blocks=num_blocks[2], pool=True, kv_ratio=kv_ratios[2])

        self.blocks4 = Blocks(dim=emb_dim[2], out_dim=emb_dim[3], num_heads=heads[2], num_blocks=num_blocks[3], pool=True, kv_ratio=kv_ratios[3])

        if decode == 'Pyramid':
            self.decode_head = Pyramid_decoder.decoder_pyramid(emb_dim, (384,192,96,64), num_classes=num_cls)
        elif decode == 'Uper':
            self.decode_head = upernet.UPerHead(emb_dim, 256, num_classes=num_cls)

    def forward(self, x):
        _, _, H, W = x.shape
        P = x
        x = self.patch_emb(x)
        H, W = H //self.patch_scale, W // self.patch_scale
        x1, H, W = self.blocks1(x, H, W)
        c1 = rearrange(x1, 'B (H W) C -> B C H W', H=H, W=W)
        x2, H, W = self.blocks2(x1, H, W)
        c2 = rearrange(x2, 'B (H W) C -> B C H W', H=H, W=W)
        x3, H, W = self.blocks3(x2, H, W)
        c3 = rearrange(x3, 'B (H W) C -> B C H W', H=H, W=W)
        x4, H, W = self.blocks4(x3, H, W)
        c4 = rearrange(x4, 'B (H W) C -> B C H W', H=H, W=W)

        out = self.decode_head((c1, c2, c3, c4))
        return out


def MVit_b0(decode):
    return MViT(img_size=224,
                 emb_dim=[96,192,384,768],
                 num_blocks=[1,2,11,2],
                 heads=[3,6,12,24],
                 kv_ratios=[4,2,2,1],
                 qkv_bias=False,
                 scale = 4,
                 num_cls=14,
                decode=decode)

def MVit_b1(decode):
    return MViT(img_size=224,
                 emb_dim=[96,192,384,768],
                 num_blocks=[1,2,18,2],
                 heads=[3,6,12,24],
                 kv_ratios=[4,2,2,1],
                 qkv_bias=False,
                 scale = 4,
                 num_cls=14,
                decode=decode)

def MVit_b2(decode):
    return MViT(img_size=224,
                 emb_dim=[96,192,384,768],
                 num_blocks=[2,2,24,2],
                 heads=[3,6,12,24],
                 kv_ratios=[4,2,2,1],
                 qkv_bias=False,
                 scale = 4,
                 num_cls=14,decode=decode)

def MVit_b3(decode):
    return MViT(img_size=224,
                 emb_dim=[96,192,384,768],
                 num_blocks=[1,2,30,2],
                 heads=[3,6,12,24],
                 kv_ratios=[4,2,2,1],
                 qkv_bias=False,
                 scale = 4,
                 num_cls=14,decode=decode)


if __name__ == '__main__':
    x = torch.zeros(4, 3, 224, 224)
    mvit = MVit_b1(decode='Uper')
    params = sum(p.numel() for p in mvit.parameters())
    print("{:.2f} M".format(params/1e6))
    out = mvit(x)
    print(out.shape)

