import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MLPDecode(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class decoder_MLP(nn.Module):
    def __init__(self, feature_channels, embedding_dim=768, num_classes=14):
        super(decoder_MLP, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = feature_channels

        self.linear_c4 = MLPDecode(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLPDecode(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLPDecode(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLPDecode(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.conv_proj = nn.Sequential(
            nn.Conv2d(embedding_dim*4, embedding_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
        )

        self.convT = nn.Sequential(
            nn.ConvTranspose2d(in_channels=embedding_dim, out_channels=embedding_dim//4,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(embedding_dim//4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=embedding_dim//4, out_channels=embedding_dim // 8,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(embedding_dim//8),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.5)
        self.linear_pred = nn.Conv2d(embedding_dim//8, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = rearrange(self.linear_c4(c4), 'B (h w) C -> B C h w', h=h, w=w)
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = rearrange(self.linear_c3(c3), 'B (h w) C -> B C h w',h=2*h,w=2*w)
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = rearrange(self.linear_c2(c2), 'B (h w) C -> B C h w',h=4*h,w=4*w)
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = rearrange(self.linear_c1(c1), 'B (h w) C -> B C h w',h=8*h,w=8*w)

        x = torch.cat([_c4, _c3, _c2, _c1], dim=1)

        x = self.conv_proj(x)
        x = self.convT(x)
        x = self.dropout(x)
        x = self.linear_pred(x)

        return x

if __name__ == '__main__':
    import torch
    B = 4
    c1, c2, c3, c4 = torch.ones(B, 64, 56, 56), torch.ones(B, 128, 28, 28), \
                     torch.ones(B, 256, 14, 14),torch.ones(B, 512, 7, 7)
    decoder = decoder_MLP(feature_channels=[64,128,256,512])
    out = decoder((c1,c2,c3,c4))
    params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print("Total parameters: {:.2f}".format(params / 1e6))
    print(out.shape)