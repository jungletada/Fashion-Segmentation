import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_nc, 4 * out_nc, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(4 * out_nc),
            nn.ReLU(),
            nn.PixelShuffle(upscale_factor=2),
        )

    def forward(self, x):
        return self.upsample(x)


class Block(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(Block, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, dims, embedding_dim=(320,160,64), num_classes=14):
        super(Decoder, self).__init__()
        self.num_classes = num_classes
        c1_nc, c2_nc, c3_nc, c4_nc = dims
        emb4_nc, emb3_nc, emb2_nc = embedding_dim
        emb1_nc = c1_nc + c1_nc

        self.upsample4 = Upsample(in_nc=c4_nc, out_nc=emb4_nc)
        self.upsample3 = Upsample(in_nc=c3_nc + emb4_nc, out_nc=emb3_nc)
        self.upsample2 = Upsample(in_nc=c2_nc + emb3_nc, out_nc=c1_nc)
        self.upsample1 = Upsample(in_nc=emb1_nc, out_nc=c1_nc)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c1_nc, out_channels=c1_nc, kernel_size=3,padding=1,stride=1,bias=False),
            nn.BatchNorm2d(c1_nc),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(c1_nc, c1_nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c1_nc),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        self.linear_pred = nn.Conv2d(c1_nc, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## decoder on C1-C4 ##############
        n, _, h, w = c1.shape

        _c4 = self.upsample4(c4)
        _c3 = self.upsample3(torch.cat((_c4, c3), dim=1))
        _c2 = self.upsample2(torch.cat((_c3, c2), dim=1))
        _c = self.upsample1(torch.cat((_c2, c1), dim=1))

        _c = F.interpolate(_c, size=(4 * h, 4 * w), mode='bilinear', align_corners=True)

        x = self.conv(_c)

        x = self.linear_pred(x)

        return x


# class Decoder(nn.Module):
#     def __init__(self, dims, num_classes=14):
#         super(Decoder, self).__init__()
#         self.num_classes = num_classes
#         embed_dim = 256
#         self.conv1 = Block(dims[0], embed_dim, kernel_size=3, padding=2, dilation=2)
#         self.conv2 = Block(dims[1], embed_dim, kernel_size=3, padding=2, dilation=2)
#         self.conv3 = Block(dims[2], embed_dim, kernel_size=1, padding=0, dilation=1)
#         self.conv4 = Block(dims[3], embed_dim, kernel_size=1, padding=0, dilation=1)
#         self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)),
#                                              nn.Conv2d(dims[0], embed_dim, 1,  1, bias=False),
#                                              nn.BatchNorm2d(embed_dim),
#                                              nn.ReLU())
#         self.fusion = Block(embed_dim * 5, embed_dim, kernel_size=1, padding=0, dilation=1)
#         self.aux = Block(dims[0], embed_dim//8, kernel_size=1, padding=0, dilation=1)
#         self.dropout = nn.Dropout(0.5)
#         self.classifier = nn.Sequential(
#             nn.Conv2d(embed_dim+embed_dim//8, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(embed_dim),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(embed_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Conv2d(embed_dim, num_classes, kernel_size=1, stride=1))
#
#     def forward(self, x):
#         c1, c2, c3, c4 = x
#         h, w = c1.shape[-2:]
#         x1 = self.aux(c1)
#
#         _c1 = self.conv1(c1)
#         _c2 = self.conv2(c2)
#         _c2 = F.interpolate(_c2, (h, w), mode='bilinear', align_corners=True)
#         _c3 = self.conv3(c3)
#         _c3 = F.interpolate(_c3, (h, w), mode='bilinear', align_corners=True)
#         _c4 = self.conv4(c4)
#         _c4 = F.interpolate(_c4, (h, w), mode='bilinear', align_corners=True)
#         _c0 = self.global_avg_pool(c1)
#         _c0 = F.interpolate(_c0, (h, w), mode='bilinear', align_corners=True)
#
#         x = torch.cat((_c1,_c2,_c3,_c4, _c0), dim=1)
#
#         x = self.fusion(x)
#         x = self.dropout(x)
#         x = torch.cat((x,x1),dim=1)
#         x = self.classifier(x)
#         x = F.interpolate(x, (4*h, 4*w), mode='bilinear', align_corners=True)
#
#         return x


if __name__ == '__main__':
    import torch
    B = 2
    c1, c2, c3, c4 = torch.ones(B, 64, 56, 56), torch.ones(B, 128, 28, 28), \
                     torch.ones(B, 256, 14, 14), torch.ones(B, 512, 7, 7)
    decoder = Decoder(dims=[64, 128, 256, 512])
    out = decoder((c1, c2, c3, c4))
    print(out.shape)
