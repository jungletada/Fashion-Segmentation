import math
import torch.nn as nn
from models.utils.sync_batchnorm import SynchronizedBatchNorm2d


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetv1c(nn.Module):
    def __init__(self, block, layers, BatchNorm=nn.BatchNorm2d, strides=[1,2,2,1]):
        inplanes = 64
        self.inplanes = inplanes
        super(ResNetv1c, self).__init__()
        dilations = [1, 1, 1, 1]
        # Modules
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes //2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            BatchNorm(inplanes //2),
            nn.ReLU(),
            nn.Conv2d(inplanes //2, inplanes //2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            BatchNorm(inplanes //2),
            nn.ReLU(),
            nn.Conv2d(inplanes //2, inplanes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            BatchNorm(inplanes),
            nn.ReLU(),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, inplanes, layers[0], stride=strides[0], dilation=dilations[0],
                                       BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=strides[1], dilation=dilations[1],
                                       BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=strides[2], dilation=dilations[2],
                                       BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=strides[3], dilation=dilations[3],
                                       BatchNorm=BatchNorm)

        self._init_weight()

    def _make_layer(self, block, planes, num_blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, dilation, downsample, BatchNorm)]
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, dilation=blocks[0] * dilation,
                        downsample=downsample, BatchNorm=BatchNorm)]
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1,c2,c3,c4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def Resnet101(BatchNorm):
    return ResNetv1c(block=Bottleneck, layers=[3, 4, 23, 3], BatchNorm=BatchNorm)


if __name__ == '__main__':
    import torch
    net = ResNetv1c(block=Bottleneck, layers=[3, 4, 23, 3], strides=[1,2,1,1])
    x = torch.ones(2, 3, 224, 224)
    y = net(x)
    # for item in y:
    #     print(item.shape)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("#params: {:.2f}M".format(total_params / 1e6))