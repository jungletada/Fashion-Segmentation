import torch
import torch.nn as nn


class ConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvModule, self).__init__()
        self.unit = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.unit(x)


class FCNHead(nn.Module):
    """Fully Convolution Networks for Semantic Segmentation.
    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.
    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self, in_channels, channels, num_convs=1, num_classes=14,
                 kernel_size=3, dilation=1, concat_input=False):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.channels = channels

        super(FCNHead, self).__init__()
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = [ConvModule(
            self.in_channels,
            self.channels,
            kernel_size=kernel_size,
            padding=conv_padding,
            dilation=dilation)]

        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,))

        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2)
        self.cls_seg = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(self.channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, inputs):
        """Forward function."""
        output = self.convs(inputs)

        if self.concat_input:
            output = self.conv_cat(torch.cat([inputs, output], dim=1))

        output = self.cls_seg(output)
        return output


if __name__ == '__main__':
    head = FCNHead(in_channels=1024, channels=256, num_classes=14)
    in_fc = torch.ones(2,1024,56,56)
    out_fc = head(in_fc)
    print(out_fc.shape)