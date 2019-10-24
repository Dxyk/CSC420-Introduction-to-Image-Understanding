import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class UNet(nn.Module):
    """
    The UNet model
    """

    def __init__(self, num_channels: int = 1, num_classes: int = 1,
                 num_filters: int = 64):
        """
        Initialize the UNet Model

        :param num_channels: the number of channels
        :param num_classes: the number of classes to predict
        :param num_filters: the number of filters
        """
        super(UNet, self).__init__()
        self.in_conv = DoubleConv(num_channels, num_filters)

        self.down_conv1 = DownConv(num_filters, 2 * num_filters)
        self.down_conv2 = DownConv(2 * num_filters, 4 * num_filters)
        self.down_conv3 = DownConv(4 * num_filters, 8 * num_filters)
        self.down_conv4 = DownConv(8 * num_filters, 8 * num_filters)

        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear",
                                     align_corners=True)

        self.up_conv1 = DoubleConv(16 * num_filters, 4 * num_filters)
        self.up_conv2 = DoubleConv(8 * num_filters, 2 * num_filters)
        self.up_conv3 = DoubleConv(4 * num_filters, num_filters)
        self.up_conv4 = DoubleConv(2 * num_filters, num_filters)

        self.out_conv = nn.Conv2d(num_filters, num_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward method

        :param x: the input tensor
        :return: the output tensor
        """
        x1 = self.in_conv(x)

        x2 = self.down_conv1(x1)
        x3 = self.down_conv2(x2)
        x4 = self.down_conv3(x3)
        x5 = self.down_conv4(x4)

        x = self.up_sample(x5)
        x = cat_tensors(x, x4)
        x = self.up_conv1(x)
        x = self.up_sample(x)
        x = cat_tensors(x, x3)
        x = self.up_conv2(x)
        x = self.up_sample(x)
        x = cat_tensors(x, x2)
        x = self.up_conv3(x)
        x = self.up_sample(x)
        x = cat_tensors(x, x1)
        x = self.up_conv4(x)

        x = self.out_conv(x)

        return torch.sigmoid(x)


# ==================== Helper Classes ====================
class DoubleConv(nn.Module):
    """
    A util convolution layer consisting of two
    Sequential (conv2d, BatchNorm2d, ReLU) layers
    """

    def __init__(self, in_channel: int, out_channel: int):
        """
        Initialize the Double Conv Module

        :param in_channel: the number of input channels
        :param out_channel: the number of output channels
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward method

        :param x: the input
        :return: the convolution result
        """
        return self.conv(x)


class DownConv(nn.Module):
    """
    The downward convolution module.
    Consist of Sequence(MaxPool2d, DoubleConv)
    """

    def __init__(self, in_channel: int, out_channel: int):
        """
        Initialize the Down Conv Module

        :param in_channel: the number of input channels
        :param out_channel: the number of output channels
        """
        super(DownConv, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward method

        :param x: the input
        :return: the convolution result
        """
        return self.max_pool_conv(x)


# ==================== Helper Methods ====================
def cat_tensors(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Concatenates two torch tensors of maybe different shapes

    :param x1: the first torch tensor
    :param x2: the second torch tensor
    :return: the concatenated tensor
    """
    diff_y = x2.size()[2] - x1.size()[2]
    diff_x = x2.size()[3] - x1.size()[3]

    pad_sizes = [diff_x // 2, diff_x - diff_x // 2, diff_y // 2,
                 diff_y - diff_y // 2]

    x1 = F.pad(x1, pad_sizes)

    return torch.cat([x2, x1], dim=1)
