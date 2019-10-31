import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, kernel, num_filters, num_colours, num_in_channels):
        super(CNN, self).__init__()

        padding = kernel // 2

        self.down_conv1 = nn.Sequential(
            nn.Conv2d(num_in_channels, num_filters, kernel_size=kernel,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=kernel,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.rf_conv = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=kernel,
                      padding=padding),
            nn.ReLU()
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters, kernel_size=kernel,
                      padding=padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.up_conv2 = nn.Sequential(
            nn.Conv2d(num_filters, 3, kernel_size=kernel, padding=padding),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.final_conv = nn.Conv2d(3, num_colours, kernel_size=kernel,
                                    padding=padding)

    def forward(self, x):
        out1 = self.down_conv1(x)
        out2 = self.down_conv2(out1)
        out3 = self.rf_conv(out2)
        out4 = self.up_conv1(out3)
        out5 = self.up_conv2(out4)
        out_final = self.final_conv(out5)
        return out_final
