import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, kernel, num_filters, num_classes, num_in_channels):
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

        self.down_conv3 = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=kernel,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = Flatten()

        self.lin1 = nn.Sequential(
            nn.Linear(160000, 1024),
            nn.ReLU()
        )

        self.lin2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        self.lin3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.out_final = nn.Linear(64, num_classes)

    def forward(self, x):
        out1 = self.down_conv1(x)
        out2 = self.down_conv2(out1)
        out3 = self.flatten(out2)
        out4 = self.lin1(out3)
        out5 = self.lin2(out4)
        out6 = self.lin3(out5)
        out_final = self.out_final(out6)
        return out_final


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
