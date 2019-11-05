import torch
from torch import nn, optim
import torch.nn.functional as F

"""
The dimensions of the H, W stays the same and it is done by using padding to the image 
"""


def down_encode(in_channels, out_channels):
    return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)]


class UNet(nn.Module):
    """
    UNet Architecture
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.down_encode_conv_layer1 = down_encode(self.in_channels, 64)
        self.down_encode_conv_layer2 = down_encode(64, 128)
        self.down_encode_conv_layer3 = down_encode(128, 256)
        self.down_encode_conv_layer4 = down_encode(256, 512)
        self.down_encode_conv_layer5 = down_encode(512, 1024)

        self.fc1 = nn.Linear(50 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Down Layer 1
        x = F.relu(self.down_encode_conv_layer1[0](x))
        layer1 = F.relu(self.down_encode_conv_layer1[1](x))
        x = F.max_pool2d(layer1, 2)

        # Down layer 2
        x = F.relu(self.down_encode_conv_layer2[0](x))
        layer2 = F.relu(self.down_encode_conv_layer2[1](x))
        x = F.max_pool2d(layer2, 2)

        # Down layer 3
        x = F.relu(self.down_encode_conv_layer3[0](x))
        layer3 = F.relu(self.down_encode_conv_layer3[1](x))
        x = F.max_pool2d(layer3, 2)

        # Down layer 4
        x = F.relu(self.down_encode_conv_layer4[0](x))
        layer4 = F.relu(self.down_encode_conv_layer4[1](x))
        x = F.max_pool2d(layer4, 2)

        # Down layer 5 (Global minimum of U of UNet architecture)
        x = F.relu(self.down_encode_conv_layer5[0](x))
        layer5 = F.relu(self.down_encode_conv_layer5[1](x))

        # Up layer 1
        x = F.upsample(layer5, scale_factor=2, mode='bilinear', align_corners=True)


        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))

        return x
