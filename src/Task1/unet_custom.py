import torch
from torch import nn
import torch.nn.functional as F

"""
The dimensions of the H, W stays the same and it is done by using padding to the image 
"""


# Padding is done using the form (kernel_size - 1) // 2
class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels, eps=1e-4)

        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,
                               bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.batch_norm1(self.conv1(x)))
        x = self.relu2(self.batch_norm2(self.conv2(x)))
        return x

def up_conv(in_channels, out_channels):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)


def out_conv(in_channels, out_channels):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)


class UNetSync(nn.Module):
    """
    UNet Customized Architecture
    """

    def __init__(self, in_channels, num_of_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_of_classes = num_of_classes

        # Given an image size of the dim [512 * 512]
        self.down_encode_conv_layer1 = DoubleConv2D(self.in_channels, 64)  # Out Image size [512] 2d
        self.down_encode_conv_layer2 = DoubleConv2D(64, 128)  # Out Image size [256] 2d
        self.down_encode_conv_layer3 = DoubleConv2D(128, 256)  # Out Image size [128] 2d
        self.down_encode_conv_layer4 = DoubleConv2D(256, 512)  # Out Image size [64] 2d
        self.down_encode_conv_layer5 = DoubleConv2D(512, 1024)  # Out Image size [32] 2d

        # self.unet_center = DoubleConv2D(1024, 1024)

        self.up_conv_layer1 = up_conv(1024, 512)
        self.up_decode_conv_layer1 = DoubleConv2D(1024, 512)  # Out Image size [64] 2d
        self.up_conv_layer2 = up_conv(512, 256)
        self.up_decode_conv_layer2 = DoubleConv2D(512, 256)  # Out Image size [128] 2d
        self.up_conv_layer3 = up_conv(256, 128)
        self.up_decode_conv_layer3 = DoubleConv2D(256, 128)  # Out Image size [256] 2d
        self.up_conv_layer4 = up_conv(128, 64)
        self.up_decode_conv_layer4 = DoubleConv2D(128, 64)  # Out Image size [512] 2d

        self.final_layer = out_conv(64, self.num_of_classes)

    def forward(self, x):
        # Down Layer 1
        layer1 = self.down_encode_conv_layer1(x)
        x = F.max_pool2d(layer1, 2)

        # Down layer 2
        layer2 = self.down_encode_conv_layer2(x)
        x = F.max_pool2d(layer2, 2)

        # Down layer 3
        layer3 = self.down_encode_conv_layer3(x)
        x = F.max_pool2d(layer3, 2)

        # Down layer 4
        layer4 = self.down_encode_conv_layer4(x)
        x = F.max_pool2d(layer4, 2)

        # Down layer 5 (customized center for the UNet)
        x = self.down_encode_conv_layer5(x)
        x = F.dropout(x, 0.4)
        # Up layer 1
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        # This will get the dimensions equal to the skip connection feature map dimensions
        x = F.relu(self.up_conv_layer1(x))
        x = torch.cat([x, layer4], dim=1)
        x = F.relu(self.up_decode_conv_layer1(x))

        # Up layer 2
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.up_conv_layer2(x))
        x = torch.cat([x, layer3], dim=1)
        x = F.relu(self.up_decode_conv_layer2(x))

        # Up layer 3
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.up_conv_layer3(x))
        x = torch.cat([x, layer2], dim=1)
        x = F.relu(self.up_decode_conv_layer3(x))

        # Up layer 4
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.up_conv_layer4(x))
        x = torch.cat([x, layer1], dim=1)
        x = F.relu(self.up_decode_conv_layer4(x))

        # Final layer
        x = self.final_layer(x)

        return x # Remove the channel dimension as we only have a single channel on the output
