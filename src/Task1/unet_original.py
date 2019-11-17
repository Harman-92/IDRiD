import torch
from torch import nn, optim
import torch.nn.functional as F

"""
The dimensions of the H, W stays the same and it is done by using padding to the image 
"""
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def double_conv(in_channels, out_channels):
    return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)]


def up_conv(in_channels, out_channels):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, padding=1)


def out_conv(in_channels, out_channels):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)


class UNet(nn.Module):
    """
    UNet Architecture
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.down_encode_conv_layer1 = double_conv(self.in_channels, 64)
        self.down_encode_conv_layer2 = double_conv(64, 128)
        self.down_encode_conv_layer3 = double_conv(128, 256)
        self.down_encode_conv_layer4 = double_conv(256, 512)
        self.down_encode_conv_layer5 = double_conv(512, 1024)

        self.up_conv_layer1 = up_conv(1024, 512)
        self.up_decode_conv_layer1 = double_conv(1024, 512)
        self.up_conv_layer2 = up_conv(512, 256)
        self.up_decode_conv_layer2 = double_conv(512, 256)
        self.up_conv_layer3 = up_conv(256, 128)
        self.up_decode_conv_layer3 = double_conv(256, 128)
        self.up_conv_layer4 = up_conv(128, 64)
        self.up_decode_conv_layer4 = double_conv(128, 64)

        self.final_layer = out_conv(64, self.num_classes)

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
        x = F.relu(self.down_encode_conv_layer5[1](x))

        # Up layer 1
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        # This will get the dimensions equal to the skip connection feature map dimensions
        x = F.relu(self.up_conv_layer1(x))
        x = torch.cat([x, layer4], dim=1)
        x = F.relu(self.up_decode_conv_layer1[0](x))
        x = F.relu(self.up_decode_conv_layer1[1](x))

        # Up layer 2
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.up_conv_layer2(x))
        x = torch.cat([x, layer3], dim=1)
        x = F.relu(self.up_decode_conv_layer2[0](x))
        x = F.relu(self.up_decode_conv_layer2[1](x))

        # Up layer 3
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.up_conv_layer3(x))
        x = torch.cat([x, layer2], dim=1)
        x = F.relu(self.up_decode_conv_layer3[0](x))
        x = F.relu(self.up_decode_conv_layer3[1](x))

        # Up layer 4
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.up_conv_layer4(x))
        x = torch.cat([x, layer1], dim=1)
        x = F.relu(self.up_decode_conv_layer4[0](x))
        x = F.relu(self.up_decode_conv_layer4[1](x))

        # Final layer
        x = self.final_layer(x)


        return x
