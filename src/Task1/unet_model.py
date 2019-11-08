import src.Task1.unet as unet
import torch
import torch.nn as nn
import torch.optim as opt


unet_model = unet.UNet(1, 10)
learning_rate = 0.001
optimizer = opt.Adam(unet_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


