import argparse
import warnings
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import dataset as dataset
import utils as utils
from unet_train import UNetTrain
from unet_custom import UNetSync as UNet

warnings.filterwarnings("ignore")

#
# # Main function code start
# args = get_command_line_args()
# no_of_epochs = args.no_epochs
# validation_percentage = args.val_size
#
# image = utils.read_image("../../resources/Task1/Train/images/IDRiD_02.tif")
# print(image.shape)
# path=os.path.join()
# images = utils.read_images_from_folder(args.model_category, args.directory, args.image_format)
# print(len(images))
#
# image = pre.scaleradius(image, 300)
# image = pre.normalization(image)
# print(image.shape)
# image = pre.enhance(image)
# utils.display_image(image)
#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global imports
num_of_classes = 5
batch_size = 2

# Train dataset import
dataset = dataset.IDRiDDataset(utils.get_train_dir())
x, y = dataset[0]
print(np.array(x).shape, np.array(y).shape)

# Train and validation split
train_size = int((1 - 0.2) * len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
print('Train dataset', len(train_dataset))
print('Validation dataset', len(validation_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
for i, batch in enumerate(train_dataloader):
    # x, y = batch
    print(i, len(batch))

validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=False)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


unet_model = UNet(3, num_of_classes).to(device)
unet_model.apply(weights_init)
train = UNetTrain(len(train_dataset), train_dataloader, len(validation_dataset), validation_dataloader)
train.train_net(net=unet_model, device=device, save_cp=True)
