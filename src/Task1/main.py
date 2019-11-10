import preprocessing as pre
from torch.utils.data import DataLoader, random_split

import src.Task1.utils as utils
import argparse
import unet
import torch
import src.Task1.dataset as dataset
import numpy as np

import warnings

from src.Task1.unet_train import UNetTrain

warnings.filterwarnings("ignore")


def get_command_line_args():
    parser = argparse.ArgumentParser(description='Get info about the task',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_category', '-m', default='Train',
                        help="Specify the model category like train / test")
    parser.add_argument('--directory', '-dir', metavar='INPUT',
                        help='Directory name of images', required=True)
    parser.add_argument('--image_format', '-imf', metavar='INPUT',
                        help='Image formatting type', required=True, default="jpg")
    parser.add_argument('--learning_rate', '-lr', type=float,
                        help="Learning rate of the model",
                        default=0.001)
    parser.add_argument('--batch_size', '-b', type=int,
                        help="Batch size of the model",
                        default=8)
    parser.add_argument('--no_epochs', '-noe', type=int,
                        help="No of epochs for the model",
                        default=5)
    parser.add_argument('--val_size', '-vs', type=int,
                        help="Validation Size",
                        default=0.2)

    return parser.parse_args()


#
# # Main function code start
# args = get_command_line_args()
# no_of_epochs = args.no_epochs
# validation_percentage = args.val_size
#
# image = utils.read_image("../../resources/Task1/Train/original_retinal_images/IDRiD_02.jpg", 1)
# print(image.shape)
#
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
#
# torch.manual_seed(42)

# Train dataset import
dataset = dataset.IDRiDDataset(utils.get_train_dir())
print(len(dataset))
print(np.array(dataset).shape)

# Train and validation split
train_size = int((1 - 0.2) * len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
print(len(train_dataset))
print(len(validation_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=10,
                              shuffle=True)
for i, batch in enumerate(train_dataloader):
    print(i, len(batch[0][0]))

validation_dataloader = DataLoader(validation_dataset, batch_size=10,
                                   shuffle=False)

unet_model = unet.UNet(3, 4)
train = UNetTrain(len(train_dataset), train_dataloader, len(validation_dataset), validation_dataloader)
train.train_net(net=unet_model, device=device, save_cp=True)
