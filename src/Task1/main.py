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


def get_command_line_args():
    parser = argparse.ArgumentParser(description='Get info about the task',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_category', '-m', default='Train',
                        help="Specify the model category like train / test")
    parser.add_argument('--directory', '-dir', metavar='INPUT',
                        help='Directory name of images', required=False)
    # parser.add_argument('--image_format', '-imf', metavar='INPUT',
    # help='Image formatting type', required=True, default="jpg")
    parser.add_argument('--learning_rate', '-lr', type=float,
                        help="Learning rate of the model",
                        default=0.001)
    parser.add_argument('--batch_size', '-b', type=int,
                        help="Batch size of the model",
                        default=1)
    parser.add_argument('--no_epochs', '-noe', type=int,
                        help="No of epochs for the model",
                        default=1)
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
#
# torch.manual_seed(42)
# print(utils.get_train_dir())
# Train dataset import
dataset = dataset.IDRiDDataset(utils.get_train_dir())
print(len(dataset))
x, y = dataset[0]
print(np.array(x).shape, np.array(y).shape)

# Train and validation split
train_size = int((1 - 0.2) * len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
print('Train dataset', len(train_dataset))
print('Validation dataset', len(validation_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=2,
                              shuffle=True)
for i, batch in enumerate(train_dataloader):
    # x, y = batch
    print(i, len(batch))

validation_dataloader = DataLoader(validation_dataset, batch_size=1,
                                   shuffle=False)

unet_model = UNet(3).to(device)
train = UNetTrain(len(train_dataset), train_dataloader, len(validation_dataset), validation_dataloader)
train.train_net(net=unet_model, device=device, save_cp=True)
