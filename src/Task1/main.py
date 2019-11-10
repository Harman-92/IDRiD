import preprocessing as pre
from torch.utils.data import DataLoader

import src.Task1.utils as utils
import argparse
import unet
import torch
import src.Task1.dataset as dataset
import numpy as np

import warnings
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

    return parser.parse_args()


# # Main function code start
# args = get_command_line_args()
# no_of_epochs = args.no_epochs
#
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
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# torch.manual_seed(42)

# Train dataset import
dataset = dataset.IDRiDDataset(utils.get_train_dir())
print(len(dataset))
print(np.array(dataset).shape)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
for i, batch in enumerate(dataloader):
        print(i, len(batch[0]))
# print(type(x_train), type(x_train_tensor), x_train_tensor.type())

# unet_model = unet.UNet(1, 10)

