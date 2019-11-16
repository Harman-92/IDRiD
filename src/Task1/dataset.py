import os
from torch.utils.data import Dataset
import utils as utils
import glob
import numpy as np


class IDRiDDataset(Dataset):
    def __init__(self, data_root, image_extension_type='tif', transform=None):
        self.data_root = data_root
        self.input = []
        self.output = []
        self.transform = transform
        self.image_extension_type = image_extension_type
        self.data_samples = self._init_dataset()

    def __len__(self):
        return len(self.data_samples['input'])

    def __getitem__(self, idx):
        return self.data_samples['input'][idx], self.data_samples['label'][idx]

    def _init_dataset(self):
        # Use the original dataset folder
        if 'images' in os.listdir(self.data_root):
            self.input = utils.read_images_from_folder(os.path.join(self.data_root, 'images'),
                                                       self.image_extension_type)

        # Use the target dataset folder
        if 'masks' in os.listdir(self.data_root):
            self.output = utils.read_images_from_folder(os.path.join(self.data_root, 'masks'),
                                                        self.image_extension_type)

        image_input = []
        image_labels = []
        if len(self.input) == len(self.output):
            for i in range(len(self.input)):
                image_input.append(self.input[i])
                image_labels.append(self.output[i])
                # self.data_samples.append((self.input[i], self.output[i]))
        else:
            exit('Wrong storage of the augmented files, please check the image size of input and output')

        sample = {'input': image_input, 'label': image_labels}
        if self.transform:
            self.data_samples = self.transform(sample)

        return sample
