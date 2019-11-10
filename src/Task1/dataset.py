import os
from torch.utils.data import Dataset
import torch
import src.Task1.utils as utils
import glob


class IDRiDDataset(Dataset):
    def __init__(self, data_root, image_extension_type='jpg'):
        self.data_root = data_root
        self.input = []
        self.output = []
        self.data_samples = []
        self.image_extension_type = image_extension_type
        self._init_dataset()

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        return self.data_samples[idx]

    def _init_dataset(self):
        # Use the original dataset folder
        if 'original_retinal_images' in os.listdir(self.data_root):
            self.input = self.read_images_from_folder(os.path.join(self.data_root, 'original_retinal_images'))

        # Use the target dataset folder
        if 'original_retinal_images' in os.listdir(self.data_root):
            self.output = self.read_images_from_folder(os.path.join(self.data_root, 'original_retinal_images'))

        if len(self.input) == len(self.output):
            for i in range(len(self.input)):
                self.data_samples.append((self.input[i], self.output[i]))
        else:
            exit('Wrong storage of the augmented files, please check the image size of input and output')

    def read_images_from_folder(self, dir_path):
        image_list = []
        image_path = os.path.join(dir_path, '*.' + self.image_extension_type)
        for image_file_name in glob.glob(image_path):
            image = utils.read_image(image_file_name, 1)
            if image is not None:
                image_list.append(image)

        return image_list
