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




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


import matplotlib.pyplot as plt
import torch.nn.functional as F
import dataset as dataset
from metrics import eval_metrics
# Define the helper function
def decode_segmap(image, nc=5):
   
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128)])
 
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
   
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
     
    return np.stack([r, g, b], axis=2)


dataset = dataset.IDRiDDataset(utils.get_train_dir())

# Train and validation split
train_size = int((1 - 0.2) * len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])


validation_dataloader = DataLoader(validation_dataset, batch_size=1,
                                   shuffle=False)
num_of_classes = 5

test_loss = 0
correct = 0
masks_pic = []
test_history_per_epoch = {'acc': [], 'ave_acc_per_class': [], 'jaccard_score': [],
                                       'dice_score': []}

model.eval()
with torch.no_grad():
    for k, val_batch in enumerate(validation_dataloader):
        validation_input, validation_label = val_batch
        validation_images = validation_input.type(torch.FloatTensor).to(device).permute(0, 3, 1, 2)
        validation_targets = validation_label.type(torch.LongTensor).to(device)
        output = model(validation_images)
        test_overall_acc, test_avg_per_class_acc, test_avg_jacc, test_avg_dice = eval_metrics(
                        validation_targets.squeeze(1),
                        output.argmax(dim=1),
                        num_of_classes)
        test_history_per_epoch['acc'].append(test_overall_acc)
        test_history_per_epoch['ave_acc_per_class'].append(test_avg_per_class_acc)
        test_history_per_epoch['jaccard_score'].append(test_avg_jacc)
        test_history_per_epoch['dice_score'].append(test_avg_dice)
        # target has now a 1 in channel c if the pixel location belongs
        # to that class (like your target images)
        om2 = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
        print(np.histogram(om2,bins=range(6)))
        rgb = decode_segmap(om2)
        rgb1 = decode_segmap(validation_targets.squeeze().cpu().numpy())
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(rgb1)
        ax1.set_title('Ground Truth')
        ax1.axis('off')
        ax2.imshow(rgb)
        ax2.set_title('Prediction')
        ax2.axis('off')
        plt.show()
        masks_pic.append(om2)

    print('Test data metrics: Overall Accuracy: {:.4f}, Accuracy per class: {:.4f}, '
                  'Train Jaccard: {:.4f}, Train Dice: {:.4f},'.format(np.mean([x.item() for x in test_history_per_epoch['acc']]),
                                                                      np.mean(
                                                                          [x.item() for x in test_history_per_epoch['ave_acc_per_class']]),
                                                                      np.mean(
                                                                          [x.item() for x in test_history_per_epoch[
                                                                              'jaccard_score']]),
                                                                      np.mean(
                                                                          [x.item() for x in test_history_per_epoch[
                                                                              'dice_score']])))