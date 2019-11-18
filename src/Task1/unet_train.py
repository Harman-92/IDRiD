import torch
import torch.optim as opt
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import jaccard_score as jsc


# torch.set_default_tensor_type('torch.FloatTensor')


class UNetTrain:
    def __init__(self, train_length, train_loader, validation_length, validation_loader, num_of_classes=5, lr=0.0001,
                 epochs=5):
        self.train_length = train_length
        self.train_loader = train_loader
        self.num_of_classes = num_of_classes
        self.validation_length = validation_length
        self.validation_loader = validation_loader
        self.lr = lr
        self.epochs = epochs

    def dice_loss(self, true, logits, eps=1e-7):
        """Computes the Sørensen–Dice loss.
        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss_v1: the Sørensen–Dice loss.
        """
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)

    def iou(self, pred, target):
        ious = []
        pred = pred.argmax(dim=1).view(-1)
        target = target.view(-1)

        # Ignore IoU for background class ("0")
        for cls in range(1, self.num_of_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
            pred_inds = pred == cls
            target_inds = target == cls
            # print(pred_inds[target_inds].long().sum().data.cpu().item())
            intersection = (
                (pred_inds[target_inds]).long().sum().data.cpu().item())  # Cast to long to prevent overflows
            union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
            if union == 0:
                ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
            else:
                ious.append(float(intersection) / float(max(union, 1)))
        return np.array(ious)

    def jaccard_sim_score(self, input, target):

        n = target.size(0)
        unet_input = input.argmax(dim=1).view(n, -1).cpu().numpy().reshape(-1)
        # print('Input jacc', unet_input.shape)
        label = target.cpu().numpy().reshape(-1)
        # print('Output jacc', label.shape)

        return jsc(unet_input, label, average='micro')

    def get_epoch_accuracies_per_classes(self, iou_acc):
        return np.mean(iou_acc, axis=0)

    def train_net(self, net,
                  device,
                  save_cp=False):

        model_checkpoints = 'checkpoints/'

        print('''
            Start train:
                Epochs: {}
                            Learning rate: {}
                                            Training size: {}
                                                            Validation size: {}
                                                                                Checkpoints: {}
                                                                                                CUDA: {}
            '''.format(self.epochs, self.lr, self.train_length,
                       self.validation_length, str(save_cp), str(device)))

        optimizer = opt.Adam(net.parameters(), lr=self.lr)
        # weights = [1 / 60, 1.0, 1.0, 1.0, 2.0]
        # class_weights = torch.FloatTensor(weights).to(device)
        criterion = self.dice_loss
        train_history = {'loss': [], 'train_step': [], 'train_jacc_score': []}
        validation_history = {'val_loss_epoch': [], 'val_loss_batch': [], 'val_jacc_epoch': []}

        # Train the model
        total_step = len(self.train_loader)
        for epoch in tqdm(range(self.epochs)):
            print('Starting epoch {}/{}.'.format(epoch + 1, self.epochs))
            net.train()
            epoch_loss = 0
            jaccard_score = []
            for i, batch in enumerate(self.train_loader):
                start = time.time()
                input, label = batch
                images = input.type(torch.FloatTensor).to(device).permute(0, 3, 1, 2)
                labels = label.type(torch.LongTensor).unsqueeze(1).to(device)
                del batch

                outputs = net(images)
                loss = criterion(labels, outputs)
                iou_accuracies = self.iou(outputs, labels)
                jaccard_score.append(iou_accuracies)
                print(iou_accuracies)
                # Backward and optimize
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                end = time.time()
                print('Time taken for the batch is {}'.format(end - start))
                if (i + 1) % 2 == 0:
                    train_history['loss'].append(loss.item())
                    train_history['train_jacc_score'].append(iou_accuracies)
                    train_history['train_step'].append(i + epoch * total_step)

                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Jaccard: {}'
                          .format(epoch + 1, self.epochs, i + 1, total_step, loss.item(),
                                  iou_accuracies))

            if save_cp:
                torch.save(net.state_dict(), model_checkpoints + 'CP{}.pth'.format(epoch + 1))
                print('Checkpoint {} saved !'.format(epoch + 1))

                print('\nEval..')
                # eval
                net.eval()
                val_losses = []
                val_dice_score = []
                val_jacc_score = []
                for k, val_batch in tqdm(enumerate(self.validation_loader)):
                    val_start = time.time()
                    validation_input, validation_label = val_batch
                    validation_images = validation_input.type(torch.FloatTensor).to(device).permute(0, 3, 1, 2)
                    validation_targets = validation_label.type(torch.LongTensor).unsqueeze(1).to(device)
                    # Predict
                    masks_pred = net(validation_images)

                    # Calculate loss
                    val_loss = criterion(validation_targets, masks_pred).item()
                    val_jacc_score.append(self.iou(masks_pred, validation_targets))
                    val_losses.append(val_loss)

                    validation_history['val_loss_batch'].append(val_loss)

                    val_end = time.time()
                    print('Time taken for the batch is {}'.format(val_end - val_start))
                validation_history['val_loss_epoch'].append(np.mean(val_losses))
                validation_history['val_jacc_epoch'].append(self.get_epoch_accuracies_per_classes(val_jacc_score))
                print('Validation Loss: {}, Validation Epoch: {}, Validation Jaccard: {}'
                      .format(np.mean(val_losses), epoch + 1,
                              self.get_epoch_accuracies_per_classes(val_jacc_score)))

            net.train()
            print('Epoch finished ! Loss: {}'.format(epoch_loss / total_step))
            print(train_history)
            print(validation_history)
            print('Epoch: {}, Train Loss: {}, Train Jaccard: {}, Validation Loss: '
                  '{}, Validation Jaccard: {} '.format(epoch + 1,
                                                       epoch_loss / total_step,
                                                       self.get_epoch_accuracies_per_classes(
                                                           jaccard_score),
                                                       validation_history['val_loss_epoch'][epoch],
                                                       validation_history['val_jacc_epoch'][
                                                           epoch]))
