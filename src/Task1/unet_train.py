import torch
import torch.optim as opt
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import jaccard_score as jsc
from metrics import eval_metrics


# torch.set_default_tensor_type('torch.FloatTensor')


class UNetTrain:
    def __init__(self, train_length, train_loader, validation_length, validation_loader, num_of_classes=5, lr=0.001,
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
        num_classes = logits.shape[1]
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)

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

        train_history = {}
        validation_history = {}
        # Train the model
        total_step = len(self.train_loader)
        for epoch in tqdm(range(self.epochs)):
            print('Starting epoch {}/{}.'.format(epoch + 1, self.epochs))

            train_history_per_epoch = {'loss': [], 'acc': [], 'ave_acc_per_class': [], 'jaccard_score': [],
                                       'dice_score': []}
            validation_history_per_epoch = {'loss': [], 'acc': [], 'ave_acc_per_class': [], 'jaccard_score': [],
                                            'dice_score': []}
            net.train()

            for i, batch in enumerate(self.train_loader):
                start = time.time()
                input, label = batch
                images = input.type(torch.FloatTensor).to(device).permute(0, 3, 1, 2)
                labels = label.type(torch.LongTensor).unsqueeze(1).to(device)
                del batch

                outputs = net(images)
                loss = criterion(labels, outputs)
                overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(labels.squeeze(1),
                                                                                  outputs.argmax(dim=1),
                                                                                  self.num_of_classes)
                train_history_per_epoch['acc'].append(overall_acc)
                train_history_per_epoch['ave_acc_per_class'].append(avg_per_class_acc)
                train_history_per_epoch['jaccard_score'].append(avg_jacc)
                train_history_per_epoch['dice_score'].append(avg_dice)
                train_history_per_epoch['loss'].append(loss.item())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                end = time.time()
                # print('Time taken for the batch is {}'.format(end - start))

                if (i + 1) % 2 == 0:
                    print('Loss, Jaccard, Dice at Epoch [{}/{}] and Step [{}/{}] is: {:.4f}, {:.4f}, {:.4f}'
                          .format(epoch + 1, self.epochs, i + 1, total_step, np.mean(train_history_per_epoch['loss']),
                                  np.mean(train_history_per_epoch['jaccard_score']),
                                  np.mean(train_history_per_epoch['dice_score'])))

            if save_cp:
                torch.save(net.state_dict(), model_checkpoints + 'CP{}.pth'.format(epoch + 1))
                print('Checkpoint {} saved !'.format(epoch + 1))

                print('\nEval..')
                # eval
                net.eval()
                for k, val_batch in enumerate(self.validation_loader):
                    val_start = time.time()
                    validation_input, validation_label = val_batch
                    validation_images = validation_input.type(torch.FloatTensor).to(device).permute(0, 3, 1, 2)
                    validation_targets = validation_label.type(torch.LongTensor).unsqueeze(1).to(device)
                    # Predict
                    masks_pred = net(validation_images)

                    # Calculate loss
                    val_loss = criterion(validation_targets, masks_pred)
                    val_overall_acc, val_avg_per_class_acc, val_avg_jacc, val_avg_dice = eval_metrics(
                        validation_targets.squeeze(1),
                        masks_pred.argmax(dim=1),
                        self.num_of_classes)
                    validation_history_per_epoch['acc'].append(val_overall_acc)
                    validation_history_per_epoch['ave_acc_per_class'].append(val_avg_per_class_acc)
                    validation_history_per_epoch['jaccard_score'].append(val_avg_jacc)
                    validation_history_per_epoch['dice_score'].append(val_avg_dice)
                    validation_history_per_epoch['loss'].append(val_loss.item())

                    val_end = time.time()
                    # print('Time taken for the batch is {}'.format(val_end - val_start))
                print('Validation Loss, Jaccard, Dice after Epoch [{}/{}] is: {:.4f}, {:.4f}, {:.4f}'
                      .format(epoch + 1, self.epochs, np.mean(validation_history_per_epoch['loss']),
                              np.mean(validation_history_per_epoch['jaccard_score']),
                              np.mean(validation_history_per_epoch['dice_score'])))

            net.train()
            train_history[epoch] = train_history_per_epoch
            validation_history[epoch] = validation_history_per_epoch

            print('Train data after epoch: {} is Loss: {:.4f}, Overall Accuracy: {:.4f}, Accuracy per class: {:.4f}, '
                  'Train Jaccard: {:.4f}, Train Dice: {:.4f},'.format(epoch + 1,
                                                                      np.mean(
                                                                          train_history_per_epoch[
                                                                              'loss']),
                                                                      np.mean(train_history_per_epoch['acc']),
                                                                      np.mean(
                                                                          train_history_per_epoch['ave_acc_per_class']),
                                                                      np.mean(
                                                                          train_history_per_epoch[
                                                                              'jaccard_score']),
                                                                      np.mean(
                                                                          train_history_per_epoch[
                                                                              'dice_score'])))

        total_loss = 0
        total_acc = 0
        total_jaccard = 0
        total_dice = 0
        total_acc_per_class = 0
        for k in train_history.keys():
            total_loss += np.mean(train_history[k]['loss'])
            total_acc += np.mean(train_history[k]['acc'])
            total_acc_per_class += np.mean(train_history[k]['ave_acc_per_class'])
            total_jaccard += np.mean(train_history[k]['jaccard_score'])
            total_dice += np.mean(train_history[k]['dice_score'])

        print('Metrics after training are Loss: {:.4f}, Overall Accuracy: {:.4f}, Accuracy per class: {:.4f}, '
              'Train Jaccard: {:.4f}, Train Dice: {:.4f},'.format(total_loss / self.epochs,
                                                                  total_acc / self.epochs,
                                                                  total_acc_per_class / self.epochs,
                                                                  total_jaccard / self.epochs,
                                                                  total_dice / self.epochs
                                                                  ))
