import torch
import torch.optim as opt
import torch.nn as nn
import time
import numpy as np


class UNetTrain:
    def __init__(self, train_length, train_loader, validation_length, validation_loader, lr=0.001,
                 epochs=5):
        self.train_length = train_length
        self.train_loader = train_loader
        self.validation_length = validation_length
        self.validation_loader = validation_loader
        self.lr = lr
        self.epochs = epochs

    def train_net(self, net,
                  device,
                  save_cp=False):

        model_checkpoints = 'checkpoints/'

        print('''
            Starting training:
                Epochs: {}
                Learning rate: {}
                Training size: {}
                Validation size: {}
                Checkpoints: {}
                CUDA: {}
            '''.format(self.epochs, self.lr, self.train_length,
                       self.validation_length, str(save_cp), str(device)))

        learning_rate = self.lr
        optimizer = opt.Adam(net.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_history = {'loss': [], 'train_step': []}
        validation_history = {'val_loss': [], 'val_step': []}

        # Train the model
        total_step = len(self.train_loader)
        for epoch in range(self.epochs):
            print('Starting epoch {}/{}.'.format(epoch + 1, self.epochs))
            net.train()
            epoch_loss = 0
            for i, batch in enumerate(self.train_loader):
                start = time.time()

                images = batch[0].type(torch.FloatTensor).to(device)
                labels = batch[1].type(torch.FloatTensor).to(device)
                images = images.permute(0, 3, 1, 2)
                labels = labels.permute(0, 3, 1, 2)
                del batch

                # Forward pass
                outputs = net(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                end = time.time()
                print('Time taken for the batch is {}'.format(end - start))
                if (i + 1) % 100 == 0:

                    train_history['loss'].append(loss.item())
                    train_history['train_step'].append(i + epoch * len(self.train_loader))

                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, self.epochs, i + 1, total_step, loss.item()))

                    # every 100 steps save model
                    if i % 500 == 0:
                        if save_cp:
                            torch.save(net.state_dict(), model_checkpoints + 'CP{}.pth'.format(epoch + 1))
                            print('Checkpoint {} saved !'.format(epoch + 1))

                        print('\nEval..')
                        # eval
                        net.eval()
                        val_losses = []

                        for k, batch in enumerate(self.validation_loader):
                            validation_images = batch[0].type(torch.FloatTensor).to(device)
                            validation_targets = batch[1].type(torch.FloatTensor).to(device)

                            # Predict
                            masks_pred = net(validation_images)

                            # Calculate loss
                            val_loss = criterion(masks_pred, validation_targets).item()
                            val_losses.append(val_loss)

                            validation_history['val_loss'].append(np.mean(val_loss))
                            validation_history['val_step'].append(i)

                            print('val loss: {0:.5f}, val step: {1:0.5}'.format(np.mean(val_loss), np.mean(i)))

                    net.train()

            print('Epoch finished ! Loss: {}'.format(epoch_loss))
