import torch
import torch.optim as opt
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm


# torch.set_default_tensor_type('torch.FloatTensor')


class UNetTrain:
    def __init__(self, train_length, train_loader, validation_length, validation_loader, lr=0.0001,
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
        weights = [1 / 60, 1.0, 1.0, 1.0, 2.0]
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_history = {'loss': [], 'train_step': []}
        validation_history = {'val_loss': [], 'val_epoch': []}

        # Train the model
        total_step = len(self.train_loader)
        for epoch in tqdm(range(self.epochs)):
            print('Starting epoch {}/{}.'.format(epoch + 1, self.epochs))
            net.train()
            epoch_loss = 0
            for i, batch in enumerate(self.train_loader):
                start = time.time()
                input, label = batch
                images = input.type(torch.FloatTensor).to(device).permute(0, 3, 1, 2)
                labels = label.type(torch.LongTensor).to(device)
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
                if i % 100 == 0:
                    train_history['loss'].append(loss.item())
                    train_history['train_step'].append(i + epoch * total_step)

                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, self.epochs, i + 1, total_step, loss.item()))

            if save_cp:
                torch.save(net.state_dict(), model_checkpoints + 'CP{}.pth'.format(epoch + 1))
                print('Checkpoint {} saved !'.format(epoch + 1))

                print('\nEval..')
                # eval
                net.eval()
                val_losses = []

                for k, val_batch in tqdm(enumerate(self.validation_loader)):
                    val_start = time.time()
                    validation_input, validation_label = val_batch
                    validation_images = validation_input.type(torch.FloatTensor).to(device).permute(0, 3, 1, 2)
                    validation_targets = validation_label.type(torch.LongTensor).to(device)

                    # Predict
                    masks_pred = net(validation_images)

                    # Calculate loss
                    val_loss = criterion(masks_pred, validation_targets).item()
                    val_losses.append(val_loss)

                    validation_history['val_loss'].append(np.mean(val_losses))
                    validation_history['val_epoch'].append(epoch + 1)

                    val_end = time.time()
                    print('Time taken for the batch is {}'.format(val_end - val_start))
                print('Validation Loss: {}, Validation Epoch: {}'.format(np.mean(val_losses), epoch + 1))

            net.train()

            print('Epoch finished ! Loss: {}'.format(np.mean(epoch_loss)))
