import sys
import numpy as np 
import sklearn.metrics as metrics
from sklearn.metrics import recall_score
import pickle
import torch
import torch.utils.data as utils
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn as nn
from CNN_model import model, ResidualBlock
from center_loss import CenterLoss
import torchvision

class CNN():
    def __init__(self, num_epochs = 10, lr = 0.01, batch_size=64, pretrained = None):
        self.epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model(ResidualBlock,[2, 2, 2])
        self.model.to(self.device)

        if pretrained:
            self._load_model(pretrained)

    def fit(self, training_set,testing_set, alpha = 0.9):
        # center_loss = CenterLoss(num_classes=2, feat_dim=4, use_gpu=True)
        cross_entropy_loss = nn.CrossEntropyLoss()
        # params = list(self.model.parameters()) + list(center_loss.parameters())
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        # schedule = scheduler.CosineAnnealingLR(optimizer, self.epochs, eta_min=1e-4)
        for i in range(0, self.epochs):
            self.model.train()
            running_loss = 0.0
            running_corrects = 0.0
            for j , (inputs, labels) in enumerate(training_set):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # loss_1 = cross_entropy_loss(outputs, labels)
                    # loss_2 = center_loss(features,labels)
                    # loss = [alpha*loss_1, (1-alpha)*loss_2]

                    loss = cross_entropy_loss(outputs, labels)
                    loss.backward()
                    # torch.autograd.backward(loss)
                    # for param in center_loss.parameters():
                    #     param.grad.data *= (1./alpha)
                    optimizer.step()

                # running_loss += (loss_1 + loss_2) * inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                correct = torch.sum(preds == labels.data).double()/float(self.batch_size)
                running_corrects += correct
                sys.stdout.flush()
                print('\repoch:{epoch} Loss: {loss:.6f} acc: {acc:.6f}'.format(epoch=i, loss= loss, acc=correct), end="")
                
            epoch_loss = running_loss / len(training_set.dataset)
            epoch_acc = running_corrects / j
            print(' ')
            print('------------------summary epoch {epoch} ------------------------'.format(epoch = i))
            print('Loss: {loss:.6f} acc: {acc:.6f}'.format( loss=epoch_loss, acc = epoch_acc))
            print('----------------------------------------------------------------')
            # schedule.step()
            torch.save({
            'epoch':i,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, './models/model_epoch_{}.pth'.format(i+1))

            self.model.eval()
            running_val_loss = 0.0
            for i , (inputs, labels) in enumerate(testing_set):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)
                    loss = cross_entropy_loss(outputs, labels) 
                    _, pred = torch.max(outputs, 1)
                    pred = pred.cpu().detach().numpy()
                    truth = labels.cpu().detach().numpy()
                    pred = pred.reshape(truth.shape)
                    if i == 0:
                        predict = pred
                        truths = truth
                    else:
                        predict = np.append(predict, pred)
                        truths = np.append(truths, truth)
                running_val_loss += loss.item() * inputs.size(0)
            val_loss = running_val_loss / len(testing_set.dataset)
            print('val loss:', val_loss)
            accuracy_test = metrics.accuracy_score(truths, predict)
            precision_test = metrics.precision_score(truths, predict)
            f1_val = metrics.f1_score(truths, predict)
            recall_test = recall_score(truths, predict)
            print('validation result')
            print('accuracy:', accuracy_test)
            print(metrics.classification_report(truths, predict))
            print(' ')

    def _load_model(self, model_path):
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt['model_state_dict'])


    def predict(self, testing_set):
        self.model.eval()
        # testing_Data = torch.tensor(testing_Data, dtype = torch.float)
        for i , inputs in enumerate(testing_set):
            # inputs = torch.from_numpy(inputs)
            inputs = torch.unsqueeze(inputs, 0).float()
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                outputs, features = self.model(inputs)
                _, pred = torch.max(outputs, 1)
                pred = pred.cpu().detach().numpy()
                if i == 0:
                    predict = pred
                else:
                    predict = np.append(predict, pred)

        return predict

def npy_loader(in_file):
    x = np.load(in_file)
    x = np.expand_dims(x,axis=0)
    return x


if __name__ == '__main__':
    BATCH_SIZE = 128
    clf = CNN(num_epochs = 20,batch_size = BATCH_SIZE, lr = 0.0005)
    train_data= torchvision.datasets.DatasetFolder(root='./dataset/features/train', loader=npy_loader, extensions='.npy')
    train_loader = torch.utils.data.DataLoader( 
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True)

    test_data= torchvision.datasets.DatasetFolder(root='./dataset/features/test', loader=npy_loader, extensions='.npy')
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=True)

    clf.fit(train_loader, test_loader)


