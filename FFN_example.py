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
from model import model

class FFN():
    def __init__(self, num_epochs = 10, batch_size = 100, lr = 0.01, feature_size = 100, pretrained = None):
        self.epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.feature_size = feature_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model(self.feature_size)
        self.model.to(self.device)

        if pretrained:
            self._load_model(pretrained)

    def fit(self, training_data, training_labels, validation_data, validation_labels):
        training_set = self._makeing_dataset(training_data, training_labels)
        validation_set = self._makeing_dataset(validation_data, validation_labels)
        cross_entropy_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        schedule = scheduler.StepLR(optimizer, 1, gamma=0.999)
        for i in range(0, self.epochs):
            self.model.train()
            running_loss = 0.0
            for _ , (inputs, labels) in enumerate(training_set):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    loss = cross_entropy_loss(outputs, labels)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                sys.stdout.flush()
                print('\repoch:{epoch} Loss {loss:.6f}'.format(epoch=i, loss=loss), end="")
                
            epoch_loss = running_loss / len(training_data)
            print(' ')
            print('------------------summary epoch {epoch} ------------------------'.format(epoch = i))
            print('Loss {loss:.6f}'.format( loss=epoch_loss))
            print('----------------------------------------------------------------')
            # schedule.step()
            torch.save({
            'epoch':i,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, './models/model_epoch_{}.pth'.format(i+1))

            self.model.eval()
            for i , (inputs, labels) in enumerate(validation_set):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)
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


    def predict(self, testing_Data):
        self.model.eval()
        testing_Data = torch.tensor(testing_Data, dtype = torch.float)
        for i , inputs in enumerate(testing_Data):
            # inputs = torch.from_numpy(inputs)
            inputs = torch.unsqueeze(inputs, 0).float()
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, pred = torch.max(outputs, 1)
                pred = pred.cpu().detach().numpy()
                if i == 0:
                    predict = pred
                else:
                    predict = np.append(predict, pred)

        return predict

    def _makeing_dataset(self, features, labels):
        # tensor_x = torch.stack([torch.Tensor(i) for i in features]) # transform to torch tensors
        # tensor_y = torch.stack([torch.Tensor(i) for i in labels])

        tensor_x = torch.tensor(features, dtype = torch.float)
        tensor_y = torch.tensor(labels, dtype=torch.long)

        dataset = utils.TensorDataset(tensor_x,tensor_y)
        dataloader = utils.DataLoader(dataset, shuffle= True, batch_size= self.batch_size, num_workers=4)

        return dataloader


if __name__ == '__main__':
    clf = FFN(num_epochs = 1, batch_size = 100, lr = 0.01, feature_size = 100, pretrained='./models/model_epoch_1.pth')
    # clf.fit(np.random.randn(1000,100), np.random.randint(2,size=(1000)), np.random.randn(200,100), np.random.randint(2,size=(200)))
    res = clf.predict(np.random.randn(200,100))
    print(res.shape)
    print(res)


