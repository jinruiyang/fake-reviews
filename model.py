import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

class model(nn.Module):
    def __init__(self, input_feature= 7):
        super(model,self).__init__()

        self.model = nn.Sequential(
        nn.Linear(input_feature,16),
        nn.ReLU(),
        nn.Linear(16,64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64,256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256,512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512,128),
        nn.ReLU(),
        nn.Linear(128,32),
        nn.ReLU(),
        nn.Linear(32,2),
        )

    def forward(self,x):
        result = self.model(x)

        return result

# test = model()
# dummy_input = torch.randn(2,7)
# result = test.forward(dummy_input)
# print(result)

