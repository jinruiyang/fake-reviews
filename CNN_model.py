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
import hiddenlayer as hl

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class model(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(model,self).__init__()
        self.in_channels = 64

        self.conv_1 = conv3x3(1, 16, stride=2)
        self.bn_1 = nn.BatchNorm2d(16)
        self.relu_1 = nn.ReLU(inplace=True)

        self.conv_2 = conv3x3(16, 32, stride=2)
        self.bn_2 = nn.BatchNorm2d(32)
        self.relu_2 = nn.ReLU(inplace=True)

        self.conv_3 = conv3x3(32, 64)
        self.bn_3 = nn.BatchNorm2d(64)
        self.relu_3 = nn.ReLU(inplace=True)


        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1],2)
        self.layer3 = self.make_layer(block, 256, layers[2],2)
        self.layer4 = self.make_layer(block, 512, layers[2],2)
        self.layer5 = self.make_layer(block, 512, layers[2],2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Linear(512, 128)
        self.dropout_1 = nn.Dropout(0.4)
        self.fc_2 = nn.Linear(128, 32)
        self.dropout_2 = nn.Dropout(0.3)
        self.classification = nn.Linear(32,2)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu_1(out)

        # print(out.shape)

        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu_2(out)

        # print(out.shape)

        out = self.conv_3(out)
        out = self.bn_3(out)
        out = self.relu_3(out)

        # print(out.shape)

        out = self.layer1(out)

        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out = self.layer5(out)
        print(out.shape)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_1(out)
        out = self.dropout_1(out)
        out = self.fc_2(out)
        out = self.dropout_2(out)
        out = self.classification(out)
        return out

if __name__ == '__main__':
    test = model(ResidualBlock, [2,2,2])
    dummy_input = torch.randn(1,1,128,768)
    result = test.forward(dummy_input)
    print(result)

