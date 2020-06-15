import pandas as pd
import numpy as np 
import itertools
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math 
import datetime
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms





def createAdamorSGDOptimizer(net, learning_rate=0.001, Adam=1):
    
    if Adam == 1:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    return(optimizer)


def to_gpu(x):
        return x.cuda() if torch.cuda.is_available() else x


        
class conv_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32 , kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
            # nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))     
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))       
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU())
            # nn.MaxPool2d(2))
        self.layer6 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU())
            # nn.MaxPool2d(2))

        self.fc = nn.Linear(800, 2)
        self.dropout1 = nn.Dropout(p=0.01)
        self.dropout2 = nn.Dropout(p=0.02)
        self.dropout3 = nn.Dropout(p=0.03)
        self.logsoftmax = nn.LogSoftmax()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.dropout1(out)
        out = self.layer2(out)
        out = self.dropout1(out)
        out = self.layer3(out)
        out = self.dropout2(out)
        out = self.layer4(out)
        out = self.dropout2(out)
        out = self.layer5(out)
        out = self.dropout2(out)
        out = self.layer6(out)
#         out = self.dropout2(out)
        out = self.layer7(out)
        out = out.view(out.size(0), -1)
        out = self.dropout3(out)
        out = self.fc(out)
        return self.logsoftmax(out)
