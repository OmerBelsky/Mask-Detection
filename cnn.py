
# data : 
# train.tar:
# https://drive.google.com/open?id=1U_wlpp_A_GFa24lZRuwqqNz5lv4Sw7yN
# test.tar:
# https://drive.google.com/open?id=1k-njgb8xXrp72pQv7k-Ro3foxJRtLWLr


import pandas as pd
import numpy as np 
import itertools
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
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
from utills import createAdamorSGDOptimizer, to_gpu, conv_net
from torchvision.datasets import ImageFolder


#TODO
# image_smallest_size = ## check histogram of image's size in dataset and intialize parameter 


# Parameters and Hyper parameters
image_width = 180
image_height =  180
batch_size = 100
epochs = 10
learning_rate = 0.005 
classes = (0,1)


# pre processing and transformation
train_transform = transforms.Compose([
    transforms.Resize((image_width,image_height)),
    transforms.RandomRotation(30),
    transforms.ToTensor()
                                                                            ])

test_transform = transforms.Compose([
    transforms.Resize((image_width,image_height)),
    transforms.ToTensor()
                                                                            ])


# loading data
train_dataset = ImageFolder("train", transform=train_transform)
train_split_size = round(len(train_dataset) * 0.8)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_split_size, len(train_dataset) - train_split_size])
test_dataset = ImageFolder("test", transform=test_transform)
#TODO: createing labels in some point..


# Dataset Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

# define model, loss and optimizer
cnn = conv_net()
cnn = to_gpu(cnn) # convert all the weights tensors to cuda()
# cnn.load_state_dict(torch.load('cnn1.pkl')) #for loading trained model

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = createAdamorSGDOptimizer(cnn, learning_rate=learning_rate, Adam=1)


# arrays for plotting
auc_train = []
auc_test = []
f1_train = []
f1_test = []
error_train = []
error_test = []
loss_train = []
loss_test = []

### Training the model ###


for epoch in range(epochs):

    for i, (images, labels) in enumerate(train_loader):
        images = to_gpu(images)
        labels = to_gpu(labels)
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, epochs, i+1,
                     len(train_dataset)//batch_size, loss.item()))


    y_train_true = []
    y_train_pred = []
    loss_epoch_train = []
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        y_train_true += labels.tolist()
        images = to_gpu(images)
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)

        y_train_pred += predicted.tolist()
        loss_epoch_train.append(criterion(outputs, to_gpu(labels)).item())
        correct_train += float((predicted.cpu() == labels).sum())
        total_train += float(labels.size(0))

    print('TRAIN - accuracy: %f' % ((correct_train / total_train)*100))
    error_train.append((1-(correct_train / total_train))*100)
    loss_train.append(np.mean(loss_epoch_train))


    y_val_true = []
    y_val_pred = []
    loss_epoch_val = []
    correct_val = 0
    total_val = 0
    for images, labels in val_loader:
        y_val_true += labels.tolist()
        images = to_gpu(images)
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)

        y_val_pred += predicted.tolist()
        loss_epoch_val.append(criterion(outputs, to_gpu(labels)).item())
        correct_val += float((predicted.cpu() == labels).sum())
        total_val += float(labels.size(0))

    print('VAL - accuracy: %f' % ((correct_val / total_val)*100))
    error_test.append((1-(correct_val / total_val))*100)
    loss_test.append(np.mean(loss_epoch_val))
    auc_train.append(roc_auc_score(y_train_true, y_train_pred))
    auc_test.append(roc_auc_score(y_val_true, y_val_pred))
    f1_train.append(f1_score(y_train_true, y_train_pred))
    f1_test.append(f1_score(y_val_true, y_val_pred))

    print(loss_train)
    print(loss_test)


plt.plot(range(epochs), loss_test)
plt.plot(range(epochs), loss_train)
plt.legend(["Test", "Train"])
plt.title("Model Loss per epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(range(epochs), error_test)
plt.plot(range(epochs), error_train)
plt.legend(["Test", "Train"])
plt.title("Model Error per epoch")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()

plt.plot(range(epochs), auc_test)
plt.plot(range(epochs), auc_train)
plt.legend(["Test", "Train"])
plt.title("Model AUC per epoch")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()

plt.plot(range(epochs), f1_test)
plt.plot(range(epochs), f1_train)
plt.legend(["Test", "Train"])
plt.title("Model F1 per epoch")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()
### Test ###

# model = torch.load('Model.pkl')

correct = 0
total = 0
for images, labels in test_loader:
    images = to_gpu(images)
    labels = to_gpu(labels)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    # if ((predicted.cpu() == labels.cpu()).sum().item() == 0) and labels.cpu().item() == 1:
    #     print(labels)
    #     plt.imshow(images.cpu().squeeze().reshape(180, 180, 3))
    #     plt.show()
    # if (predicted.cpu() == labels.cpu()).sum().item() == 0:
        # plt.imshow(images)

    total += labels.size(0)
    correct += (predicted.cpu() == labels.cpu()).sum()

print('Accuracy of the model on the test images: %.4f %%' % (float(correct) / total))



# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn2.pkl')