# ----------------------------------------------------------------------------------
# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import pickle
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import sys
import numpy as np
from collections import namedtuple
from itertools import count
import random
import time

# ----------------------------------------------------------------------------------
# Global paramerets
use_WeightDecay = 0
use_BatchNorm = 0
use_Dropout = 0
dropout_prob = 0.2
learning_rate = 1e-3
num_epochs = 10
evaluation_mode = 0

# ----------------------------------------------------------------------------------
# Things for running on GPU
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


# ----------------------------------------------------------------------------------
# Dataset
batch_size = 100

train_dataset = dsets.FashionMNIST(root='./data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
test_dataset = dsets.FashionMNIST(root='./data',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


num_images = train_dataset.train_data.shape[0]
image_x = train_dataset.train_data.shape[1]
image_y = train_dataset.train_data.shape[2]
num_classes = len(set(train_dataset.train_labels))

# ----------------------------------------------------------------------------------
# Neural Network architecture and helper classes
criterion = nn.CrossEntropyLoss()
optimizer_type = torch.optim.Adam
optimizer_params =dict(lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=use_WeightDecay)
class NN_basic(nn.Module):
    def __init__(self, num_classes, use_BatchNorm, use_Dropout, dropout_prob=0.2):
        super(NN_basic, self).__init__()
        # 1x32x32   ->   6x28x28
        self.C1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # 6x28x28  ->   6x14x14
        self.S2 = nn.MaxPool2d(2)
        # 6x14x14    ->   16x10x10
        self.C3 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        # 16x10x10  ->   16x5x5
        self.S4 = nn.MaxPool2d(2)
        self.F5 = nn.Linear(16*5*5, 120)
        self.F6 = nn.Linear(120, 84)
        self.F7 = nn.Linear(84, num_classes)

        self.BN1 = nn.BatchNorm2d(6)
        self.BN3 = nn.BatchNorm2d(16)
        self.BN5 = nn.BatchNorm1d(120)
        self.BN6 = nn.BatchNorm2d(84)

        self.D1 = nn.Dropout2d(dropout_prob)
        self.D3 = nn.Dropout2d(dropout_prob)
        self.D5 = nn.Dropout(dropout_prob)
        self.D6 = nn.Dropout(dropout_prob)

        self.use_BatchNorm = use_BatchNorm
        self.use_Dropout = use_Dropout

    def forward(self, input):
        x = self.C1(input)
        if self.use_Dropout:
            x = self.D1(x)
        if self.use_BatchNorm:
            x = self.BN1(x)
        x = F.relu(x)
        x = self.S2(x)
        x = self.C3(x)
        if self.use_Dropout:
            x = self.D3(x)
        if self.use_BatchNorm:
            x = self.BN3(x)
        x = F.relu(x)
        x = self.S4(x)
        x.view(x.size(0), -1)
        x = self.F5(x.view(x.size(0), -1))
        if self.use_Dropout:
            x = self.D5(x)
        if self.use_BatchNorm:
            x = self.BN5(x)
        x = F.relu(x)
        x = self.F6(x)
        if self.use_Dropout:
            x = self.D6(x)
        if self.use_BatchNorm:
            x = self.BN6(x)
        x = F.relu(x)
        output = self.F7(x)
        return output

my_net = NN_basic(num_classes, use_BatchNorm, use_Dropout, dropout_prob)
if (USE_CUDA):
    my_net.cuda()

optimizer = optimizer_type(my_net.parameters(), **optimizer_params)




Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": [],
    "running_times": []
}

# ----------------------------------------------------------------------------------
# Finally - train the model!
if (evaluation_mode == 1):
    my_net.load_state_dict(torch.load('model.pkl'))
else:
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            images = Variable(images)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = my_net(images)
            loss = criterion(outputs, Variable(labels))
            loss.backward()
            optimizer.step()

            a, winner_class = torch.max(outputs, -1)

            correct = sum(winner_class.data.numpy() == labels)
            total = labels.size(0)

            if (i % (1000/batch_size) == 0):
                print("[Epoch=" + str(epoch) + ",Batch=" + str(i)+ ",Accuracy=" + str(correct/total) + ": loss is: %.3f" % loss.data[0] + "]")
        torch.save(my_net.state_dict(), 'model.pkl')

torch.save(my_net.state_dict(), 'model.pkl')
# Test the Model
my_net.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = my_net(images)
    a, winner_class = torch.max(outputs,0)

    correct += sum(winner_class.data.numpy() == labels)
    total += labels.size(0)

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


#with open('statistics.pkl', 'wb') as f:
#    pickle.dump(Statistic, f)
#    print("Saved to %s" % 'statistics.pkl')


#


