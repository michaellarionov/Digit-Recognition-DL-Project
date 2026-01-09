from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

trainingData=datasets.MNIST(root="data"
                            , train=True
                            , download=True
                            , transform=ToTensor())

testData=datasets.MNIST(root="data"
                            , train=False
                            , download=True
                            , transform=ToTensor())

loaders = {
    'train': DataLoader(trainingData, batch_size=100, shuffle=True, num_workers=1),
    'test': DataLoader(testData, batch_size=100, shuffle=True, num_workers=1),
}

class CNN(nn.module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.conv2=nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop=nn.Dropout2d()
        self.fc1=nn.Linear(320,50)
        self.fc2=nn.Linear(50,10)