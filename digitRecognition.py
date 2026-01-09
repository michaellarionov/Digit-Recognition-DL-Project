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
