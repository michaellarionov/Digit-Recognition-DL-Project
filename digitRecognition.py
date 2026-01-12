from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

transform = transforms.Compose([
    transforms.Pad(2),      # 28×28 → 32×32
    transforms.ToTensor()
])

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
# Define the CNN architecture
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)      
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)     
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)   

        self.fc1 = nn.Linear(120, 84)                    
        self.fc2 = nn.Linear(84, 10)                     

    def forward(self, x):
        x = F.avg_pool2d(torch.tanh(self.conv1(x)), 2)
        x = F.avg_pool2d(torch.tanh(self.conv2(x)), 2)
        x = torch.tanh(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=LeNet5().to(device)
optimizer=optim.Adam(model.parameters(), lr=0.001)
lossFunction=nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_index, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = lossFunction(outputs, target)
        loss.backward()
        optimizer.step()
        if batch_index % 20 == 0:
            print(f"Train Epoch: {epoch} [{batch_index * len(data)}/{len(loaders['train'].dataset)}"
                  f" ({100. * batch_index / len(loaders['train']):.0f}%)]\tLoss: {loss.item():.6f}")
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += lossFunction(outputs, target).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders['test'].dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders['test'].dataset)} "
          f"({100. * correct / len(loaders['test'].dataset):.0f}%)")

for epoch in range(1, 11):
    train(epoch)
    test()