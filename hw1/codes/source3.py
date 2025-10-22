import sys
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device:{device}')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]
) # standard transform for MNIST dataset

train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)

pic_num = len(train_dataset)
train_size , val_size = int(pic_num * 0.8), int(pic_num * 0.2)
print(f'length of train_dataset:{pic_num}, train_size:{train_size}, val_size:{val_size}')

train_set, val_set = random_split(dataset=train_dataset,lengths=[train_size,val_size])

batch_size = 64
train_loader = DataLoader(dataset=train_set,batch_size = batch_size,shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

class convNet(nn.Module):
    def __init__(self):
        super(convNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2) 
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(32*7*7, 10)
    def forward(self, x):
        x = self.conv1(x) # [batch,16,28,28]
        x = self.relu(x)
        x = self.pooling(x) # [batch,16,14,14]

        x = self.conv2(x) # [batch,32,14,14]
        x = self.relu(x)
        x = self.pooling(x) # [batch,32,7,7]

        x = self.flatten(x)
        x = self.linear(x) # [batch,10]
        return x

model = convNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(),momentum=0.9,lr=0.1)

train_errors = []
val_errors = []
for epoch in range(500): 
    model.train()

    train_accuracy = 0
    val_accuracy = 0
    tot_num = 0
    acc_num = 0

    for images, lables in train_loader:
        images, lables = images.to(device), lables.to(device) # now shape of images is [64,1,28,28]
        
        optimizer.zero_grad()
        scores = model(images) # shape of scores:[64,10]
        loss = criterion(scores, lables)
        loss.backward()
        optimizer.step()

        prediction = scores.argmax(dim=1)
        tot_num += images.size(0)
        acc_num += (prediction==lables).sum()
    
    train_accuracy = 100 * (acc_num / tot_num).cpu().numpy()
    train_errors.append(100 - train_accuracy) 

    model.eval()

    tot_num = 0
    acc_num = 0
    with torch.no_grad():
        for images, lables in val_loader:
            images, lables = images.to(device), lables.to(device)
            scores = model(images)

            prediction = scores.argmax(dim=1)
            tot_num += images.size(0)
            acc_num += (prediction==lables).sum()

        val_accuracy = 100 * (acc_num / tot_num).cpu().numpy()
        val_errors.append(100 - val_accuracy)

    if epoch % 20 == 0:
        print(f'train_acc:{train_accuracy}, val_acc:{val_accuracy}')

plt.figure(figsize=(10, 5))
plt.plot(range(1, 501), train_errors, label='Training Error (%)', color='blue')
plt.plot(range(1, 501), val_errors, label='Validation Error (%)', color='red')
plt.xlabel('Epoch')
plt.ylabel('Error (%)')
plt.title('Training and Validation Error over Epochs')
plt.legend()
plt.grid(True)
plt.show() 
    

