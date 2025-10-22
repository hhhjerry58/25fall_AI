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
class OneLayerNet(nn.Module):
    def __init__(self):
        super(OneLayerNet,self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(28*28,500)
        self.linear2 = nn.Linear(500,10)
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = OneLayerNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(),momentum=0.9,lr=0.1)

train_errors = []
val_errors = []
for epoch in range(500):

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
    

