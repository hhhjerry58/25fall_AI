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
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
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

b_values = [i * 0.01 for i in range(21)] # i from 0.00 to 0.20
best_b = -1
min_final_val_error = 101

print("--- Starting search for optimal b ---")
search_epochs = 500

for b in b_values:
    print(f"--- Training for b = {b:.2f} ---")
    
    model = OneLayerNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(search_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            scores = model(images)
            loss = abs(criterion(scores, labels) - b) + b
            loss.backward()
            optimizer.step()

    model.eval()
    val_tot_num = 0
    val_acc_num = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)
            prediction = scores.argmax(dim=1)
            val_tot_num += images.size(0)
            val_acc_num += (prediction == labels).sum().item()

    val_accuracy = 100 * (val_acc_num / val_tot_num)
    final_val_error = 100 - val_accuracy
    
    print(f"Final validation error for b={b:.2f}: {final_val_error:.2f}%")

    if final_val_error < min_final_val_error:
        min_final_val_error = final_val_error
        best_b = b

print(f"\n--- Search Complete ---")
print(f"Optimal b found: {best_b} with a validation error of {min_final_val_error:.2f}%")


print(f"\n--- Training with optimal b = {best_b} for 500 epochs ---")

model = OneLayerNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

train_errors = []
val_errors = []
num_epochs_final = 500

for epoch in range(num_epochs_final):
    model.train()
    train_tot_num = 0
    train_acc_num = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        scores = model(images)
        loss = abs(criterion(scores, labels) - best_b) + best_b
        loss.backward()
        optimizer.step()

        prediction = scores.argmax(dim=1)
        train_tot_num += images.size(0)
        train_acc_num += (prediction == labels).sum().item()
    
    train_accuracy = 100 * (train_acc_num / train_tot_num)
    train_errors.append(100 - train_accuracy)

    model.eval()
    val_tot_num = 0
    val_acc_num = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)
            prediction = scores.argmax(dim=1)
            val_tot_num += images.size(0)
            val_acc_num += (prediction == labels).sum().item()

    val_accuracy = 100 * (val_acc_num / val_tot_num)
    val_errors.append(100 - val_accuracy)

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs_final}], Train Error: {100-train_accuracy:.2f}%, Val Error: {100-val_accuracy:.2f}%')

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs_final + 1), train_errors, label='Training Error (%)', color='blue')
plt.plot(range(1, num_epochs_final + 1), val_errors, label='Validation Error (%)', color='red')
plt.xlabel('Epoch')
plt.ylabel('Error (%)')
plt.title(f'Training and Validation Error (Optimal b = {best_b})')
plt.legend()
plt.grid(True)
plt.show()
 
    

