
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import time
import torchvision.transforms as transforms
from randomaug import RandAugment

import progress_bar
from ViT import ViT


bs = 256 # batch size
imgsize = 256 #image size (square)
epochs = 100 # total training epochs
rand_aug = True # use random augmentation

# computing device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(imgsize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(imgsize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# RandAugment
if rand_aug:
    N = 2
    M = 14
    transform_train.transforms.insert(0, RandAugment(N, M))

# Download dataset and define data loader
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

# CIFAR classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## Model
net=ViT(img_size=256, img_channels=3, patch_size= 16, d_model=256, N=4, heads=4,num_classes=len(classes))

# For Multi-GPU
if 'cuda' in device:
    print(device)
    print("using data parallel")
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True

# Loss is CE
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

##### Training
def train():
    
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print('\r %d %d -- Loss: %.1f | Acc: %.1f%%' % (batch_idx+1, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total), end="")

    return train_loss/(batch_idx+1),100.*correct/total

##### Validation
def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            print('\r %d %d -- Loss: %.1f | Acc: %.1f%%' % (batch_idx+1, len(testloader), test_loss/(batch_idx+1), 100.*correct/total), end="")
    return test_loss/(batch_idx+1),100.*correct/total
            

net.cuda()
for epoch in range(epochs):
    start = time.time()
    print('\n============ Epoch: %d ==============' % epoch)
    print("Training")
    trainloss,acc = train()
    print("")
    print("Validation")
    val_loss, acc = test()
    print("")

    scheduler.step() # step cosine scheduling
    
 
