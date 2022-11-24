
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import os

# TorchVision and data augmentation
import torchvision
import torchvision.transforms as transforms
from randomaug import RandAugment

# Models
from ViT import ViT

# Data
from getdata import getData


# Some important parameters
bs = 128 # batch size
img_size = 32 #image size (square)
epochs = 200 # total training epochs
rand_aug = True # use random augmentation
img_channels=3 # number of image channels
patch_size= 4 # patch size (square)
d_model=384 # dimensionality transformer representation
N=7 # Number of transformers blocks
heads=12 # Number of transformer block heads 
load_check = False # to load a checkpoint


# computing device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# RandAugment
if rand_aug:
    N = 2
    M = 14
    transform_train.transforms.insert(0, RandAugment(N, M))

trainset,trainloader,testset,testloader,num_classes=getData("CIFAR100", bs, transform_train, transform_test)

## Model
net=ViT(img_size, img_channels, patch_size, d_model, N, heads, num_classes)
net.to(device)

best_acc = 0.0
if load_check:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/vit-ckpt.t7')
    net.load_state_dict(checkpoint['model_state_dict'])
    best_acc = checkpoint['acc']


# Loss is CE
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.00005)

if load_check:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# use reduce on plateau
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)

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

        print('\r %d %d -- Loss: %.3f | Acc: %.2f%%' % (batch_idx+1, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total), end="")

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
            
            print('\r %d %d -- Loss: %.3f | Acc: %.2f%%' % (batch_idx+1, len(testloader), test_loss/(batch_idx+1), 100.*correct/total), end="")

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print("")
        print('Saving checkpoint..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            'acc': acc}, './checkpoint/vit-ckpt.t7')
        best_acc = acc

    return test_loss/(batch_idx+1),100.*correct/total
            

for epoch in range(epochs):
    print('\n============ Epoch: %d ==============' % epoch)
    print()

    print("Training, lr= %f" %(optimizer.param_groups[0]['lr']))
    trainloss,acc = train()
    print("")

    print("Validation, best acc=%f" %(best_acc))
    val_loss, acc = test()
    print("")

    #scheduler.step(trainloss) # step scheduling
    scheduler.step() # step scheduling
    
    
 
