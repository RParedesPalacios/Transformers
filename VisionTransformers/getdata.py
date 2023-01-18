import torch
import torchvision
import torchvision.transforms as transforms
from randomaug import RandAugment


## CIFAR10: 32x32 10 classes
## CIFAR100: 32x32 100 classes
## FOOD101: 512x512(max) 101 classes
## FLOWERS102: small 102 classes

def getData(name, bs, transform_train, transform_test):
    if name=="CIFAR10":
        print("Running over CIFAR10")
        # Download dataset and define data loader
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

        num_classes = 10
        
    elif name=="CIFAR100":
        print("Running over CIFAR100")
        # Download dataset and define data loader
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

        num_classes = 100

    elif name=="FOOD101":
        print("Running over FOOD101")
        # Download dataset and define data loader
        trainset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

        testset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

        num_classes = 101

    elif name=="FLOWERS102":
        print("Running over FLOWERS102")
        # Download dataset and define data loader
        trainset = torchvision.datasets.Flowers102(root='./data', split='train', download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

        testset = torchvision.datasets.Flowers102(root='./data', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

        num_classes = 102

    return trainset, trainloader, testset, testloader, num_classes


## DATA AUGMENTATION
def getAugmentation(img_size,rand_aug):
    transform_train = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.RandomCrop(img_size, padding=img_size//8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    # RandAugment
    if rand_aug:
        N = 2
        M = 14
        transform_train.transforms.insert(0, RandAugment(N, M))

    return transform_train,transform_test

