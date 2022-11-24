import torch
import torchvision
import torchvision.transforms as transforms

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


    return trainset, trainloader, testset, testloader, num_classes

