import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

def get_cifar10_loader(bs=256):
#     classes = ('plane', 'car', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck')

    train_transform = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = tv.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=2)

    test_dataset = tv.datasets.CIFAR10(
        root='../data', train=False, download=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=bs, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def get_cifar100_loader(bs=256):

    train_transform = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    test_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    train_dataset = tv.datasets.CIFAR100(
        root='../data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=2)

    test_dataset = tv.datasets.CIFAR100(
        root='../data', train=False, download=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=bs, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def get_svhn_loader(bs=256):

    train_transform = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    test_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    train_dataset = tv.datasets.SVHN(
        root='../data', split='train', download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=2)

    test_dataset = tv.datasets.SVHN(
        root='../data', split='test', download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=bs, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


# def get_imagenet_loader(bs=32):
#     train_dataset = tv.datasets.ImageFolder(
#         '../data/imagenet/train',
#         tv.transforms.Compose([
#             tv.transforms.RandomResizedCrop(224),
#             tv.transforms.RandomHorizontalFlip(),
#             tv.transforms.ToTensor(),
#             tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225]),
#         ]))
#     test_dataset = tv.datasets.ImageFolder(
#         '../data/imagenet/val', 
#         tv.transforms.Compose([
#             tv.transforms.Resize(256),
#             tv.transforms.CenterCrop(224),
#             tv.transforms.ToTensor(),
#             tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225]),
#         ]))
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=bs, shuffle=True,
#         num_workers=16, pin_memory=True)
#     test_loader = torch.utils.data.DataLoader(
#         test_dataset, batch_size=bs, shuffle=False,
#         num_workers=16, pin_memory=True)
# 
#     return train_loader, test_loader
