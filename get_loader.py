import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from random_labels_dataset import *
root = '/app/projs/data'
root = '.'

def get_loader(args):
    if args.dataset=='mnist':
        train_loader = torch.utils.data.DataLoader(
            corruptMNIST(corrupt_prob=args.corrupt, root=root, train=True, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(7),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
            sampler = SubsetRandomSampler(np.random.randint(0, 60000, args.dataset_size)),
            batch_size=args.batch_size)
        
        test_loader = torch.utils.data.DataLoader(
            corruptMNIST(corrupt_prob=args.corrupt, root=root, train=False, transform=transforms.Compose([
                                transforms.Resize(7),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
            #sampler = SubsetRandomSampler(np.arange(5000)),
            sampler = SubsetRandomSampler(np.random.randint(0, 10000, args.test_size)),
            batch_size=args.batch_size)
    elif args.dataset=='fashion_mnist':
        train_loader = torch.utils.data.DataLoader(
            corruptFashionMNIST(corrupt_prob=args.corrupt, root=root, train=True, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(7),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
            sampler = SubsetRandomSampler(np.random.randint(0, 60000, args.dataset_size)),
            batch_size=args.batch_size)
        
        test_loader = torch.utils.data.DataLoader(
            corruptFashionMNIST(corrupt_prob=args.corrupt, root=root, train=False, transform=transforms.Compose([
                                transforms.Resize(7),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
            #sampler = SubsetRandomSampler(np.arange(5000)),
            sampler = SubsetRandomSampler(np.random.randint(0, 10000, args.test_size)),
            batch_size=args.batch_size)
    elif args.dataset=='cifar10':
        N, N_te = min(args.dataset_size, 50000), min(args.dataset_size, 10000)
        args.dataset_size = N
        args.test_size = N_te
        train_loader = torch.utils.data.DataLoader(
            corruptCIFAR10(corrupt_prob=args.corrupt, root=root, train=True, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(7),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                            ])),
            sampler = SubsetRandomSampler(np.random.randint(0, 50000, args.dataset_size)),
            batch_size=args.batch_size)
        
        test_loader = torch.utils.data.DataLoader(
            corruptCIFAR10(corrupt_prob=args.corrupt, root=root, train=False, transform=transforms.Compose([
                                transforms.Resize(7),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                            ])),
            #sampler = SubsetRandomSampler(np.arange(5000)),
        sampler = SubsetRandomSampler(np.random.randint(0, 10000, args.test_size)),
            batch_size=args.batch_size)
    elif args.dataset=='svhn':
        train_loader = torch.utils.data.DataLoader(
            corruptSVHN(corrupt_prob=args.corrupt, root=root, split='train', download=True,
                            transform=transforms.Compose([
                                transforms.Resize(7),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                            ])),
            sampler = SubsetRandomSampler(np.random.randint(0, 60000, args.dataset_size)),
            batch_size=args.batch_size)
        
        test_loader = torch.utils.data.DataLoader(
            corruptSVHN(corrupt_prob=args.corrupt, root=root, split='test', download=True, transform=transforms.Compose([
                                transforms.Resize(7),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                            ])),
            #sampler = SubsetRandomSampler(np.arange(5000)),
        sampler = SubsetRandomSampler(np.random.randint(0, 10000, args.test_size)),
            batch_size=args.batch_size)
    print('Dataset loaders done')
    return train_loader, test_loader
