import numpy as np
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader
from utils import GaussianBlur, get_multiclient_trainloader_list
from PIL import Image
import os

STL10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
STL10_TRAIN_STD = (0.2471, 0.2435, 0.2616)
CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
TINYIMAGENET_TRAIN_MEAN = (0.5141, 0.5775, 0.3985)
TINYIMAGENET_TRAIN_STD = (0.2927, 0.2570, 0.1434)
SVHN_TRAIN_MEAN = (0.3522, 0.4004, 0.4463)
SVHN_TRAIN_STD = (0.1189, 0.1377, 0.1784)
IMAGENET_TRAIN_MEAN = (0.485, 0.456, 0.406)
IMAGENET_TRAIN_STD = (0.229, 0.224, 0.225)

def denormalize(x, dataset): # normalize a zero mean, std = 1 to range [0, 1]
    
    if dataset == "cifar10":
        std = [0.2023, 0.1994, 0.2010]
        mean = [0.4914, 0.4822, 0.4465]
    elif dataset == "cifar100":
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    elif dataset == "imagenet":
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
    elif dataset == "tinyimagenet":
        std = (0.2927, 0.2570, 0.1434)
        mean = (0.5141, 0.5775, 0.3985)   
    elif dataset == "svhn":
        std = (0.1189, 0.1377, 0.1784)
        mean = (0.3522, 0.4004, 0.4463)
    elif dataset == "stl10":
        std = (0.2471, 0.2435, 0.2616)
        mean = (0.4914, 0.4822, 0.4465)
    # 3, H, W, B
    tensor = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(range(tensor.size(0)), mean, std):
        tensor[t] = tensor[t].mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(tensor, 0, 1).permute(3, 0, 1, 2)


def get_cifar10(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    if pairloader_option != "None":
        if data_proportion > 0.0:
            train_loader = get_cifar10_pairloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, pairloader_option, hetero, hetero_string, path_to_data)
        else:
            train_loader = None
        mem_loader = get_cifar10_trainloader(128, num_workers, False, path_to_data = path_to_data)
        test_loader = get_cifar10_testloader(128, num_workers, False, path_to_data)
        return train_loader, mem_loader, test_loader
    else:
        if data_proportion > 0.0:
            train_loader = get_cifar10_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string, path_to_data)
        else:
            train_loader = None
        test_loader = get_cifar10_testloader(128, num_workers, False, path_to_data)
        return train_loader, test_loader

def get_cifar100(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    if pairloader_option != "None":
        train_loader = get_cifar100_pairloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, pairloader_option, hetero, hetero_string, path_to_data)
        mem_loader = get_cifar100_trainloader(128, num_workers, False, path_to_data = path_to_data)
        test_loader = get_cifar100_testloader(128, num_workers, False, path_to_data)
        return train_loader, mem_loader, test_loader
    else:
        train_loader = get_cifar100_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string, path_to_data)
        test_loader = get_cifar100_testloader(128, num_workers, False, path_to_data)
        return train_loader, test_loader

def get_tinyimagenet(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./tiny-imagenet-200"):
    if pairloader_option != "None":
        train_loader = get_tinyimagenet_pairloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, pairloader_option, hetero, hetero_string, path_to_data)
        mem_loader = get_tinyimagenet_trainloader(128, num_workers, False, path_to_data = path_to_data)
        test_loader = get_tinyimagenet_testloader(128, num_workers, False, path_to_data)
        return train_loader, mem_loader, test_loader
    else:
        train_loader = get_tinyimagenet_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string, path_to_data)
        test_loader = get_tinyimagenet_testloader(128, num_workers, False, path_to_data)
        return train_loader, test_loader

def get_imagenet12(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data/imagnet-12"):
    if pairloader_option != "None":
        train_loader = get_imagenet12_pairloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, pairloader_option, hetero, hetero_string, path_to_data)
        mem_loader = get_imagenet12_trainloader(128, num_workers, False, path_to_data = path_to_data)
        test_loader = get_imagenet12_testloader(128, num_workers, False, path_to_data)
        return train_loader, mem_loader, test_loader
    else:
        train_loader = get_imagenet12_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string, path_to_data)
        test_loader = get_imagenet12_testloader(128, num_workers, False, path_to_data)
        return train_loader, test_loader

def get_stl10(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    if pairloader_option != "None":
        train_loader = get_stl10_pairloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, pairloader_option, hetero, hetero_string, path_to_data)
        mem_loader = get_stl10_trainloader(128, num_workers, False, path_to_data = path_to_data)
        test_loader = get_stl10_testloader(128, num_workers, False, path_to_data)
        return train_loader, mem_loader, test_loader
    else:
        train_loader = get_stl10_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string, path_to_data)
        test_loader = get_stl10_testloader(128, num_workers, False, path_to_data)
        return train_loader, test_loader

def get_svhn(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    if pairloader_option != "None":
        train_loader = get_svhn_pairloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, pairloader_option, hetero, hetero_string, path_to_data)
        mem_loader = get_SVHN_trainloader(128, num_workers, False, path_to_data = path_to_data)
        test_loader = get_SVHN_testloader(128, num_workers, False, path_to_data)
        return train_loader, mem_loader, test_loader
    else:
        train_loader = get_SVHN_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string, path_to_data)
        test_loader = get_SVHN_testloader(128, num_workers, False, path_to_data)
        return train_loader, test_loader

# def get_svhn(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2"):
#     train_loader = get_SVHN_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string)
#     test_loader = get_SVHN_testloader(128, num_workers, False)
#     return train_loader, test_loader

# def get_imagenet(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2"):
#     train_loader = get_imagenet_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string)
#     test_loader = get_imagenet_testloader(128, num_workers, False)
#     return train_loader, test_loader

def get_tinyimagenet_pairloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./tiny-imagenet-200"):
    class tinyimagenetPair(torchvision.datasets.ImageFolder):
        """tinyimagenet Dataset.
        """
        def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            path, _ = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                im_1 = self.transform(sample)
                im_2 = self.transform(sample)
            
            return im_1, im_2

    # tinyimagenet_training = datasets.ImageFolder('tiny-imagenet-200/train', transform=transform_train)
    # tinyimagenet_testing = datasets.ImageFolder('tiny-imagenet-200/val', transform=transform_test)
    if pairloader_option == "mocov1":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(TINYIMAGENET_TRAIN_MEAN, TINYIMAGENET_TRAIN_STD)])
    elif pairloader_option == "mocov2":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(TINYIMAGENET_TRAIN_MEAN, TINYIMAGENET_TRAIN_STD)])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TINYIMAGENET_TRAIN_MEAN, TINYIMAGENET_TRAIN_STD)])
    # data prepare
    train_data = tinyimagenetPair(f'{path_to_data}/train', transform=train_transform)
    
    indices = torch.randperm(len(train_data))[:int(len(train_data)* data_portion)]

    train_data = torch.utils.data.Subset(train_data, indices)
    
    cifar100_training_loader = get_multiclient_trainloader_list(train_data, num_client, shuffle, num_workers, batch_size, noniid_ratio, 200, hetero, hetero_string)
    
    return cifar100_training_loader


def get_cifar100_pairloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    class CIFAR100Pair(torchvision.datasets.CIFAR100):
        """CIFAR100 Dataset.
        """
        def __getitem__(self, index):
            img = self.data[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                im_1 = self.transform(img)
                im_2 = self.transform(img)

            return im_1, im_2
    if pairloader_option == "mocov1":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])
    elif pairloader_option == "mocov2":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])
    # data prepare
    train_data = CIFAR100Pair(root=path_to_data, train=True, transform=train_transform, download=True)
    
    indices = torch.randperm(len(train_data))[:int(len(train_data)* data_portion)]

    train_data = torch.utils.data.Subset(train_data, indices)
    
    cifar100_training_loader = get_multiclient_trainloader_list(train_data, num_client, shuffle, num_workers, batch_size, noniid_ratio, 100, hetero, hetero_string)
    
    return cifar100_training_loader

def get_cifar10_pairloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    class CIFAR10Pair(torchvision.datasets.CIFAR10):
        """CIFAR10 Dataset.
        """
        def __getitem__(self, index):
            img = self.data[index]
            img = Image.fromarray(img)
            if self.transform is not None:
                im_1 = self.transform(img)
                im_2 = self.transform(img)

            return im_1, im_2
    if pairloader_option == "mocov1":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    elif pairloader_option == "mocov2":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    # data prepare
    
    train_data = CIFAR10Pair(root=path_to_data, train=True, transform=train_transform, download=True)
    
    indices = torch.randperm(len(train_data))[:int(len(train_data)* data_portion)]

    train_data = torch.utils.data.Subset(train_data, indices)
    
    cifar10_training_loader = get_multiclient_trainloader_list(train_data, num_client, shuffle, num_workers, batch_size, noniid_ratio, 10, hetero, hetero_string)
    
    return cifar10_training_loader

def get_tinyimagenet_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, augmentation_option = False, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./tiny-imagenet-200"):
    """ return training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(TINYIMAGENET_TRAIN_MEAN, TINYIMAGENET_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TINYIMAGENET_TRAIN_MEAN, TINYIMAGENET_TRAIN_STD)
        ])

    if not os.path.isdir(f"{path_to_data}/train"):
        import subprocess
        subprocess.call("python prepare_tinyimagenet.py", shell=True)
    tinyimagenet_training = datasets.ImageFolder(f'{path_to_data}/train', transform=transform_train)
    

    indices = torch.randperm(len(tinyimagenet_training))[:int(len(tinyimagenet_training)* data_portion)]

    tinyimagenet_training = torch.utils.data.Subset(tinyimagenet_training, indices)

    tinyimagenet_training_loader = get_multiclient_trainloader_list(tinyimagenet_training, num_client, shuffle, num_workers, batch_size, noniid_ratio, 200, hetero, hetero_string)

    

    return tinyimagenet_training_loader

def get_tinyimagenet_testloader(batch_size=16, num_workers=2, shuffle=False, path_to_data = "./tiny-imagenet-200"):
    """ return training dataloader
    Returns: imagenet_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(TINYIMAGENET_TRAIN_MEAN, TINYIMAGENET_TRAIN_STD)
    ])
    tinyimagenet_testing = datasets.ImageFolder(f'{path_to_data}/val', transform=transform_test)
    tinyimagenet_testing_loader = torch.utils.data.DataLoader(tinyimagenet_testing,  batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers)
    return tinyimagenet_testing_loader



def get_imagenet_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, augmentation_option = False, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "../../imagenet"):
    """ return training dataloader
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
        ])
    train_dir = os.path.join(path_to_data, 'train')
    imagenet_training = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)

    indices = torch.randperm(len(imagenet_training))[:int(len(imagenet_training)* data_portion)]

    imagenet_training = torch.utils.data.Subset(imagenet_training, indices)

    imagenet_training_loader = get_multiclient_trainloader_list(imagenet_training, num_client, shuffle, num_workers, batch_size, noniid_ratio, 1000, hetero, hetero_string)

    return imagenet_training_loader


def get_imagenet_testloader(batch_size=16, num_workers=2, shuffle=False, path_to_data = "../../imagenet"):
    """ return training dataloader
    Returns: imagenet_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
    ])
    train_dir = os.path.join(path_to_data, 'val')
    imagenet_test = torchvision.datasets.ImageFolder(train_dir, transform=transform_test)
    imagenet_test_loader = DataLoader(imagenet_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    return imagenet_test_loader



def get_imagenet12_pairloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data/imagnet-12"):
    class imagenet12Pair(torchvision.datasets.ImageFolder):
        """tinyimagenet Dataset.
        """
        def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            path, _ = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                im_1 = self.transform(sample)
                im_2 = self.transform(sample)
            
            return im_1, im_2

    # tinyimagenet_training = datasets.ImageFolder('tiny-imagenet-200/train', transform=transform_train)
    # tinyimagenet_testing = datasets.ImageFolder('tiny-imagenet-200/val', transform=transform_test)
    if pairloader_option == "mocov1":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)])
    elif pairloader_option == "mocov2":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)])
    # data prepare
    train_data = imagenet12Pair(f'{path_to_data}/train', transform=train_transform)
    
    indices = torch.randperm(len(train_data))[:int(len(train_data)* data_portion)]

    train_data = torch.utils.data.Subset(train_data, indices)
    
    imagenet_training_loader = get_multiclient_trainloader_list(train_data, num_client, shuffle, num_workers, batch_size, noniid_ratio, 12, hetero, hetero_string)
    
    return imagenet_training_loader

def get_imagenet12_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, augmentation_option = False, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data/imagnet-12"):
    """ return training dataloader
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
        ])
    train_dir = os.path.join(path_to_data, 'train')
    imagenet_training = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)

    indices = torch.randperm(len(imagenet_training))[:int(len(imagenet_training)* data_portion)]

    imagenet_training = torch.utils.data.Subset(imagenet_training, indices)

    imagenet_training_loader = get_multiclient_trainloader_list(imagenet_training, num_client, shuffle, num_workers, batch_size, noniid_ratio, 12, hetero, hetero_string)

    return imagenet_training_loader


def get_imagenet12_testloader(batch_size=16, num_workers=2, shuffle=False, path_to_data = "./data/imagnet-12"):
    """ return training dataloader
    Returns: imagenet_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
    ])
    train_dir = os.path.join(path_to_data, 'val')
    imagenet_test = torchvision.datasets.ImageFolder(train_dir, transform=transform_test)
    imagenet_test_loader = DataLoader(imagenet_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    return imagenet_test_loader

def get_stl10_pairloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    class STL10Pair(torchvision.datasets.STL10):
        """CIFAR10 Dataset.
        """
        def __getitem__(self, index):
            img = self.data[index]
            img = Image.fromarray(img)
            if self.transform is not None:
                im_1 = self.transform(img)
                im_2 = self.transform(img)

            return im_1, im_2
    if pairloader_option == "mocov1":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    elif pairloader_option == "mocov2":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(96),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    # data prepare
    train_data = STL10Pair(root=path_to_data, split = 'train+unlabeled', transform=train_transform, download=True)
    
    indices = torch.randperm(len(train_data))[:int(len(train_data)* data_portion)]

    train_data = torch.utils.data.Subset(train_data, indices)
    
    stl10_training_loader = get_multiclient_trainloader_list(train_data, num_client, shuffle, num_workers, batch_size, noniid_ratio, 10, hetero, hetero_string)
    
    return stl10_training_loader

def get_stl10_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, augmentation_option = False, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(STL10_TRAIN_MEAN, STL10_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(STL10_TRAIN_MEAN, STL10_TRAIN_STD)
        ])
    #cifar00_training = CIFAR10Train(path, transform=transform_train)
    stl10_training = torchvision.datasets.STL10(root=path_to_data, split='train', download=True, transform=transform_train)

    indices = torch.randperm(len(stl10_training))[:int(len(stl10_training)* data_portion)]

    stl10_training = torch.utils.data.Subset(stl10_training, indices)

    stl10_training_loader = get_multiclient_trainloader_list(stl10_training, num_client, shuffle, num_workers, batch_size, noniid_ratio, 10, hetero, hetero_string)

    return stl10_training_loader

def get_stl10_testloader(batch_size=16, num_workers=2, shuffle=False, path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of stl10 test dataset
        std: std of stl10 test dataset
        path: path to stl10 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: stl10_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(STL10_TRAIN_MEAN, STL10_TRAIN_STD)
    ])

    stl10_test = torchvision.datasets.STL10(root=path_to_data, split='test', download=True, transform=transform_test)
    stl10_test_loader = DataLoader(
        stl10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return stl10_test_loader

def get_cifar10_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, augmentation_option = False, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
        ])
    #cifar00_training = CIFAR10Train(path, transform=transform_train)
    cifar10_training = torchvision.datasets.CIFAR10(root=path_to_data, train=True, download=True, transform=transform_train)

    indices = torch.randperm(len(cifar10_training))[:int(len(cifar10_training)* data_portion)]

    cifar10_training = torch.utils.data.Subset(cifar10_training, indices)

    cifar10_training_loader = get_multiclient_trainloader_list(cifar10_training, num_client, shuffle, num_workers, batch_size, noniid_ratio, 10, hetero, hetero_string)

    return cifar10_training_loader

def get_cifar10_testloader(batch_size=16, num_workers=2, shuffle=False, path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of cifar10 test dataset
        std: std of cifar10 test dataset
        path: path to cifar10 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar10_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
    ])

    cifar10_test = torchvision.datasets.CIFAR10(root=path_to_data, train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader

def get_cifar100_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, augmentation_option = False, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
    # print("num_client is", num_client)
    cifar100_training = torchvision.datasets.CIFAR100(root=path_to_data, train=True, download=True, transform=transform_train)

    indices = torch.randperm(len(cifar100_training))[:int(len(cifar100_training)* data_portion)]

    cifar100_training = torch.utils.data.Subset(cifar100_training, indices)

    cifar100_training_loader = get_multiclient_trainloader_list(cifar100_training, num_client, shuffle, num_workers, batch_size, noniid_ratio, 100, hetero, hetero_string)

    return cifar100_training_loader

def get_cifar100_testloader(batch_size=16, num_workers=2, shuffle=False, path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root=path_to_data, train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def get_SVHN_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, augmentation_option = False, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of SVHN training dataset
        std: std of SVHN training dataset
        path: path to SVHN training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD)
        ])
    #cifar00_training = SVHNTrain(path, transform=transform_train)
    SVHN_training = torchvision.datasets.SVHN(root=path_to_data, split='train', download=True, transform=transform_train)

    indices = torch.randperm(len(SVHN_training))[:int(len(SVHN_training)* data_portion)]

    SVHN_training = torch.utils.data.Subset(SVHN_training, indices)

    SVHN_training_loader = get_multiclient_trainloader_list(SVHN_training, num_client, shuffle, num_workers, batch_size, noniid_ratio, 10, hetero, hetero_string)

    return SVHN_training_loader

def get_SVHN_testloader(batch_size=16, num_workers=2, shuffle=False, path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of SVHN test dataset
        std: std of SVHN test dataset
        path: path to SVHN test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: SVHN_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD)
    ])
    SVHN_test = torchvision.datasets.SVHN(root=path_to_data, split='test', download=True, transform=transform_test)
    SVHN_test_loader = DataLoader(
        SVHN_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return SVHN_test_loader



def get_svhn_pairloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    class SVHNPair(torchvision.datasets.SVHN):
        """SVHN Dataset.
        """
        def __getitem__(self, index):
            img = self.data[index]
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            if self.transform is not None:
                im_1 = self.transform(img)
                im_2 = self.transform(img)

            return im_1, im_2
    if pairloader_option == "mocov1":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD)])
    elif pairloader_option == "mocov2":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD)])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD)])
    # data prepare
    
    train_data = SVHNPair(root=path_to_data, split='train', transform=train_transform, download=True)
    
    indices = torch.randperm(len(train_data))[:int(len(train_data)* data_portion)]

    train_data = torch.utils.data.Subset(train_data, indices)
    
    svhn_training_loader = get_multiclient_trainloader_list(train_data, num_client, shuffle, num_workers, batch_size, noniid_ratio, 10, hetero, hetero_string)
    
    return svhn_training_loader
