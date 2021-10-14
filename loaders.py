import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from IPython.display import Image
from matplotlib import cm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.utils import make_grid, save_image

from custom_dataset import (CustomDataLoader, PretrainImageFolder,
                            TransferImageFolder)


def create_loaders(configs, data_path, num_workers=0):
    """
    This function loads data and creates pytorch loaders.
    params:
        - configs (dict):
            Dictionary which contains necessary variables for loading data.
        - data_path (str):
            absolute path where train and val data is located.
        - num_workers (int): 
            workers for loading batches of data.
    returns:
        - loaders_dict (dict):
            dictionary of loaders for train and validation.
    """

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # load augmentation transforms.
    train_val_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # generate paths for train, valid and test.
    train_path = data_path + "/cifar10/train/"
    val_path = data_path + "/cifar10/test"

    # get the dataset.
    train_dataset = datasets.ImageFolder(
        root=train_path, transform=train_val_transform)
    valid_dataset = datasets.ImageFolder(
        root=val_path, transform=test_transform)

    # print # of classes for training and validation.
    print("Total number of classes for training: {}".format(
        len(train_dataset.class_to_idx)))
    print("Total number of classes for validation: {}".format(
        len(valid_dataset.class_to_idx)))

    # prepare data loaders (combine dataset and sampler).
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configs.batch_size,
                                               num_workers=num_workers, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=configs.batch_size,
                                               num_workers=num_workers, shuffle=True)
    # Create loaders dictionary
    loaders_dict = {'train': train_loader,
                    'valid': valid_loader}

    # print data for iteration lengths for train-val for the dataset.
    print('Number of iterations required to get through training data of length {}: {}'.format(
        len(train_dataset), len(train_loader)))

    print('Number of iterations required to get through validation data of length {}: {}'.format(
        len(valid_dataset), len(valid_loader)))

    return loaders_dict


def initialization_loaders(configs, data_path, num_workers=0):
    """
    This function loads data and creates pretraining and transfer-learning pytorch loaders.
    params:
        - configs (dict):
            Dictionary which contains necessary variables for loading data.
        - data_path (str):
            absolute path where train adn val data is located.
        - num_workers (int): 
            workers for loading batches of data.
    returns:
        - p_loaders_dict (dict):
            dictionary of pretraining data for train and validation
        - t_loaders_dict (dict):
            dictionary of transfer-learning data for train and validation 
        - classes_pretrain (list):
            list of classes for pretraining data.
        - classes_transfer (list):
            list of classes for transfer-learning data.
    """

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # load augmentation transforms.
    t_train_transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    p_train_transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # using test set as validation for fine-tuning
    t_val_path = data_path + "/test"
    data_path = data_path + "/train"

    # create imagefolder objects for augmented transfer and pretraining dataset.
    transfer_dataset = TransferImageFolder(
        root=data_path, transform=t_train_transform, subset_split=30000, class_split=10)
    pretrain_dataset = PretrainImageFolder(
        root=data_path, transform=p_train_transform, subset_split=20000, class_split=10)

    # # create imagefolder objects for non-augmented pretraining dataset.
    # pretrain_dataset_2 = PretrainImageFolder(
    #     root=data_path, transform=test_transform, subset_split=20000, class_split=10)

    # get transfer and pretraining classes.
    classes_transfer = transfer_dataset.t_classes
    classes_pretrain = pretrain_dataset.p_classes

    # create train and val datasets for transfer.
    t_train_dataset = transfer_dataset
    t_val_dataset = datasets.ImageFolder(
        root=t_val_path, transform=test_transform)

    # # create split to get augmented training data and non-augmented validation data for pretraining.
    # p_train_dataset, p_val_dataset = get_train_val_splits(
    #     pretrain_dataset, pretrain_dataset_2)
    p_train_dataset = pretrain_dataset

    # release memory.
    transfer_dataset, pretrain_dataset = None, None

    # dataloaders for transfer learning.
    t_train_loader = torch.utils.data.DataLoader(t_train_dataset, batch_size=configs.t_batch_size,
                                                 num_workers=num_workers, shuffle=True)

    t_val_loader = torch.utils.data.DataLoader(t_val_dataset, batch_size=configs.t_batch_size,
                                               num_workers=num_workers, shuffle=True)

    # dataloaders for pretraining.
    p_train_loader = torch.utils.data.DataLoader(p_train_dataset, batch_size=configs.p_batch_size,
                                                 num_workers=num_workers, shuffle=True)

    # Create loaders dictionary for transfer learning and pretraining.
    t_loaders_dict = {"train": t_train_loader, "valid": t_val_loader}
    p_loaders_dict = {"train": p_train_loader}

    # print data for iteration lengths for train-val for both transfer and pretrain.
    print("Transfer-Learning Data:")
    print("Number of iterations required to get through training data of length {}: {}".format(
        len(t_train_dataset), len(t_train_loader)))
    print("Number of iterations required to get through validation data of length {}: {}\n".format(
        len(t_val_dataset), len(t_val_loader)))

    print("Pretraining Data:")
    print("Number of iterations required to get through training data of length {}: {}".format(
        len(p_train_dataset), len(p_train_loader)))
    # print("Number of iterations required to get through validation data of length {}: {}\n".format(
    #     len(p_val_dataset), len(p_val_loader)))

    return p_loaders_dict, t_loaders_dict, classes_pretrain, classes_transfer
