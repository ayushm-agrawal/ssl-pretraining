import argparse
import json
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import DataLoader, GenericDataset
from train import training
from utils import load_dataset, load_model, load_pretrain_dataset


def scratch_training(configs):

    # set seed for reproducibility.
    seed = configs.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # gpu training specific seed settings.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load data
    # configs.loaders = load_dataset(configs)
    configs.dataset = GenericDataset(
        'cifar10', 'train', random_sized_crop=False, num_imgs_per_cat=2000)
    configs.dataloader = DataLoader(
        configs.dataset, batch_size=configs.batch_size, unsupervised=True)

    # load model
    configs.model = load_model(configs)

    # loss
    configs.criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    configs.optimizer = optim.SGD(configs.model.parameters(), configs.lr,
                                  momentum=configs.momentum,
                                  weight_decay=configs.weight_decay)

    # train
    train_acc, test_acc, train_loss, test_loss = training(configs)

    # save arrays
    np.save(f"{configs.save_directory}{configs.exp_name}_train_acc", train_acc)
    np.save(f"{configs.save_directory}{configs.exp_name}_train_loss", train_loss)
    np.save(f"{configs.save_directory}{configs.exp_name}_test_acc", test_acc)
    np.save(f"{configs.save_directory}{configs.exp_name}_test_loss", test_loss)


def pretrain_training(configs):

    # set seed for reproducibility.
    seed = configs.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # gpu training specific seed settings.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    configs.loader = {}
    # load data
    # configs.loaders = load_dataset(configs)
    configs.dataset = GenericDataset(
        'cifar10', 'train', random_sized_crop=False, num_imgs_per_cat=2000)
    configs.loader['train'] = DataLoader(
        configs.dataset, batch_size=configs.batch_size, unsupervised=True)

    # load model
    configs.model = load_model(configs)

    # loss
    configs.criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    configs.optimizer = optim.SGD(configs.model.parameters(), configs.lr,
                                  momentum=configs.momentum,
                                  weight_decay=configs.weight_decay)

    # train
    train_acc, train_loss = training(configs)

    # save arrays
    np.save(f"{configs.save_directory}{configs.exp_name}_train_acc", train_acc)
    np.save(f"{configs.save_directory}{configs.exp_name}_train_loss", train_loss)

    # finetuning

    # update epochs
    configs.epochs = configs.fepochs

    # update exp name
    configs.exp_name = configs.exp_name[:-11] + "Finetune0.4"

    # update loader
    configs.loaders = t_loaders

    # update params for finetuning
    configs.finetune = True
    configs.num_classes = 100

    # update model
    configs.model = load_model(configs)

    # loss
    configs.criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    configs.optimizer = optim.SGD(configs.model.parameters(), configs.lr,
                                  momentum=configs.momentum,
                                  weight_decay=configs.weight_decay)

    # train
    train_acc, test_acc, train_loss, test_loss = training(configs)

    # save arrays
    np.save(f"{configs.save_directory}{configs.exp_name}_train_acc", train_acc)
    np.save(f"{configs.save_directory}{configs.exp_name}_train_loss", train_loss)
    np.save(f"{configs.save_directory}{configs.exp_name}_test_acc", test_acc)
    np.save(f"{configs.save_directory}{configs.exp_name}_test_loss", test_loss)


if __name__ == "__main__":

    # Parse Arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=int,
                        default=1, help='pretrain flag')
    parser.add_argument('--config', type=str,
                        default="./configs.json", help='configs_path')
    args = parser.parse_args()

    # load configs
    configs = SimpleNamespace(**json.load(open(args.config)))

    print(configs)

    if args.pretrain:
        pretrain_training(configs)
    else:
        scratch_training(configs)
