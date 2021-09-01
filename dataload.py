import random

import numpy as np
import torch
import torch.utils.data as data
import torchnet as tnt
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from custom_dataset import PretrainImageFolder, TransferImageFolder

# Set the paths of the datasets here.
_CIFAR10_PRE_DATASET_DIR = './pre_cifar10'
_CIFAR10_FINE_DATASET_DIR = './fine_cifar10'
_CIFAR100_DATASET_DIR = './datasets/CIFAR100'
_STANFORDCARS_DATASET_DIR = './datasets/Stanford-Cars'


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def rotate_img(img, rot):
    if rot == 0:  # 0 degrees rotation
        return img
    elif rot == 90:  # 90 degrees rotation
        return np.flipud(np.transpose(img, (1, 0, 2))).copy()
    elif rot == 180:  # 90 degrees rotation
        return np.fliplr(np.flipud(img)).copy()
    elif rot == 270:  # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1, 0, 2)).copy()
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class GenericDataset(data.Dataset):
    def __init__(self, dataset_name, split):
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split

        if self.dataset_name == 'cifar10':
            self.mean_pix = [x/255.0 for x in [125.3, 123.0, 113.9]]
            self.std_pix = [x/255.0 for x in [63.0, 62.1, 66.7]]

            if self.random_sized_crop:
                raise ValueError(
                    'The random size crop option is not supported for the CIFAR dataset')

            transform = []
            if (split != 'test'):
                transform.append(transforms.RandomCrop(32, padding=4))
                transform.append(transforms.RandomHorizontalFlip())
            transform.append(lambda x: np.asarray(x))
            self.transform = transforms.Compose(transform)
            self.data = datasets.__dict__[self.dataset_name.upper()](
                _CIFAR10_PRE_DATASET_DIR, train=self.split == 'train',
                download=True, transform=self.transform)
        else:
            raise ValueError('Not recognized dataset {0}'.format(dname))


class PretrainDataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(
            dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers

        mean_pix = self.dataset.mean_pix
        std_pix = self.dataset.std_pix
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1, 2, 0).astype(np.uint8),
        ])

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        if self.unsupervised:
            # if in unsupervised mode define a loader function that given the
            # index of an image it returns the 4 rotated copies of the image
            # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
            # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img0, _ = self.dataset[idx]
                rotated_imgs = [
                    self.transform(img0),
                    self.transform(rotate_img(img0,  90)),
                    self.transform(rotate_img(img0, 180)),
                    self.transform(rotate_img(img0, 270))
                ]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                return torch.stack(rotated_imgs, dim=0), rotation_labels

            def _collate_fun(batch):
                batch = default_collate(batch)
                assert(len(batch) == 2)
                batch_size, rotations, channels, height, width = batch[0].size(
                )
                batch[0] = batch[0].view(
                    [batch_size*rotations, channels, height, width])
                batch[1] = batch[1].view([batch_size*rotations])
                return batch
        else:  # supervised mode
            # if in supervised mode define a loader function that given the
            # index of an image it returns the image and its categorical label
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label = self.dataset[idx]
                img = self.transform(img)
                return img, categorical_label
            _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
                                              load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=_collate_fun, num_workers=self.num_workers,
                                           shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    valid_size = 0.4
    shuffle = True

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    pretrain_dataset = datasets.CIFAR10(
        root=_CIFAR10_PRE_DATASET_DIR, train=True,
        download=True, transform=train_transform,
    )

    finetune_dataset = datasets.CIFAR10(
        root=_CIFAR10_FINE_DATASET_DIR, train=True,
        download=True, transform=train_transform,
    )

    num_train = len(pretrain_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(0)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    pretrain_loader = PretrainDataLoader(
        pretrain_dataset, batch_size=8, unsupervised=True)

    finetune_loader = torch.utils.data.DataLoader(
        finetune_dataset, batch_size=64, sampler=train_sampler,
        num_workers=0,
    )

    print(
        f"Finetune Dataset: {len(finetune_dataset)}\n Pretrain Dataset: {len(pretrain_dataset)}")

    for b in pretrain_loader(0):
        data, label = b
        break

    inv_transform = pretrain_loader.inv_transform
    for i in range(data.size(0)):
        plt.subplot(data.size(0)/4, 4, i+1)
        fig = plt.imshow(inv_transform(data[i]))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.show()
