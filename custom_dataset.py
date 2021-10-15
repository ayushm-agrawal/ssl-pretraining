import os
import os.path
import random

import numpy as np
import torch
# import torchnet as tnt
from matplotlib import cm
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.vision import VisionDataset

#### These were in the original Pytorch Docs and were not changed ####


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

######################################################################


def make_dataset(directory, classes, class_to_idx,
                 extensions=None, is_valid_file=None, subset_split=36000, class_split=12, dist="normal"):
    """
    This function creates custom pretraining and transfer datsets.
    params:
        - directory (string):
            path to data directory.
        - classes (list):
            list of classes for the whole dataset.
        - class_to_idx (dict):
            dictionary mapping classes to their idxs
        - extensions:
            file extensions.
        - is_valid_file:
            if the file is valid.
        - subset_split (int):
            Number of examples for pretraining.
        - class_split (int):
            Number of classes for pretraining.
        - dist (str):
            sampling distribution for the dataset.
    returns:
        - dataset (tuple):
            returns a tuple of variables which make up the pretrain and transfer datsets
            transfer:
                transfer_data - list of (img, label) tuples.
                transfer_classes - list of classes.
                t_class_to_index - dictionary mapping classes to idxs
            pretrain:
                pretrain_data - list of (img, label) tuples.
                pretrain_classes - list of classes.
                p_class_to_index - dictionary mapping classes to idxs
    """

    instances = []
    # counts all files under directory
    directory = os.path.expanduser(directory)
    # checks for extensions
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None

    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)

    # creates instances for the whole dataset
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

    ## START CREATING PRETRAINING AND TRANSFER DATASETS ##

    # Create pretrain classes by taking a subset

    # create a counter dictionary for keeping track of
    # examples per class
    dataset_counts = {i[1]: 0 for i in instances}

    # count #of examples per class
    for img, label in instances:
        dataset_counts[label] += 1

    # Create pretraining and transfer split based on distribution param.
    if dist == "normal":

        # dictionary to store number of images per class for normal distribution.
        dataset_percentage = {}
        total_counts = sum(dataset_counts.values())

        # compute the contribution of each class to the total dataset.
        # and then scale it to the pretraining size.
        for k, v in dataset_counts.items():
            dataset_percentage[k] = int(0.4*v)
            # int(np.floor((v/total_counts)*subset_split))

        # pick the pretraining classes
        pretraining_classes = [i[0] for i in dataset_percentage.items()]

        # create dict for counter for pretraining classes.
        pretrain_counts = {i: 0 for i in pretraining_classes}

        # lists to store transfer and pretraining tuples
        transfer_data = []
        pretrain_data = []
        t_classes = []

        # go through all instances
        # append to transfer or pretrain
        # based on certain conditions.
        for idx, (img, label) in enumerate(instances):
            # if label is in the pretrain classes and the number of examples for this
            # label in the pretraining data is < pretraining limit for class.
            # then add tuple to pretraining dataset
            if label in pretraining_classes and pretrain_counts[label] < dataset_percentage[label]:
                # index it to mapped ids
                pretrain_data.append((img, pretraining_classes.index(label)))
                # increment label count in order to keep track
                pretrain_counts[label] += 1
            # otherwise it goes to transfer
            else:
                transfer_data.append((img, label))
                t_classes.append(label)

    else:
        # sort based on counts for each class
        # slice sorted count list to get classes with most examples
        sorted_dataset_counts = sorted(
            dataset_counts.items(), key=lambda item: item[1])[-class_split:]

        # pick the pretraining classes based on class_split (we pick the classes with most examples first)
        pretraining_classes = [i[0] for i in sorted_dataset_counts]

        # create dict for counter for pretraining classes (each class should have subset_split/class_split) examples.
        pretrain_counts = {i: 0 for i in pretraining_classes}

        # lists to store transfer and pretraining tuples
        transfer_data = []
        pretrain_data = []
        t_classes = []

        # go through all instances
        # append to transfer or pretrain
        # based on certain conditions.
        for idx, (img, label) in enumerate(instances):
            # if label is in the pretrain classes and the number of examples for this
            # label in the pretraining data is < subset_split/class_split
            # then add tuple to pretraining dataset
            if label in pretraining_classes and pretrain_counts[label] < (subset_split/class_split):
                # index it to mapped ids
                pretrain_data.append((img, pretraining_classes.index(label)))
                # increment label count in order to keep track
                pretrain_counts[label] += 1
            # otherwise it goes to transfer
            else:
                transfer_data.append((img, label))
                t_classes.append(label)

    ### create classes and class_to_idx for transfer and pretrain ###

    # get unique transfer learning classes
    t_classes = np.unique(t_classes)

    # create classes for transfer and pretraining
    transfer_classes, pretrain_classes = [classes[i] for i in t_classes], [
        classes[i] for i in pretraining_classes]

    # create class_to_idx dictionaries for transfer and pretrain
    t_class_to_index = {i: class_to_idx[i] for i in transfer_classes}
    p_class_to_index = {i: class_to_idx[i] for i in pretrain_classes}
    ######################################################

    return (transfer_data, transfer_classes, t_class_to_index, pretrain_data, pretrain_classes, p_class_to_index)


class DatasetFolder(VisionDataset):
    """
    A custom data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    params:
        - root (string): 
            Root directory path.
        - loader (callable): 
            A function to load a sample given its path.
        - extensions (tuple[string]):
            A list of allowed extensions both extensions and is_valid_file 
            should not be passed.
        - transform (callable, optional): 
            A function/transform that takes in a sample and returns a transformed 
            version. E.g, ``transforms.RandomCrop`` for images.
        - target_transform (callable, optional): 
            A function/transform that takes in the target and transforms it.
        - is_valid_file (callable, optional): 
            A function that takes path of a file and check if the file is a valid 
            file (used to check of corrupt files) both extensions and is_valid_file 
            should not be passed.
        - subset_split (int):
            Number of examples for pretraining.
        - class_split (int):
            Number of classes for pretraining.
        - dist (str):
            sampling distribution for the dataset.
    attributes:
        - t_classes (list): 
            List of the class names for transfer sorted alphabetically.
        - p_classes (list): 
            List of the class names for pretraining sorted alphabetically.
        - t_class_to_idx (dict): 
            Dict with items (class_name, class_index) for transfer.
        - p_class_to_idx (dict): 
            Dict with items (class_name, class_index) for pretrain.
        - t_samples (list): 
            List of (sample path, class_index) tuples for transfer.
        - p_samples (list): 
            List of (sample path, class_index) tuples for pretrain.
        - t_targets (list): 
            The class_index value for each image in the transfer dataset
        - p_targets (list): 
            The class_index value for each image in the pretrain dataset
    """

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None,  subset_split=36000, class_split=12, dist="normal"):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)

        # finds classes for original dataset
        classes, class_to_idx = self._find_classes(self.root)

        # creates datasets
        t_samples, t_classes, t_class_to_idx, p_samples, p_classes, p_class_to_idx = make_dataset(self.root,
                                                                                                  classes,
                                                                                                  class_to_idx,
                                                                                                  extensions,
                                                                                                  is_valid_file,
                                                                                                  subset_split=subset_split,
                                                                                                  class_split=class_split,
                                                                                                  dist=dist)

        if len(p_samples) == 0 and len(t_samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.p_classes = p_classes
        self.p_class_to_idx = p_class_to_idx
        self.p_samples = p_samples
        self.p_targets = [s[1] for s in p_samples]

        self.t_classes = t_classes
        self.t_class_to_idx = t_class_to_idx
        self.t_samples = t_samples
        self.t_targets = [s[1] for s in t_samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class TransferImageFolder(DatasetFolder):
    """
    A custom data loader for the transfer-learning dataset 
    where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    params:
        - root (string): 
            Root directory path.
        - transform (callable, optional): 
            A function/transform that  takes in an PIL image and returns a 
            transformed version. E.g, ``transforms.RandomCrop``
        - target_transform (callable, optional): 
            A function/transform that takes in the target and transforms it.
        - loader (callable, optional): 
            A function to load an image given its path.
        - is_valid_file (callable, optional): 
            A function that takes path of an Image file and check if the file 
            is a valid file (used to check of corrupt files)
    attributes:
        - classes (list): 
            List of the class names sorted alphabetically.
        - class_to_idx (dict): 
            Dict with items (class_name, class_index).
        - imgs (list): 
            List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None,  subset_split=30000, class_split=10, dist="normal"):
        # calls DatasetFolder
        super(TransferImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                  transform=transform,
                                                  target_transform=target_transform,
                                                  is_valid_file=is_valid_file,
                                                  subset_split=subset_split, class_split=class_split, dist=dist)
        # retrive transfer-learning images
        self.imgs = self.t_samples

    def __len__(self):
        # returns length of transfer-learning examples
        return len(self.t_samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.t_samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class PretrainImageFolder(DatasetFolder):
    """
    A custom data loader for the pretraining dataset 
    where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    params:
        - root (string): 
            Root directory path.
        - transform (callable, optional): 
            A function/transform that  takes in an PIL image and returns a 
            transformed version. E.g, ``transforms.RandomCrop``
        - target_transform (callable, optional): 
            A function/transform that takes in the target and transforms it.
        - loader (callable, optional): 
            A function to load an image given its path.
        - is_valid_file (callable, optional): 
            A function that takes path of an Image file and check if the file 
            is a valid file (used to check of corrupt files)
    attributes:
        - classes (list): 
            List of the class names sorted alphabetically.
        - class_to_idx (dict): 
            Dict with items (class_name, class_index).
        - imgs (list): 
            List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None,  subset_split=20000, class_split=10, dist="normal"):
        # calls DatasetFolder
        super(PretrainImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                  transform=transform,
                                                  target_transform=target_transform,
                                                  is_valid_file=is_valid_file,
                                                  subset_split=subset_split, class_split=class_split, dist=dist)
        # retrive pretraining images
        self.imgs = self.p_samples

    def __len__(self):
        # returns length of pretraining examples
        return len(self.p_samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.p_samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            # sample = self.transform(sample)
            sample = [self.transform(sample),
                      self.transform(sample.rotate(90, expand=True)),
                      self.transform(sample.rotate(180, expand=True)),
                      self.transform(sample.rotate(270, expand=True))]
        # if self.target_transform is not None:
        #     # target = self.target_transform(target)
        target = torch.LongTensor([0, 1, 2, 3])

        return torch.stack(sample, dim=0), target


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


# class CustomDataLoader(object):
#     def __init__(self,
#                  dataset,
#                  transforms,
#                  batch_size=1,
#                  unsupervised=True,
#                  epoch_size=None,
#                  num_workers=0,

#                  shuffle=True):
#         self.dataset = dataset
#         self.shuffle = shuffle
#         self.epoch_size = epoch_size if epoch_size is not None else len(
#             dataset)
#         self.batch_size = batch_size
#         self.unsupervised = unsupervised
#         self.num_workers = num_workers

#         self.transform = transforms

#     def get_iterator(self, epoch=0):
#         rand_seed = epoch * self.epoch_size
#         random.seed(rand_seed)
#         if self.unsupervised:
#             # if in unsupervised mode define a loader function that given the
#             # index of an image it returns the 4 rotated copies of the image
#             # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
#             # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
#             def _load_function(idx):
#                 idx = idx % len(self.dataset)
#                 img0, _ = self.dataset[idx]
#                 rotated_imgs = [
#                     self.transform(img0),
#                     self.transform(rotate_img(img0,  90)),
#                     self.transform(rotate_img(img0, 180)),
#                     self.transform(rotate_img(img0, 270))
#                 ]
#                 rotation_labels = torch.LongTensor([0, 1, 2, 3])
#                 return torch.stack(rotated_imgs, dim=0), rotation_labels

#             def _collate_fun(batch):
#                 batch = default_collate(batch)
#                 assert(len(batch) == 2)
#                 batch_size, rotations, channels, height, width = batch[0].size(
#                 )
#                 batch[0] = batch[0].view(
#                     [batch_size*rotations, channels, height, width])
#                 batch[1] = batch[1].view([batch_size*rotations])
#                 return batch
#         else:  # supervised mode
#             # if in supervised mode define a loader function that given the
#             # index of an image it returns the image and its categorical label
#             def _load_function(idx):
#                 idx = idx % len(self.dataset)
#                 img, categorical_label = self.dataset[idx]
#                 img = self.transform(img)
#                 return img, categorical_label
#             _collate_fun = default_collate

#         tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
#                                               load=_load_function)
#         data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
#                                            collate_fn=_collate_fun, num_workers=self.num_workers,
#                                            shuffle=self.shuffle)
#         return data_loader

#     def __call__(self, epoch=0):
#         return self.get_iterator(epoch)

#     def __len__(self):
#         return self.epoch_size / self.batch_size
