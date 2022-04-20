"""CIFAR-10 test dataset.

Module to download the dataset and to create a custom torch Dataset instance.

    Typical usage example:

    run this module directly to download the dataset
    dataset = Cifar10Dataset(labels, images) to create the Dataset instance.
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from src.ares.utils import get_res_path, download_res


_DATA_PATH = get_res_path("./cifar10/cifar-10-batches-py/")
_TARGET_PATH = get_res_path("./cifar10/target.npy")


class Cifar10Dataset(Dataset):
    """Custom Dataset implementation for CIFAR-10 test dataset.

    An instance of torch.utils.data.Dataset, to wrap using torch's DataLoader instance.

    :param targets: False by default. If True, loads the targets file.
    :param: transform: None by default. If provided, applies the transformation to the images (must be a
    torchvision.transforms instance).
    :param: transform_labels: None by default. If provided, applies the transformation to the labels (must be a
    torchvision.transforms instance).
    :param: transform_targets: None by default. If provided, applies the transformation to the targets (must be a
    torchvision.transforms instance).
    """

    def __init__(self, targets=False, transform=None, transform_labels=None, transform_targets=None):
        imgs, img_labels = _load_data(_DATA_PATH)
        self.imgs = imgs
        self.img_labels = img_labels
        self.img_targets = _load_npy(_TARGET_PATH) if targets else None
        self.img_transform = transform
        self.labels_transform = transform_labels
        self.targets_transform = transform_targets

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]

        if self.img_transform:
            image = self.img_transform(image)
        if self.labels_transform:
            label = self.labels_transform(label)

        image_item = {"image": image, "label": label}

        if self.img_targets is not None:
            target = self.img_targets[idx]
            if self.targets_transform:
                target = self.targets_transform(target)

            image_item["target"] = target

        return image_item

def _load_data(path):
    """Load the images and the labels as tensors.
    """

    data_path = os.path.join(path, "test_batch")

    with open(data_path, "rb") as data_file:
        img_data = pickle.load(data_file, encoding="bytes")

    imgs = _to_imgs(img_data[b"data"])
    labels = _to_labels(img_data[b"labels"])

    return imgs, labels


def _to_imgs(img_data):
    """Converts data from the Cifar-10 format to a tensor of images.
    """

    imgs = torch.ones((10000, 3, 32, 32), dtype=torch.int64)

    for n in range(10000):
        for c in range(3):
            temp_arr = np.array(img_data[n, c*1024:(c+1)*1024], dtype=np.intc).reshape((32, 32))
            imgs[n, c] = torch.from_numpy(temp_arr)

    return imgs


def _to_labels(labels_data):
    """Converts data from the Cifar-10 format to a tensor of labels.
    """

    labels = torch.ones((10000,), dtype=torch.int64)

    for n in range(10000):
        labels[n] = int(labels_data[n])

    return labels


def _load_npy(path):
    """Load the images' targets from a .npy file and return a tensor.
    """

    targets = np.load(path)
    return torch.from_numpy(targets)


if __name__ == "__main__":
    if not os.path.exists(_TARGET_PATH):
        os.makedirs(os.path.dirname(_TARGET_PATH), exist_ok=True)
        download_res("https://ml.cs.tsinghua.edu.cn/~qian/ares/target.npy", _TARGET_PATH)

    if not os.path.exists(_DATA_PATH):
        print("Please download 'cifar-10-python.tar.gz' from "
              "'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', "
              "and extract it to '{}'.".format(_DATA_PATH))
