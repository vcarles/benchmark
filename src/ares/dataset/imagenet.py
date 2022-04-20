"""ImageNet ILSVRC 2012 validation dataset.

Module to download the dataset and to create a custom torch Dataset instance.

    Typical usage example:

    run this module directly to download the dataset
    dataset = ImagenetDataset(labels, images) to create the Dataset instance.
"""

import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

from src.ares.utils import get_res_path, download_res


_IMAGES_PATH = get_res_path("./imagenet/ILSVRC2012_img_val/")
_LABELS_PATH = get_res_path("./imagenet/val.txt")
_TARGET_PATH = get_res_path("./imagenet/target.txt")


class ImageNetDataset(Dataset):
    """Custom Dataset implementation for ImageNet ILSVRC 2012 validation dataset.

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
        self.img_dir = _IMAGES_PATH
        img_filenames, img_labels = _load_txt(_LABELS_PATH)
        self.img_filenames = img_filenames
        self.img_labels = img_labels
        if targets:
            _, img_targets = _load_txt(_TARGET_PATH)
        else:
            img_targets = None
        self.img_targets = img_targets
        self.img_transform = transform
        self.labels_transform = transform_labels
        self.targets_transform = transform_targets

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_filenames[idx])

        # In case images are RGB with transparency, discard the 4th channel.
        try:
            image = read_image(img_path, mode=ImageReadMode.RGB)
        except RuntimeError:
            image = read_image(img_path)[:3]

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


def _load_txt(path):
    """Load the images' filenames and labels from a .txt file and return a list of string and a tensor.
    """

    img_filenames = []
    img_labels = torch.zeros((50000,), dtype=torch.int64)
    i = 0

    with open(path, "r") as labels_file:
        for line in labels_file:
            filename, label = line.strip("\n").split(" ")
            img_filenames.append(filename)
            img_labels[i] = int(label)
            i += 1

    return img_filenames, img_labels


if __name__ == "__main__":
    if not os.path.exists(_LABELS_PATH):
        os.makedirs(os.path.dirname(_LABELS_PATH), exist_ok=True)
        download_res("http://ml.cs.tsinghua.edu.cn/~yinpeng/downloads/val.txt", _LABELS_PATH)

    if not os.path.exists(_TARGET_PATH):
        os.makedirs(os.path.dirname(_TARGET_PATH), exist_ok=True)
        download_res("http://ml.cs.tsinghua.edu.cn/~yinpeng/downloads/target.txt", _TARGET_PATH)

    if not os.path.exists(_IMAGES_PATH):
        print("ImageNet are not publicly available, please download 'ILSVRC2012_img_val.tar' from "
              "'http://www.image-net.org/download-images', "
              "and extract it to '{}'.".format(_IMAGES_PATH))
