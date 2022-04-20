"""This file provides a wrapper class for WideResNet model for the Cifar-10 Dataset.

(https://github.com/yaircarmon/semisup-adv)
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Lambda

from src.ares.model import ClassifierWithLogits
from src.ares.utils import get_res_path


_WEIGHTS_PATH = get_res_path("./cifar10/cifar10_rst_adv.pt.ckpt")

# Creating the 'torchvision.transforms' modules to transform the dataset images, labels and targets for the model. The
# modules are used to initialize the 'ClassifierWithLogits' instance and then accessed when loading the dataset.
transform_label = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter(0, y, src=torch.Tensor([1.0])))
transform_target = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter(0, y, src=torch.Tensor([1.0])))
transform_image = Lambda(lambda t: t / 255)


class RST(ClassifierWithLogits):
    def __init__(self):
        super().__init__(n_class=10, x_min=0., x_max=1., x_shape=(3, 32, 32,), x_dtype=torch.float, y_dtype=torch.float,
                         transform_image=transform_image, transform_label=transform_label,
                         transform_target=transform_target)
        self.model = WideResNet(depth=28, num_classes=10, widen_factor=10)
        self._load()
        # self.model = torch.nn.DataParallel(self.model).cuda()

    def _forward(self, xs):
        logits = self.model(xs)

        proba = nn.Softmax(dim=1)(logits)
        labels = proba.argmax(1).to("cpu")

        size = labels.size()[0]
        labels.resize_(size, 1, )
        labels = torch.zeros((size, 10,), dtype=torch.float).scatter(1, labels, src=torch.ones((size, 1,)))

        return logits, labels

    def _load(self):
        weights = torch.load(_WEIGHTS_PATH)
        self.model.load_state_dict(weights["state_dict"])


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, return_prelogit=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if return_prelogit:
            return self.fc(out), out
        else:
            return self.fc(out)


if __name__ == "__main__":
    if not os.path.exists(_WEIGHTS_PATH):
        os.makedirs(os.path.dirname(_WEIGHTS_PATH), exist_ok=True)
    url = "https://drive.google.com/file/d/1S3in_jVYJ-YBe5-4D0N70R4bN82kP5U2/view"
    print("Please download the model weights at :'{}' and save them under :'{}'.".format(url, _WEIGHTS_PATH))
