"""Classifier base module.

Module to create an interface for the classifier model. Contains metadata from the torch model.
"""

from abc import ABCMeta, abstractmethod

from torch import nn


class Classifier(nn.Module, metaclass=ABCMeta):
    """An abstract interface for the classifier model.
    """

    def __init__(self, n_class, x_min, x_max, x_shape, x_dtype, y_dtype, transform_image=None, transform_label=None,
                 transform_target=None):
        """Initialize the abstract classifier with metadata from the torch model.

        :param n_class: 'int' type number. The number of classes of the classifier.
        :param x_min: 'float' type number. The min value for the classifier's input.
        :param x_max: 'float' type number. The max value for the classifier's input.
        :param x_shape: a 'tuple' of 'int' numbers. The shape of the classifier's input.
        :param x_dtype: 'torch.dtype' instance. The data type of the classifier's input.
        :param y_dtype: 'torch.dtype' instance. The data type of the classifier's prediction results.
        :param transform_image: a 'torchvision.transforms' module to transform the dataset images for the model.
        :param transform_label: a 'torchvision.transforms' module to transform the dataset labels for the model.
        :param transform_target: a 'torchvision.transforms' module to transform the targets for the model.
        """

        super().__init__()
        self.n_class = n_class
        self.x_min, self.x_max = x_min, x_max
        self.x_shape = x_shape
        self.x_dtype, self.y_dtype = x_dtype, y_dtype

        self.transform_image = transform_image
        self.transform_label = transform_label
        self.transform_target = transform_target

        # Cache labels output.
        self._xs_labels_map = {}

    @abstractmethod
    def _forward(self, xs):
        """Take as input a 'torch.Tensor' and return the classifier's classification result as a 'torch.Tensor'.

        :param xs: a 'torch.Tensor' instance. Input for the classifier with shape 'self.x_shape' and with data type
            'self.x_dtype'. Data should be in the range of ['self.x_min', 'self.x_max']. May be a single image or a
            batch of images.
        :return: a 'torch.Tensor' instance with shape of (batch_size, 'self.n_class',) and with data type 'self.y_dtype'
        .Represents the classification result.
        """

        pass

    def forward(self, xs):
        """A wrapper for 'self._forward()' to cache labels output.
        """

        # when computing gradients for the input, do not save the labels in order to prevent memory overflow.
        if xs.requires_grad:
            labels = self._forward(xs)
        else:
            try:
                labels = self._xs_labels_map[xs]
            except KeyError:
                labels = self._forward(xs)

        return labels


class ClassifierWithLogits(nn.Module, metaclass=ABCMeta):
    """An abstract interface for the classifier model which also provides the logits output.
    """

    def __init__(self, n_class, x_min, x_max, x_shape, x_dtype, y_dtype, transform_image=None, transform_label=None,
                 transform_target=None):
        """Initialize the abstract classifier with metadata from the torch model.

        :param n_class: 'int' type number. The number of classes of the classifier.
        :param x_min: 'float' type number. The min value for the classifier's input.
        :param x_max: 'float' type number. The max value for the classifier's input.
        :param x_shape: a 'tuple' of 'int' numbers. The shape of the classifier's input.
        :param x_dtype: 'torch.dtype' instance. The data type of the classifier's input.
        :param y_dtype: 'torch.dtype' instance. The data type of the classifier's prediction results.
        :param transform_image: a 'torchvision.transforms' module to transform the dataset images for the model.
        :param transform_label: a 'torchvision.transforms' module to transform the dataset labels for the model.
        :param transform_target: a 'torchvision.transforms' module to transform the targets for the model.
        """

        super().__init__()
        self.n_class = n_class
        self.x_min, self.x_max = x_min, x_max
        self.x_shape = x_shape
        self.x_dtype, self.y_dtype = x_dtype, y_dtype

        self.transform_image = transform_image
        self.transform_label = transform_label
        self.transform_target = transform_target

        # Cache logits and labels output.
        self._xs_logits_labels_map = {}

    @abstractmethod
    def _forward(self, xs):
        """Take as input a 'torch.Tensor' and return the classifier's logits and classification results as a
        'torch.Tensor'.

        :param xs: a 'torch.Tensor' instance. Input for the classifier with shape 'self.x_shape' and with data type
            'self.x_dtype'. Data should be in the range of ['self.x_min', 'self.x_max']. May be a single image or a
            batch of images.
        :return: a tuple of 'torch.Tensor' instances, with shape of (batch_size, 'self.n_class',) and with data type
        'self.y_dtype'. Represents the logits and classification result.
        """

        pass

    def forward(self, xs):
        """A wrapper for 'self._forward()' to cache logits and labels output.
        """

        # when computing gradients for the input, do not save the logits and labels in order to prevent memory overflow.
        if xs.requires_grad:
            logits, labels = self._forward(xs)
        else:
            try:
                logits, labels = self._xs_logits_labels_map[xs]
            except KeyError:
                logits, labels = self._forward(xs)

        return logits, labels
