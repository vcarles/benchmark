"""Attack base module.

Module to create an interface for the adversarial attack. Supports single and batch attacks.
"""

from abc import ABCMeta, abstractmethod


class BatchAttack(metaclass=ABCMeta):
    """An abstract interface for the adversarial attack methods which support attacking a batch of input.

    All the model or method should be constructed in the '__init__()' method. The 'config()' method should only tweak
    the parameters of the method and not create additional nodes on the model or method. Else it could lead to large
    memory leakage during benchmarks.
    """

    @abstractmethod
    def config(self, **kwargs):
        """(Re)Config the adversarial attack.

        :param kwargs: Change the configuration of the attack method. Should support partial configuration, so that for
        each configuration option, only the newest values are kept.
        """

        pass

    @abstractmethod
    def batch_attack(self, xs, ys=None, ls=None):
        """Generate a batch of adversarial examples from the input batch.

        :param xs: A batch of images, each image tensor must activate 'requires_grad = True'.
        :param ys: The ground truth labels of the original batch, could be 'None'.
        :param ls: The targeted labels for the original batch, could be 'None'.
        :return: A batch of adversarial examples. Detailed information may be returned by storing in its 'details'
        attribute.
        """

        pass


class Attack(metaclass=ABCMeta):
    """An abstract interface for the adversarial attack methods which only support attacking one input at a time.

    All the model or method should be constructed in the '__init__()' method. The 'config()' method should only tweak
    the parameters of the method and not create additional nodes on the model or method. Else it could lead to large
    memory leakage during benchmarks.
    """

    @abstractmethod
    def config(self, **kwargs):
        """(Re)Config the adversarial attack.

        :param kwargs: Change the configuration of the attack method. Should support partial configuration, so that for
        each configuration option, only the newest values are kept.
        """

        pass

    @abstractmethod
    def attack(self, x, y=None, l=None):
        """Generate an adversarial example from the input image.

        :param x: The input image, the image tensor must activate 'requires_grad = True'.
        :param y: The ground truth label of the original image, could be 'None'.
        :param l: The targeted label for the original image, could be 'None'.
        :return: An adversarial example. Detailed information may be returned by storing in its 'details'
        attribute.
        """

        pass
