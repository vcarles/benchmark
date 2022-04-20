"""FGSM attack module.

Module to create an interface for the FGSM adversarial attack. Based on the BatchAttack base class.
"""

import torch

from src.ares.attack.base import BatchAttack
from src.ares.attack.utils import maybe_to_tensor, tensor_unit


class FGSM(BatchAttack):
    """Implements the Fast Gradient Sign Method (FGSM) adversarial attack. Works as a white-box single-step
    constraint-based method. Requires a differentiable loss function and a classifier model.

    Parameters :
    - Supported distance metric : 'l_2', 'l_inf'.
    - Supported goal : 't', 'tm', 'ut'.

    Reference : https://arxiv.org/abs/1412.6572.
    """

    def __init__(self, model, batch_size, loss, goal, distance_metric, device="cpu"):
        """Initializes the FGSM attack.

        :param model: The model to attack. An 'ares.model.Classifier' instance.
        :param batch_size: Batch size used for benchmarking the attacks, as an 'int' number. CAUTION : when going
            through the dataset using the iterator, the effective batch size for the last batch may be smaller than the
            defined value.
        :param loss: The loss function used to optimize the attack, a 'torch.nn' loss function.
        :param goal: Adversarial goal mode. Supported values are 't', 'tm', 'ut'.
        :param distance_metric: Distance metric used to compute distortion. Supported values are 'l_2', 'l_inf'.
        :param device: The device to run the computation on, 'cpu' by default.
        """

        self.model = model
        self.loss = loss
        self.device = device
        self.batch_size = batch_size
        # The 'real' batch size when computing with batches of different sizes. In that case, batch_size is the maximum
        # size of the batches.
        self.eff_batch_size = batch_size

        self.goal = goal
        self.distance_metric = distance_metric

        # Magnitude initialized to 0., must be configured using config() method.
        # It could be either a 'float' type number, or a 'torch.Tensor' of 'float' numbers and of shape = (effective
        # batch size,). If it is a number, it will be converted to a tensor during the 'config()' method.
        self.eps = 0.

        # Initializes the output dictionary
        self.fgsm = {"details": {}}

        # Inverts the gradient for targeted attacks.
        if goal in ("t", "tm"):
            self.grad_sign = -1
        elif goal == "ut":
            self.grad_sign = 1
        else:
            raise NotImplementedError

    def config(self, **kwargs):
        """(Re)Config the adversarial attack.

        :param magnitude: The magnitude value of the perturbation, can be a 'float' type number or a 'torch.Tensor' of
        'float' numbers and of shape = (effective batch size,).
        """

        if "magnitude" in kwargs:
            self.eps = maybe_to_tensor(kwargs["magnitude"], self.batch_size)

    def batch_attack(self, xs, ys=None, ts=None):
        """Generate a batch of adversarial examples from the input batch.
        """

        # The effective batch size for the attack.
        self.eff_batch_size = xs.size()[0]

        # Initializes the tensor to store the batch of adversarial images.
        xs_adv = torch.zeros(((self.eff_batch_size,) + self.model.x_shape))

        for i in range(self.eff_batch_size):
            # Computes the perturbation value depending on the distance metric used. The perturbation is then used to
            # update the adversarial example.
            if self.distance_metric == "l_2":
                update = tensor_unit(xs.grad[i].to("cpu"))
            elif self.distance_metric == "l_inf":
                update = torch.sign(xs.grad[i].to("cpu"))
            else:
                raise NotImplementedError

            x_adv = xs[i].detach().to("cpu") + self.eps[i] * self.grad_sign * update
            x_adv = torch.clamp(x_adv, self.model.x_min, self.model.x_max)

            xs_adv[i] = x_adv

        self.fgsm["xs_adv"] = xs_adv
        return self.fgsm
