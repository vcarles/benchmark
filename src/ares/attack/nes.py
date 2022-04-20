"""NES attack module.

Module to create an interface for the NES adversarial attack. Based on the Attack base class.
"""

import torch

from src.ares.attack.base import Attack
from src.ares.attack.utils import norm_clamp, tensor_unit


class NES(Attack):
    """Implements the Natural Evolution Strategies (NES) adversarial attack. Works as a black-box iterative
    constraint-based method. Requires a classifier model and a differentiable loss function.

    Parameters :
    - Supported distance metric : 'l_2', 'l_inf'.
    - Supported goal : 't', 'tm', 'ut'.

    References : https://arxiv.org/abs/1804.08598 and https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf.
    """

    def __init__(self, model, loss, goal, distance_metric, iteration_callback=False, device="cpu"):
        """Initializes the NES attack.

        :param model: The model to attack. An 'ares.model.Classifier' instance.
        :param loss: The loss function used to optimize the attack, a 'torch.nn' loss function.
        :param goal: Adversarial goal mode. Supported values are 't', 'tm', 'ut'.
        :param distance_metric: Distance metric used to compute distortion. Supported values are 'l_2', 'l_inf'.
        :param iteration_callback: False by default. If True, saves the adversarial images for each iteration. Must be
        set to True when running an 'IterationBenchmark' instance.
        :param device: The device to run the computation on, 'cpu' by default.
        """

        self.model = model
        self.loss = loss
        self.device = device

        self.goal = goal
        self.distance_metric = distance_metric

        # Magnitude initialized to 0., must be configured using the config() method.
        self.eps = 0.
        # The number of iteration_steps initialized to 0, must be configured using the 'config()' method.
        self.iteration_steps = 0
        # The number of samples initialized to 0, must be configured using the 'config()' method.
        self.samples = 0
        # The sampling variance initialized to 0, must be configured using the 'config()' method.
        self.sigma = 0
        # The learning rate initialized to 0., must be configured using the 'config()' method.
        self.learning_rate = 0.
        self.iteration_callback = iteration_callback

        # Initializes the output dictionary.
        self.nes = {"details": {}}

        # Inverts the gradient for targeted attacks.
        if goal in ("t", "tm"):
            self.grad_sign = -1
        elif goal == "ut":
            self.grad_sign = 1
        else:
            raise NotImplementedError

    def config(self, **kwargs):
        """(Re)Config the adversarial attack.

        :param magnitude: The max perturbation, a 'float' type number.
        :param iteration_steps: The number of iteration_steps to generate the perturbation, a 'int' type number.
        :param samples: The number of samples (half the number of queries) to approximate the gradient, a 'int' type
        number.
        :param sigma: The sampling variance for the gradient, a 'float' type number.
        :param learning_rate: The learning rate for each iteration step, a 'float' type number.
        """

        if "magnitude" in kwargs:
            self.eps = kwargs["magnitude"]
        if "iteration_steps" in kwargs:
            self.iteration_steps = kwargs["iteration_steps"]
        if "samples" in kwargs:
            self.samples = kwargs["samples"]
        if "sigma" in kwargs:
            self.sigma = kwargs["sigma"]
        if "learning_rate" in kwargs:
            self.learning_rate = kwargs["learning_rate"]

    def attack(self, x, y=None, t=None):
        """Generate an adversarial example for the input image.
        """

        # Initialize the tensor to store the adversarial image.
        x_adv = x.detach().clone().to(self.device)

        # Stores the input tensor for the maximum distortion value.
        x_og = x.detach().clone().to(self.device)

        # Initialize the perturbation value.
        pert = torch.zeros_like(x).to(self.device)

        if self.iteration_callback:
            self.nes["Queries_by_iteration"] = 2 * self.samples

        alpha = self.eps * 1.5 / self.iteration_steps

        for it in range(1, self.iteration_steps+1):

            grad = self._grad(x_adv, y, t)

            if self.distance_metric == "l_2":
                pert = tensor_unit(grad) * alpha * self.grad_sign
            elif self.distance_metric == "l_inf":
                pert = torch.sign(grad) * alpha * self.grad_sign
            else:
                raise NotImplementedError

            pert = norm_clamp(pert, x_og, x_adv, self.distance_metric, self.eps)

            x_adv = x_og + pert
            x_adv = torch.clamp(x_adv, self.model.x_min, self.model.x_max)

            if self.iteration_callback:
                iter_key = "x_adv_{}".format(it)
                self.nes[iter_key] = x_adv.detach().clone().to("cpu")

        self.nes["x_adv"] = x_adv.detach().to("cpu")
        return self.nes

    def _grad(self, x, y, t):
        """Estimate the gradient of the classifier model using the natural evolution strategy.
        """

        # Initialize the output tensor to store the estimated gradient.
        grad = torch.zeros_like(x).to(self.device)

        # Initialize the perturbation tensor.
        mu = torch.zeros_like(x)

        # The mean and standard deviation values for the normal distribution.
        mean = torch.zeros_like(x)
        std = torch.ones_like(x)

        for i in range(self.samples):
            mu = torch.normal(mean, std).to(self.device)

            with torch.no_grad():
                logits_plus, _ = self.model((x + self.sigma * mu).reshape((1,) + x.shape))
                logits_minus, _ = self.model((x - self.sigma * mu).reshape((1,) + x.shape))

            p_plus = self._loss(logits_plus[0], y, t)
            p_minus = self._loss(logits_minus[0], y, t)
            grad = grad + (p_plus - p_minus) * mu

        return grad / (2 * self.samples * self.sigma)

    def _loss(self, logits, y, t):

        if self.goal in ("t", "tm"):
            target = t.argmax()
            return logits[target]
        else:
            truth = y.argmax()
            logits[truth] = 0
            label = logits.argmax()
            return logits[label]
