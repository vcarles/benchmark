"""NAttack attack module.

Module to create an interface for the NAttack adversarial attack. Based on the Attack base class.
"""

import torch

from src.ares.attack.base import Attack
from src.ares.attack.utils import norm_clamp


class NAttack(Attack):
    """Implements the NAttack adversarial attack. Works as a black-box iterative
    constraint-based method. Requires a classifier model and a differentiable loss function.

    Parameters :
    - Supported distance metric : 'l_2', 'l_inf'.
    - Supported goal : 't', 'tm', 'ut'.

    References : https://arxiv.org/abs/1905.00441.
    """

    def __init__(self, model, loss, goal, distance_metric, iteration_callback=False, device="cpu"):
        """Initializes the NAttack attack.

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
        self.nattack = {"details": {}}

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
        :param samples: The number of samples (or queries) to approximate the gradient, a 'int' type
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
            self.nattack["Queries_by_iteration"] = self.samples

        # Initialize the perturbation tensor to generate the output image.
        smp = torch.zeros_like(x)
        # Initialize the tensor of samples to optimize the mean of the normal distribution.
        smps = torch.zeros((self.samples,) + x_og.shape).to(self.device)

        # The mean and standard deviation values for the normal distribution.
        mean = torch.zeros_like(x)
        std = torch.ones_like(x) * self.sigma * self.sigma

        # Initialize the tensor the store the loss value for each sample.
        losses = torch.zeros((self.samples,)).to(self.device)

        # N attack generates the perturbation based on the normal function, computes the adversarial images and the loss
        # and update the value of the mean to generate the new perturbation.
        for it in range(1, self.iteration_steps+1):

            # For each sample, generate an adversarial example, compute the loss, and update the mean.
            for s in range(self.samples):
                smps[s] = torch.normal(mean, std)
                pert = norm_clamp(torch.zeros_like(x), x_og, torch.tanh(smps[s]), self.distance_metric, self.eps)

                with torch.no_grad():
                    logits, _ = self.model((x_og + pert).reshape((1,) + x.shape))
                losses[s] = self._loss(logits[0].to("cpu"), y, t)

            # Update the mean value based on the samples.
            loss_mean = losses.mean()
            loss_std = losses.std()

            upd = 0.
            for s in range(self.samples):
                upd += smps[s] * (losses[s] - loss_mean) / (loss_std + 0.01)

            mean = mean - self.learning_rate * upd / (self.samples * self.sigma)

            if self.iteration_callback:
                # Generate the perturbation based on the optimized mean value for every iteration.
                smp = torch.normal(mean, std)
                pert = norm_clamp(torch.zeros_like(x), x_og, torch.tanh(smp), self.distance_metric, self.eps)
                x_adv = x_og + pert
                x_adv = torch.clamp(x_adv, self.model.x_min, self.model.x_max)

                iter_key = "x_adv_{}".format(it)
                self.nattack[iter_key] = x_adv.to("cpu")

        # Generate the perturbation based on the optimized mean value.
        smp = torch.normal(mean, std)
        pert = norm_clamp(torch.zeros_like(x), x_og, torch.tanh(smp), self.distance_metric, self.eps)
        x_adv = x_og + pert
        x_adv = torch.clamp(x_adv, self.model.x_min, self.model.x_max)

        self.nattack["x_adv"] = x_adv.to("cpu")
        return self.nattack

    def _loss(self, logits, y, t):

        if self.goal in ("t", "tm"):
            target = t.argmax()
            correct_logit = logits[target]
            max_logit = logits.argmax()
            return max_logit - correct_logit
        else:
            truth = y.argmax()
            wrong_logit = logits[truth]
            logits[truth] = 0
            max_logit = logits.argmax()
            return wrong_logit - max_logit
