"""BIM attack module.

Module to create an interface for the BIM adversarial attack. Based on the BatchAttack base class.
"""

import torch

from src.ares.attack.base import BatchAttack
from src.ares.attack.utils import maybe_to_tensor, norm_clamp, tensor_unit


class BIM(BatchAttack):
    """Implements the Basic Iterative Method (BIM) adversarial attack. Works as a white-box iterative constraint-based
    method. Requires a differentiable loss function and a classifier model.

    Parameters :
    - Supported distance metric : 'l_2', 'l_inf'.
    - Supported goal : 't', 'tm', 'ut'.

    Reference : https://arxiv.org/abs/1607.02533.
    """

    def __init__(self, model, batch_size, loss, goal, distance_metric, iteration_callback=False, device="cpu"):
        """Initializes the BIM attack.

        :param model: The model to attack. An 'ares.model.Classifier' instance.
        :param batch_size: Batch size used for benchmarking the attacks, as an 'int' number. CAUTION : when going
            through the dataset using the iterator, the effective batch size for the last batch may be smaller than the
            defined value.
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
        self.batch_size = batch_size
        # The 'real' batch size when computing with batches of different size, in that case, batch size is the maximum
        # size of the batches.
        self.eff_batch_size = batch_size

        self.goal = goal
        self.distance_metric = distance_metric

        # Magnitude initialized to 0., must be configured using config() method.
        # It could be either a 'float' type number, or a 'torch.Tensor' of 'float' numbers and of shape = (effective
        # batch size,). If it is a number, it will be converted to a tensor during the 'config()' method.
        self.eps = 0.
        # Step size initialized to 0., must be configured using config() method.
        # It could be either a 'float' type number, or a 'torch.Tensor' of 'float' numbers and of shape = (effective
        # batch size,). If it is a number, it will be converted to a tensor during the 'config()' method.
        self.alpha = 0.
        # The number of iterations initialized to 0, must be configured using the 'config()' method.
        self.iteration = 0
        self.iteration_callback = iteration_callback

        # Initializes the output dictionary.
        self.bim = {"details": {}}

        # Inverts the gradient for targeted attacks.
        if goal in ("t", "tm"):
            self.grad_sign = -1
        elif goal == "ut":
            self.grad_sign = 1
        else:
            raise NotImplementedError

    def config(self, **kwargs):
        """(Re)Config the adversarial attack.

        :param magnitude: The max perturbation, can be a 'float' type number or a 'torch.Tensor' of 'float' numbers and
        of shape = (effective batch size,).
        :param alpha: The step size for each iteration, can be a 'float' type number or a 'torch.Tensor' of 'float'
        numbers and of shape = (effective batch size,).
        :param iteration: The number of iterations for the attack, a 'int' type number.
        """

        if "magnitude" in kwargs:
            self.eps = maybe_to_tensor(kwargs["magnitude"], self.batch_size)
        if "alpha" in kwargs:
            self.alpha = maybe_to_tensor(kwargs["alpha"], self.batch_size)
        if "iteration" in kwargs:
            self.iteration = kwargs["iteration"]

    def batch_attack(self, xs, ys=None, ts=None):
        """Generate a batch of adversarial examples from the input batch.
        """

        # The effective batch_size() for the attack.
        self.eff_batch_size = xs.size()[0]

        # Initializes the tensor to store the batch of adversarial images.
        xs_adv = torch.zeros(((self.eff_batch_size,) + self.model.x_shape))

        # Stores the input tensor for the maximum distortion value.
        xs_og = xs.detach().clone().to("cpu")

        for it in range(1, self.iteration+1):
            # For the first iteration, the adversarial image is crafted based on the gradient with respect to the
            # input. Else it is the gradient with respect to the previously crafted adversarial example.
            if it == 1:
                grad = xs.grad.to("cpu")
                xs_adv = xs_og.clone()
            else:
                grad = self._grad(xs_adv.to(self.device), ys, ts)

            for i in range(self.eff_batch_size):
                # Computes the perturbation value depending on the distance metric used. The perturbation is then used
                # to update the adversarial example.
                if self.distance_metric == "l_2":
                    update = tensor_unit(grad[i]) * self.alpha[i] * self.grad_sign
                elif self.distance_metric == "l_inf":
                    update = torch.sign(grad[i]) * self.alpha[i] * self.grad_sign
                else:
                    raise NotImplementedError

                # The adversarial image data values must remain below the max distortion value (=magnitude or self.eps)
                # and inside the x_min and x_max values of the model.
                update = norm_clamp(update, xs_og[i], xs_adv[i], self.distance_metric, self.eps[i])

                x_adv = xs_og[i] + update
                x_adv = torch.clamp(x_adv, self.model.x_min, self.model.x_max)

                xs_adv[i] = x_adv

            if self.iteration_callback:
                iter_key = "xs_adv_{}".format(it)
                self.bim[iter_key] = xs_adv.clone()

        self.bim["xs_adv"] = xs_adv
        return self.bim

    def _grad(self, xs, ys, ts):
        """Calculates the gradient with respect to the input.
        """

        # compute gradient for each image based on the parameters
        xs.requires_grad_()
        logits, _ = self.model(xs)

        # the gradient of the loss is computed with the ground truth for untargeted attacks and with the targets for
        # targeted attacks.
        if self.goal == "ut":
            loss = self.loss(logits, ys)
        else:
            loss = self.loss(logits, ts)

        self.model.zero_grad()
        loss.backward()

        # empty memory on the cpu and on the device used for computations
        grad = xs.grad.to("cpu")

        xs.detach_()
        logits.detach_()
        del xs, logits

        return grad
