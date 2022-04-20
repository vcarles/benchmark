"""DeepFool attack module.

Module to create an interface for the DeepFool adversarial attack. Based on the BatchAttack base class.
"""

import torch

from src.ares.attack.base import BatchAttack


class DeepFool(BatchAttack):
    """Implements the DeepFool adversarial attack. Works as a white-box iterative optimization method. Requires a
    differentiable loss function and a classifier model.

    Parameters :
    - Supported distance metric : 'l_2', 'l_inf'.
    - Supported goal : 'ut'.

    Reference : https://arxiv.org/abs/1511.04599.
    """

    def __init__(self, model, batch_size, loss, goal, distance_metric, iteration_callback=False, device="cpu"):
        """Initializes the DeepFool attack.

        :param model: The model to attack. An 'ares.model.Classifier' instance.
        :param batch_size: Batch size used for benchmarking the attacks, as an 'int' number. CAUTION : when going
            through the dataset using the iterator, the effective batch size for the last batch may be smaller than the
            defined value.
        :param loss: The loss function used to optimize the attack, a 'torch.nn' loss function.
        :param goal: Adversarial goal mode. Supported value is 'ut'.
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

        if goal in ("t", "tm",):
            raise NotImplementedError
        self.goal = goal
        self.distance_metric = distance_metric

        # The overshoot rate initialized to 0., must be configured using the 'config()' method.
        self.overshoot = 0.
        # The number of iterations initialized to 0, must be configured using the 'config()' method.
        self.iteration = 0
        self.iteration_callback = iteration_callback

        # Initializes the output dictionary.
        self.deepfool = {"details": {}}

    def config(self, **kwargs):
        """(Re)Config the adversarial attack.

        :param overshoot: The overshoot rate to generate the perturbations, a 'float' type number.
        :param iteration: The number of iterations for the attack, a 'int' type number.
        """

        if "overshoot" in kwargs:
            self.overshoot = kwargs["overshoot"]
        if "iteration" in kwargs:
            self.iteration = kwargs["iteration"]

    def batch_attack(self, xs, ys=None, ts=None):
        """Generate a batch of adversarial examples from the input batch.
        """

        # The effective batch_size() for the attack.
        self.eff_batch_size = xs.size()[0]

        # Initializes the tensor to store the batch of adversarial images.
        xs_adv = torch.zeros(((self.eff_batch_size,) + self.model.x_shape))

        # Initializes the tensor to store the iteration tracking of adversarial images.
        xs_adv_iter = torch.zeros(((self.iteration, self.eff_batch_size,) + self.model.x_shape))

        for i in range(self.eff_batch_size):
            x_adv, adv_iter = self._deepfool(xs.detach()[i], ys[i])
            xs_adv[i] = x_adv

            if self.iteration_callback:
                iter_size = len(adv_iter)
                for it in range(self.iteration):
                    if it >= iter_size:
                        xs_adv_iter[it, i, :] = adv_iter[-1]
                    else:
                        xs_adv_iter[it, i, :] = adv_iter[it]

        if self.iteration_callback:
            for it in range(self.iteration):
                iter_key = "xs_adv_{}".format(it+1)
                self.deepfool[iter_key] = xs_adv_iter[it].clone()

        self.deepfool["xs_adv"] = xs_adv
        return self.deepfool

    def _deepfool(self, x, y):
        """Generate an adversarial example using the DeepFool method for a single original input.
        """

        # x_adv is the placeholder to store the perturbed image.
        x_adv = x.clone()
        # Reshapes the adversarial tensor to include the batch value.
        x_adv = x_adv.reshape((1,) + x_adv.shape)
        x_adv.requires_grad_()

        # Initializing the weights and perturbation tensors.
        w = torch.zeros(self.model.x_shape).to(self.device)
        r = torch.zeros(self.model.x_shape).to(self.device)

        step = 0

        # Reshapes the adversarial tensor to include the batch value.
        f_logits, f_labels = self.model(x_adv)

        y_label = y.argmax()
        f_label = f_labels[0].argmax()
        k_label = f_label

        # Stores the progress of the adversarial example for the iteration benchmark.
        adv_iter = []

        # If the model does not correctly classify the original image, no need to craft an adversarial example.
        if y_label != f_label:
            return x_adv.detach()[0], [x_adv.detach()[0]]

        while k_label == f_label and step < self.iteration:
            # l stores the perturbation value for the k-th step
            l = torch.inf

            f_logits[:, f_label].backward(retain_graph=True)
            grad_og = x_adv.grad.clone()

            for k in range(self.model.n_class):
                # Calculating the perturbation value for all classes except the ground-truth.
                if k != f_label:

                    f_logits[:, k].backward(retain_graph=True)
                    grad_adv = x_adv.grad.clone()

                    # Compute the new weights, logits and perturbation value for the k-th class.
                    w_k = grad_adv[0] - grad_og[0]
                    f_k = f_logits[0, f_label] - f_logits[0, k]

                    if self.distance_metric == "l_2":
                        w_k_norm = torch.linalg.vector_norm(w_k, ord=2)
                    else:
                        w_k_norm = torch.linalg.vector_norm(w_k, ord=1)

                    l_k = torch.abs(f_k) / w_k_norm

                    # The goal is to minimize the perturbation.
                    if l_k < l:
                        l = l_k
                        w = w_k

            # Update the adversarial image using the minimal perturbation.
            if self.distance_metric == "l_2":
                w_norm = torch.linalg.vector_norm(w, ord=2)
                r_k = l * w / w_norm
            else:
                r_k = l * torch.sign(w)

            r = r + r_k

            x_adv = x.clone() + (1 + self.overshoot) * r
            x_adv.detach_()

            # Makes sure the adversarial image remains within the [min, max] values of the pixels.
            x_adv = torch.clamp(x_adv, self.model.x_min, self.model.x_max)

            adv_iter.append(x_adv)

            # Reshapes the adversarial tensor to include the batch value.
            x_adv = x_adv.reshape((1,) + x_adv.shape)

            x_adv.requires_grad_()

            f_logits, f_labels = self.model(x_adv)
            k_label = f_labels[0].argmax()

            step += 1

        return x_adv.detach()[0], adv_iter
