"""C&W attack module.

Module to create an interface for the C&W adversarial attack. Based on the BatchAttack base class.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.ares.attack.base import BatchAttack
from src.ares.attack.utils import maybe_to_tensor


class CW(BatchAttack):
    """Implements the Carlini & Wagner adversarial attack. Works as a white-box iterative optimization method. Requires
    a differentiable loss function and a classifier model.

    Parameters :
    - Supported distance metric : 'l_2'.
    - Supported goal : 't', 'tm' and 'ut'.

    Reference : https://arxiv.org/abs/1511.04599.
    """

    def __init__(self, model, batch_size, loss, goal, distance_metric, iteration_callback=False, device="cpu"):
        """Initializes the C&W attack.

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

        if distance_metric == "l_inf":
            raise NotImplementedError
        self.goal = goal
        self.distance_metric = distance_metric

        # Cost initialized to 0., must be configured using config() method.
        # It could be either a 'float' type number, or a 'torch.Tensor' of 'float' numbers and of shape = (effective
        # batch size,). If it is a number, it will be converted to a tensor during the 'config()' method.
        self.cs = 0.
        # The confidence initialized to 0., must be configured using the 'config()' method.
        self.kappa = 0.
        # The learning rate initialized to 0., must be configured using the 'config()' method.
        self.learning_rate = 0.
        # The number of search steps initialized to 0, must be configured using the 'config()' method.
        self.search_steps = 0
        # The number of binsearch steps initialized to 0, must be configured using the 'config()' method.
        self.binsearch_steps = 0
        # The number of iterations for each step initialized to 0, must be configured using the 'config()' method.
        self.iteration_steps = 0
        # The number of iterations when using the iteration benchmark, must be configured using the 'config()' method.
        self.iteration = 0
        self.iteration_callback = iteration_callback

        # Initializes the output dictionary.
        self.cw = {"details": {}}

    def config(self, **kwargs):
        """(Re)Config the adversarial attack.

        :param cs: The initial cost value c, can be a 'float' type number or a 'torch.Tensor' of 'float' numbers and
        of shape = (effective batch size,).
        :param kappa: The confidence for the attack, a 'float' type number.
        :param learning_rate: The learning rate of the optimizer for the attack, a 'float' type number.
        :param search_steps: The number of search steps for the attack, a 'int' type number.
        :param binsearch_steps: The number of binsearch steps for the attack, a 'int' type number.
        :param iteration_steps: The number of iterations for each step of the attack, a 'int' type number.
        :param iteration: The number of iterations for the benchmark, a 'int' type number.
        """

        if "cs" in kwargs:
            self.cs = maybe_to_tensor(kwargs["cs"], self.batch_size).to(self.device)
        if "kappa" in kwargs:
            self.kappa = kwargs["kappa"]
        if "learning_rate" in kwargs:
            self.learning_rate = kwargs["learning_rate"]
        if "search_steps" in kwargs:
            self.search_steps = kwargs["search_steps"]
        if "binsearch_steps" in kwargs:
            self.binsearch_steps = kwargs["binsearch_steps"]
        if "iteration_steps" in kwargs:
            self.iteration_steps = kwargs["iteration_steps"]
        if "iteration" in kwargs:
            self.iteration = kwargs["iteration"]

        # For the iteration benchmark, each 'iteration' will refer to a search step or binary search step. The total
        # number of steps should be the same as the number of iterations of the benchmark. When using the attack
        # benchmark or the distortion benchmark, the 'iteration' number is not needed and should be left equal to 0.
        if self.iteration != 0 and (self.iteration != (self.search_steps + self.binsearch_steps)):
            print("When using the iteration benchmark with the C&W adversarial attack, the iteration number should be "
                  "initialized to the total number of search steps")
            raise ValueError

    def batch_attack(self, xs, ys=None, ts=None):
        """Generate a batch of adversarial examples from the input batch.
        """

        # The effective batch_size() for the attack.
        self.eff_batch_size = xs.size()[0]

        # Initializes the tensor to store the batch of original images.
        xs_og = xs.detach().clone()

        # Initializes the tensor to store the batch of adversarial images.
        xs_adv = torch.zeros(((self.eff_batch_size,) + self.model.x_shape))

        # Initializes the tensor to store the optimal batch (for minimum perturbation) of adversarial images.
        xs_adv_min = xs.detach().clone()

        # The weights to optimize for the attack.
        w = torch.atanh(2 * xs.detach() - 1)
        w.requires_grad_()

        # The minimal norm for the attack.
        l2_min = torch.inf * torch.ones((self.eff_batch_size,)).to(self.device)

        cw_loss = nn.MSELoss(reduction="none")

        # Iteration step counter for the iteration benchmark.
        iteration_step = 1

        # Find the cost value cs to begin the binsearch with.
        # Store the images for which an adversarial attack has been found.
        found_succ = torch.zeros((self.eff_batch_size,))
        for step in range(self.search_steps):
            # Reset the optimizer for each iteration.
            optimizer = optim.Adam([w], lr=self.learning_rate)

            for it in range(self.iteration_steps):
                # Generate the new batch of adversarial images.
                xs_adv = 0.5 * (torch.tanh(w.clone()) + 1)

                l2 = cw_loss(xs_adv.flatten(start_dim=1), xs_og.flatten(start_dim=1)).sum(dim=1).to(self.device)

                logits_adv, labels_adv = self.model(xs_adv)
                labels_adv = labels_adv.to(self.device)

                if self.goal in ("t", "tm"):
                    f_tot = self._f(logits_adv, ts)
                    # Tensor of adversarial attacks that have been successful.
                    succ_pred = torch.eq(labels_adv.argmax(1), ts.argmax(1))
                else:
                    f_tot = self._f(logits_adv, ys)
                    # Tensor of adversarial attacks that have been successful.
                    succ_pred = torch.ne(labels_adv.argmax(1), ys.argmax(1))

                cost = (l2 + self.cs[:self.eff_batch_size] * f_tot).sum()

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                l2.detach_()
                xs_adv.detach_()
                # Tensor of adversarial attacks that have successfully decreased the distance.
                succ_dist = l2_min > l2

                # Only adversarial examples which achieve correct prediction and decrease the minimal loss are kept.
                succ = torch.logical_and(succ_dist, succ_pred)
                l2_min[succ] = l2[succ]
                xs_adv_min[succ] = xs_adv[succ]
                found_succ[succ] = 1.

            if self.iteration_callback:
                iter_key = "xs_adv_{}".format(iteration_step)
                self.cw[iter_key] = xs_adv_min.detach().clone()
                iteration_step += 1

            # If all adversarial images are successful, stops the search early.
            # Else, increase the base cost value c for all images that had an unsuccessful attack.
            if found_succ.sum().item() == self.eff_batch_size:
                # In case the search is stopped early, complete the iteration callback for the iteration benchmark using
                # the same adversarial examples.
                if self.iteration_callback:
                    # Compute the number of steps skipped and update the output dictionary.
                    skipped = self.search_steps - step - 1
                    for k in range(skipped):
                        iter_key = "xs_adv_{}".format(iteration_step)
                        self.cw[iter_key] = xs_adv_min.detach().clone()
                        iteration_step += 1

                break
            else:
                self.cs[:self.eff_batch_size][torch.logical_not(found_succ)] *= 10.0

        # Optimize the cost value cs using a binary search.
        cs_low = torch.zeros_like(self.cs[:self.eff_batch_size])
        cs_high = self.cs[:self.eff_batch_size].clone()
        self.cs[:self.eff_batch_size] = (cs_low + cs_high) / 2.0

        for step in range(self.binsearch_steps):
            # Reset the optimizer for each iteration.
            optimizer = optim.Adam([w], lr=self.learning_rate)

            for it in range(self.iteration_steps):
                # Generate the new batch of adversarial images.
                xs_adv = 0.5 * (torch.tanh(w.clone()) + 1)

                l2 = cw_loss(xs_adv.flatten(start_dim=1), xs_og.flatten(start_dim=1)).sum(dim=1).to(self.device)

                logits_adv, labels_adv = self.model(xs_adv)
                labels_adv = labels_adv.to(self.device)

                if self.goal in ("t", "tm"):
                    f_tot = self._f(logits_adv, ts)
                    # Tensor of adversarial attacks that have been successful.
                    succ_pred = torch.eq(labels_adv.argmax(1), ts.argmax(1))
                else:
                    f_tot = self._f(logits_adv, ys)
                    # Tensor of adversarial attacks that have been successful.
                    succ_pred = torch.ne(labels_adv.argmax(1), ys.argmax(1))

                cost = (l2 + self.cs[:self.eff_batch_size] * f_tot).sum()

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                l2.detach_()
                xs_adv.detach_()
                # Tensor of adversarial attacks that have successfully decreased the distance.
                succ_dist = l2_min > l2

                # Only adversarial examples which achieve correct prediction and decrease the minimal loss are kept.
                succ = torch.logical_and(succ_dist, succ_pred)
                l2_min[succ] = l2[succ]
                xs_adv_min[succ] = xs_adv[succ]

            if self.iteration_callback:
                iter_key = "xs_adv_{}".format(iteration_step)
                self.cw[iter_key] = xs_adv_min.detach().clone()
                iteration_step += 1

            # Update the value of cs.
            not_succ = torch.logical_not(succ)
            cs_low[not_succ] = self.cs[:self.eff_batch_size][not_succ]
            cs_high[succ] = self.cs[:self.eff_batch_size][succ]
            self.cs[:self.eff_batch_size] = (cs_low + cs_high) / 2.0

        self.cw["xs_adv"] = xs_adv_min.detach()
        return self.cw

    def _f(self, logits, ys):
        """Implement the f function to optimize.
        """

        truth_logit = torch.masked_select(logits, ys.bool())
        largest_logit, _ = torch.max((1 - ys) * logits, dim=1)

        if self.goal in ("t", "tm"):
            return torch.clamp((largest_logit - truth_logit), min=-self.kappa)
        else:
            return torch.clamp((truth_logit - largest_logit), min=-self.kappa)
