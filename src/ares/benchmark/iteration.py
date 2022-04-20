"""Iteration benchmark module.

Module to create an interface to benchmark adversarial attacks on defense models, returns accuracy and distortion values
for each attack iteration.
"""

import logging

import torch

from src.ares.benchmark.utils import load_attack
from src.ares.dataset import dataset_to_iterator


class IterationBenchmark(object):
    """Run the iteration benchmark for an adversarial attack on some model and report the results.
    """

    def __init__(self, attack_name, model, loss, batch_size, dataset, goal, distance_metric, iteration, device="cpu",
                 **kwargs):
        """Initializes the Iteration Benchmark module.

        :param attack_name: The adversarial attack module name. Valid values are : 'bim', 'pgd', 'deepfool', 'cw',
            'nes', 'spsa' and 'nattack'.
        :param model: The model to run the attack on, a 'ClassifierWithLogits' instance.
        :param loss: The loss function used to optimize the attacks, a 'torch.nn' loss function.
        :param batch_size: Batch size used for benchmarking the attacks, as an 'int' number. CAUTION : when going
            through the dataset using the iterator, the effective batch size for the last batch may be smaller than the
            defined value.
        :param dataset: The dataset for the model and the attack, a 'torch.utils.data.Dataset' instance.
        :param goal: The adversarial goal for the attack method. All valid values are 't', for targeted attack, 'tm',
            for targeted missclassification attack and 'ut', for untargeted attack.
        :param distance_metric: The adversarial distance metric for the attack method. All valid values are 'l_2' and
            'l_inf'.
        :param iteration: The number of iterations for the attack, as an 'int' type number.
        :param device: The device to run the computation on, "cpu" by default.
        :param kwargs: Other specific keyword arguments to pass to the attack methods during initialization.
        """

        # Setting up the init kwargs for the attack method, see 'load_attack()' doc for more details.
        init_kwargs = {
                "model": model,
                "batch_size": batch_size,
                "loss": loss,
                "goal": goal,
                "distance_metric": distance_metric,
                "iteration_callback": True,
                "device": device
                }

        for k, v in kwargs.items():
            init_kwargs[k] = v

        # Iteration benchmark parameters.
        self.iteration = iteration

        self.attack_name = attack_name
        # Model must be in model.eval() state.
        self.model = model
        self.loss = loss
        self.device = device

        self.batch_size = batch_size
        # The 'real' batch size when computing with batches of different size, in that case, batch size is the maximum
        # size of the batches.
        self.eff_batch_size = batch_size
        self.dataset = dataset

        self.goal = goal
        self.distance_metric = distance_metric

        # loading the attack module, a 'Attack' or 'BatchAttack' instance.
        self.attack = load_attack(attack_name, init_kwargs)

        # creating the dataset iterator to run the attack with.
        self.dataset_iterator = dataset_to_iterator(self.dataset, self.batch_size)

        # computing the run format required for the chosen attack.
        if self.attack_name in ("bim", "pgd", "deepfool", "cw",):
            self._run = self._run_basic
        elif self.attack_name in ("nes", "spsa", "nattack",):
            self._run = self._run_queries
        else:
            raise NotImplementedError

    def config(self, **kwargs):
        """(Re)Config the adversarial attack.

        :param kwargs: The arguments for the attack method's 'config()' function.
        """

        if self.attack_name in ("bim", "pgd", "deepfool", "cw",):
            kwargs["iteration"] = self.iteration

        self.attack.config(**kwargs)

    def _run_basic(self, logger=None):
        """The sub run function for the 'bim', 'pgd', 'deepfool' and 'cw' attacks.
        """
        # the dictionary of results, each element is a tuple of (adversarial accuracy, distortion) for each iteration.
        rs = {}

        for batch, data in enumerate(self.dataset_iterator):
            xs = data["image"].to(self.device)
            ys, ts = data["label"].to(self.device), data["target"].to(self.device)

            # the effective batch_size() for the attack.
            self.eff_batch_size = xs.size()[0]

            # computing the grad with respect to the input
            self._grad(xs, ys, ts)

            # creating the adversarial data
            data_adv = self.attack.batch_attack(xs, ys, ts)

            xs.detach_()

            for it in range(self.iteration):
                it_key = "iteration_{}".format(it+1)

                if batch == 0:
                    rs[it_key] = ([], [])

                xs_adv = data_adv["xs_adv_{}".format(it+1)].to(self.device)

                # feeding the adversarial data to the defense model
                with torch.no_grad():
                    _, labels_adv = self.model(xs_adv)

                acc = torch.mean(self._batch_info(ys.to("cpu"), ts.to("cpu"), labels_adv.to("cpu")))
                dist = torch.mean(self._distance(xs.to("cpu") - xs_adv.to("cpu")))

                rs[it_key][0].append(acc)
                rs[it_key][1].append(dist)

                # empty memory on the cpu and on the device used for computations
                del xs_adv, labels_adv

            # Empty memory on the cpu and on the device used for computations.
            del xs, ys, ts

        torch.cuda.empty_cache()

        return rs

    def _run_queries(self, logger=None):
        """The sub run function for the 'nes', 'spsa' and 'nattack' attacks.
        """

        # The dictionary of results, each element is a tuple of (adversarial accuracy, distortion, queries) for each
        # iteration.
        rs = {}

        for batch, data in enumerate(self.dataset_iterator):
            xs = data["image"].to(self.device)
            ys, ts = data["label"].to(self.device), data["target"].to(self.device)

            # The effective batch_size() for the attack.
            self.eff_batch_size = xs.size()[0]

            # Initialize the tensor to store the batch of adversarial images.
            xs_adv = torch.zeros_like(xs.detach())

            # Initialize a list to store all the iteration values for a batch of images.
            data = torch.zeros((self.eff_batch_size, self.iteration,) + xs[0].shape)

            # Computing the grad with respect to the input.
            self._grad(xs, ys, ts)

            # Creating the adversarial data.
            for i in range(self.eff_batch_size):
                nes = self.attack.attack(xs[i], ys[i], ts[i])

                for it in range(self.iteration):
                    it_key = "x_adv_{}".format(it + 1)
                    data[i][it] = nes[it_key].clone().to(self.device)

            queries = nes["Queries_by_iteration"]

            xs.detach_()

            for it in range(self.iteration):
                it_key = "iteration_{}".format(it + 1)

                if batch == 0:
                    rs[it_key] = ([], [], [])

                xs_adv = data[:, it].clone().to(self.device)

                # feeding the adversarial data to the defense model
                with torch.no_grad():
                    _, labels_adv = self.model(xs_adv)

                acc = torch.mean(self._batch_info(ys.to("cpu"), ts.to("cpu"), labels_adv.to("cpu")))
                dist = torch.mean(self._distance(xs.to("cpu") - xs_adv.to("cpu")))

                rs[it_key][0].append(acc)
                rs[it_key][1].append(dist)
                rs[it_key][2].append(queries)

            # empty memory on the cpu and on the device used for computations
            del xs_adv, labels_adv

            # Empty memory on the cpu and on the device used for computations.
            del xs, ys, ts

        torch.cuda.empty_cache()

        return rs

    def run(self, logger=None):
        """Run the distortion benchmark attack on the model.

        :param logger: A standard Python logger. If not logger is provided, create a default logger called .__name__.
        :return: A numpy array where each element is a tuple of accuracy and distortion values for each attack
        iterations.
        """

        if logger is None:
            logger_format = "[%(levelname)s] %(asctime)s : %(filename)s Line-%(lineno)s @%(message)s"
            logging.basicConfig(filename=__name__, filemode="w", level=logging.DEBUG, format=logger_format)
            logger = logging.getLogger(__name__)

        return self._run(logger)

    def _grad(self, xs, ys, ts):
        """Calculates the gradient with respect to the input.
        """

        # compute gradient for each image based on the parameters
        xs.requires_grad_()
        logits, labels = self.model(xs)

        # the gradient of the loss is computed with the ground truth for untargeted attacks and with the targets for
        # targeted attacks.
        if self.goal == "ut":
            loss = self.loss(logits, ys)
        else:
            loss = self.loss(logits, ts)
        self.model.zero_grad()
        loss.backward()

        # empty memory on the cpu and on the device used for computations
        logits.detach_()
        del logits, labels

    def _distance(self, zs):
        """Calculate the distance according to the distance metric with zs = xs - xs_adv.
        """

        zs = zs.flatten(start_dim=1)

        if self.distance_metric == "l_2":
            return torch.linalg.vector_norm(zs, ord=2, dim=1)
        else:
            return torch.linalg.vector_norm(zs, ord=torch.inf, dim=1)

    def _batch_info(self, ys, ts, labels_adv):
        """Creates a boolean array where each element checks whether the image is adversarial.
        """

        if self.goal in ("ut", "tm"):
            succ = torch.logical_not(torch.eq(ys.argmax(1), labels_adv.argmax(1)))
        else:
            succ = torch.eq(ts.argmax(1), labels_adv.argmax(1))

        return succ.double()
