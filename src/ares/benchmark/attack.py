"""Attack benchmark module.

Module to create an interface to run adversarial attacks on defense models.
"""

import logging

import numpy as np
import torch

from src.ares.benchmark.utils import load_attack
from src.ares.dataset import dataset_to_iterator


class AttackBenchmark(object):
    """Run an adversarial attack on some model and report the results.
    """

    def __init__(self, attack_name, model, loss, batch_size, dataset, goal, distance_metric, device="cpu", **kwargs):
        """Initializes the Benchmark module.

        :param attack_name: The adversarial attack module name. Valid values are : 'fgsm', 'bim', 'pgd', 'deepfool',
            'cw', 'nes', 'spsa', and 'nattack'.
        :param model: The model to run the attack on, a 'ClassifierWithLogits' instance.
        :param loss: The loss function used to optimize the attacks, a 'torch.nn' loss function.
        :param batch_size: Batch size used for benchmarking the attacks, as an 'int' number. CAUTION : when going
            through the dataset using the iterator, the effective batch size for the last batch may be smaller than the
            defined value.
        :param dataset: The dataset for the model and the attack, a 'torch.utils.data.Dataset' instance.
        :param goal: The adversarial goal for the attack method. All valid values are 't', for targeted attack, 'tm',
            for targeted misclassification attack and 'ut', for untargeted attack.
        :param distance_metric: The adversarial distance metric for the attack method. All valid values are 'l_2' and
        'l_inf'.
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
                "device": device,
                }

        for k, v in kwargs.items():
            init_kwargs[k] = v

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

        # Loading the attack module, a 'Attack' or 'BatchAttack' instance.
        self.attack = load_attack(attack_name, init_kwargs)

        # Creating the dataset iterator to run the attack with.
        self.dataset_iterator = dataset_to_iterator(self.dataset, self.batch_size)

    def config(self, **kwargs):
        """(Re)Config the adversarial attack.

        :param kwargs: The arguments for the attack method's 'config()' function.
        """

        self.attack.config(**kwargs)

    def run(self, logger=None):
        """Run the attack on the model for a given magnitude.

        :param logger: A standard Python logger. If not logger is provided, creates a default logger called __name__.
        :return: A tuple of five numpy arrays. The first one is the accuracy of the model for each image in the dataset.
        The second is the accuracy of the model for each adversarial image generated from the dataset. The third one is
        the total number of adversarial images considered non-adversarial by the model (=properly classified, based on
        attack goal). The forth one represents the number of successful attacks (based on attack goal). The last one is
        the distance between the adversarial examples and the dataset's images (based on the distance metric).
        """

        if logger is None:
            logger_format = "[%(levelname)s] %(asctime)s : %(filename)s Line-%(lineno)s @%(message)s"
            logging.basicConfig(filename=__name__, filemode="w", level=logging.DEBUG, format=logger_format)
            logger = logging.getLogger(__name__)

        acc, acc_adv, total, succ, dist = [], [], [], [], []

        def update(accs, accs_adv, totals, succs, dists, batch_id):
            # switching from torch.Tensor to numpy.Array instances.
            accs, accs_adv, totals, succs = np.array(accs), np.array(accs_adv), np.array(totals), np.array(succs)
            dists = np.array(dists)

            acc.append(accs)
            acc_adv.append(accs_adv)
            total.append(totals)
            succ.append(succs)
            dist.append(dists)

            if batch_id % 10 == 0:
                logger.info("For batch number {} : acc={:3f}, acc_adv={:3f}, succ={:3f}, dist_mean={:3f}"
                            .format(batch, np.mean(accs), np.mean(accs_adv),
                                    np.sum(succs) / np.sum(totals), np.mean(dists)))

        if self.attack_name in ("fgsm", "bim", "pgd", "deepfool", "cw",):
            for batch, data in enumerate(self.dataset_iterator):
                xs = data["image"].to(self.device)
                ys, ts = data["label"].to(self.device), data["target"].to(self.device)

                # the effective batch_size() for the attack.
                self.eff_batch_size = xs.size()[0]

                labels = self._grad(xs, ys, ts)

                # creating the adversarial data
                data_adv = self.attack.batch_attack(xs, ys, ts)
                xs_adv = data_adv["xs_adv"].to(self.device)

                # feeding the adversarial data to the defense model
                with torch.no_grad():
                    _, labels_adv = self.model(xs_adv)

                xs.detach_()

                update(*self._batch_info(xs.to("cpu"), xs_adv.to("cpu"), ys.to("cpu"), ts.to("cpu"),
                                         labels.to("cpu"), labels_adv.to("cpu")), batch)

                # empty memory on the cpu and on the device used for computations
                del xs, xs_adv, ys, ts, labels, labels_adv

        if self.attack_name in ("nes", "spsa", "nattack",):
            for batch, data in enumerate(self.dataset_iterator):
                xs = data["image"].to(self.device)
                ys, ts = data["label"].to(self.device), data["target"].to(self.device)

                # Initialize the tensor to save the batch of adversarial images.
                xs_adv = torch.zeros_like(xs)

                # The effective batch_size() for the attack.
                self.eff_batch_size = xs.size()[0]

                labels = self._grad(xs, ys, ts)

                for i in range(self.eff_batch_size):
                    # Creating the adversarial data.
                    data_adv = self.attack.attack(xs[i], ys[i], ts[i])
                    x_adv = data_adv["x_adv"].to(self.device)

                    xs_adv[i] = x_adv.detach()

                # Feeding the adversarial data to the defense model.
                with torch.no_grad():
                    _, labels_adv = self.model(xs_adv)

                xs.detach_()

                update(*self._batch_info(xs.to("cpu"), xs_adv.to("cpu"), ys.to("cpu"), ts.to("cpu"),
                                         labels.to("cpu"), labels_adv.to("cpu")), batch)

                # empty memory on the cpu and on the device used for computations
                del xs, xs_adv, ys, ts, labels, labels_adv

        torch.cuda.empty_cache()

        return acc, acc_adv, total, succ, dist

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
        xs.detach_()
        logits.detach_()
        del xs, logits

        return labels

    def _distance(self, zs):
        """Calculate the distance according to the distance metric with zs = xs - xs_adv.
        """

        zs = zs.flatten(start_dim=1)

        if self.distance_metric == "l_2":
            return torch.linalg.vector_norm(zs, ord=2, dim=1)
        else:
            return torch.linalg.vector_norm(zs, ord=torch.inf, dim=1)

    def _batch_info(self, xs, xs_adv, ys, ts, labels, labels_adv):
        """Calculate the benchmark information for a batch of examples.
        """

        dists = self._distance(xs - xs_adv)
        accs = torch.eq(ys.argmax(1), labels.argmax(1))
        accs_adv = torch.eq(ys.argmax(1), labels_adv.argmax(1))

        if self.goal in ("ut", "tm"):
            totals = torch.eq(ys.argmax(1), labels.argmax(1))
            succs = torch.logical_and(totals, torch.logical_not(torch.eq(ys.argmax(1), labels_adv.argmax(1))))
        else:
            totals = torch.logical_not(torch.eq(ts.argmax(1), labels.argmax(1)))
            succs = torch.logical_and(totals, torch.eq(ts.argmax(1), labels_adv.argmax(1)))

        return accs, accs_adv, totals, succs, dists
