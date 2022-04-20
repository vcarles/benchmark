"""Distortion benchmark module.

Module to create an interface to benchmark adversarial attacks on defense models, returns minimal distortion value.
"""

import logging

import numpy as np
import torch

from src.ares.benchmark.utils import load_attack
from src.ares.dataset import dataset_to_iterator


class DistortionBenchmark(object):
    """Run the distortion benchmark for an adversarial attack on some model and report the results.
    """

    def __init__(self, attack_name, model, loss, batch_size, dataset, goal, distance_metric, distortion=0.,
                 search_steps=5, binsearch_steps=10, device="cpu", **kwargs):
        """Initializes the Distortion Benchmark module.

        :param attack_name: The adversarial attack module name. Valid values are : 'fgsm', 'bim', 'pgd', 'deepfool',
            'cw', 'nes', 'spsa' and 'nattack'.
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
        :param distortion: The initial distortion, used as a starting point when benchmarking magnitude-based attacks.
        :param search_steps: The number of search steps for finding an effective initial distortion.
        :param binsearch_steps: The number of binary search steps for refining the distortion.
        :param device: The device to run the computation on, 'cpu' by default.
        :param kwargs: Other specific keyword arguments to pass to the attack methods during initialization.
        """

        # setting up the init kwargs for the attack method, see 'load_attack()' doc for more details.
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

        # distortion benchmark parameters.
        self.init_distortion = distortion
        self.search_steps = search_steps
        self.binsearch_steps = binsearch_steps

        self.attack_name = attack_name
        # model must be in model.eval() state
        self.model = model
        self.loss = loss
        self.device = device

        self.batch_size = batch_size
        # the 'real' batch size when computing with batches of different size, in that case, batch size is the maximum
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
        if self.attack_name == "fgsm":
            self._run = self._run_binsearch
        elif self.attack_name in ("bim", "pgd",):
            self._run = self._run_binsearch_alpha
        elif self.attack_name in ("deepfool", "cw",):
            self._run = self._run_optimized
        elif self.attack_name in ("nes", "spsa", "nattack",):
            self._run = self._run_binsearch_single
        else:
            raise NotImplementedError

    def config(self, **kwargs):
        """(Re)Config the adversarial attack.

        :param kwargs: The arguments for the attack method's 'config()' function.
        """

        self.attack.config(**kwargs)

    def _run_binsearch(self, logger=None):
        """The sub run function for the 'fgsm' attack.
        """

        # the array of results, each element is the distortion for the corresponding image in the dataset.
        rs = []

        for batch, data in enumerate(self.dataset_iterator):
            xs = data["image"].to(self.device)
            ys, ts = data["label"].to(self.device), data["target"].to(self.device)

            # the effective batch_size() for the attack.
            self.eff_batch_size = xs.size()[0]

            # lo is the lower bound and hi is the higher bound for the minimal distortion.
            lo = torch.zeros(self.eff_batch_size, dtype=torch.float)
            hi = lo + self.init_distortion

            # initializing the results value for the batch to zeros, this allows to identify whether the attack failed.
            xs_result = torch.zeros_like(xs.detach()).to(self.device)

            # computing the grad with respect to the input
            self._grad(xs, ys, ts)

            # using linear search to initialize the 'effective' distortion, a distortion capable of crafting
            # adversarial attack. The attack runs with a magnitude in range of :
            # [ init_distortion, init_distortion * 2, init_distortion * 3, ..., init_distortion * (2**search_steps) ].
            for i in range(2**self.search_steps):
                magnitude = self.init_distortion * (2**self.search_steps - i)

                # reconfig the attack.
                self.attack.config(magnitude=magnitude)

                # creating the adversarial data.
                data_adv = self.attack.batch_attack(xs, ys, ts)
                xs_adv = data_adv["xs_adv"].to(self.device)

                # feeding the adversarial data to the defense model.
                with torch.no_grad():
                    _, labels_adv = self.model(xs_adv)

                # checks whether the attack succeeded or not for each image, creates a boolean tensor succ.
                succ = self._batch_info(ys.to("cpu"), ts.to("cpu"), labels_adv.to("cpu"))

                # update the adversarial examples and smallest distortion.
                xs_result[succ] = xs_adv[succ]
                hi[succ] = magnitude

            del xs_adv, labels_adv

            # updating the logger.
            logger.info("Linsearch for batch number {}, success rate is {:.3f}"
                        .format(batch, succ.numpy().astype(np.float).mean()))

            lo = hi - self.init_distortion

            # run binsearch to find the minimal adversarial magnitude, for each image, between lo and hi.
            for i in range(self.binsearch_steps):
                # reconfig the attack.
                mi = (lo + hi) / 2
                self.attack.config(magnitude=mi)

                # creating the adversarial data.
                data_adv = self.attack.batch_attack(xs, ys, ts)
                xs_adv = data_adv["xs_adv"].to(self.device)

                # feeding the adversarial data to the defense model.
                with torch.no_grad():
                    _, labels_adv = self.model(xs_adv)

                # checks whether the attack succeeded or not for each image, creates a boolean tensor succ.
                succ = self._batch_info(ys.to("cpu"), ts.to("cpu"), labels_adv.to("cpu"))

                # update the adversarial examples and smallest distortion.
                xs_result[succ] = xs_adv[succ]

                # update hi (lower, if succeed) or lo (higher, if succeed)
                not_succ = torch.logical_not(succ)
                hi[succ] = mi[succ]
                lo[not_succ] = mi[not_succ]

            # updating the logger.
            logger.info("Binsearch for batch number {}, success rate is {:.3f}"
                        .format(batch, succ.numpy().astype(np.float).mean()))

            # the lowest the magnitude of the attack, the lowest the distortion. For each image computes the minimum
            # distortion and return the results.
            xs.detach_()

            with torch.no_grad():
                _, labels = self.model(xs)

            succ_model = self._batch_info(ys, ts, labels.to(self.device))

            xs_fail = torch.zeros_like(xs).to(self.device)

            for i in range(self.eff_batch_size):
                if succ_model[i]:
                    rs.append(0.)
                else:
                    if torch.equal(xs_result[i], xs_fail[i]):
                        rs.append(np.nan)
                    else:
                        rs.append(self._distance(xs_result[i].to("cpu") - xs[i].to("cpu")))

            del xs_adv, labels_adv, xs, xs_fail, xs_result

        torch.cuda.empty_cache()

        return rs

    def _run_binsearch_alpha(self, logger=None):
        """The sub run function for the 'bim' and 'pgd' attacks.
        """

        iteration = self.attack.iteration

        # the array of results, each element is the distortion for the corresponding image in the dataset.
        rs = []

        for batch, data in enumerate(self.dataset_iterator):
            xs = data["image"].to(self.device)
            ys, ts = data["label"].to(self.device), data["target"].to(self.device)

            # the effective batch_size() for the attack.
            self.eff_batch_size = xs.size()[0]

            # lo is the lower bound and hi is the higher bound for the minimal distortion.
            lo = torch.zeros(self.eff_batch_size, dtype=torch.float)
            hi = lo + self.init_distortion

            # initializing the results value for the batch to zeros, this allows to identify whether the attack failed.
            xs_result = torch.zeros_like(xs.detach()).to(self.device)

            # computing the grad with respect to the input
            self._grad(xs, ys, ts)

            # using exponential search to initialize the 'effective' distortion, a distortion capable of crafting
            # adversarial attacks.
            for i in range(self.search_steps):
                # reconfig the attack.
                self.attack.config(magnitude=hi, alpha=hi * 1.5 / iteration)

                # creating the adversarial data.
                data_adv = self.attack.batch_attack(xs, ys, ts)
                xs_adv = data_adv["xs_adv"].to(self.device)

                # feeding the adversarial data to the defense model.
                with torch.no_grad():
                    _, labels_adv = self.model(xs_adv)

                # checks whether the attack succeeded or not for each image, creates a boolean tensor succ.
                succ = self._batch_info(ys.to("cpu"), ts.to("cpu"), labels_adv.to("cpu"))

                # update the adversarial examples and smallest distortion.
                xs_result[succ] = xs_adv[succ]

                # if the attack is not successful, use a larger magnitude for next iteration
                not_succ = torch.logical_not(succ)
                lo[not_succ] = hi[not_succ]
                hi[not_succ] *= 2

                if torch.all(succ):
                    break

            del xs_adv, labels_adv

            # updating the logger.
            logger.info("Linsearch for batch number {}, success rate is {:.3f}"
                        .format(batch, succ.numpy().astype(np.float).mean()))

            # run binsearch to find the minimal adversarial magnitude, for each image, between lo and hi.
            for i in range(self.binsearch_steps):
                # reconfig the attack.
                mi = (lo + hi) / 2
                self.attack.config(magnitude=mi, alpha=mi * 1.5 / iteration)

                # creating the adversarial data.
                data_adv = self.attack.batch_attack(xs, ys, ts)
                xs_adv = data_adv["xs_adv"].to(self.device)

                # feeding the adversarial data to the defense model.
                with torch.no_grad():
                    _, labels_adv = self.model(xs_adv)

                # checks whether the attack succeeded or not for each image, creates a boolean tensor succ.
                succ = self._batch_info(ys.to("cpu"), ts.to("cpu"), labels_adv.to("cpu"))

                # update the adversarial examples and smallest distortion.
                xs_result[succ] = xs_adv[succ]
                # update hi (lower, if succeed) or lo (higher, if succeed)
                not_succ = torch.logical_not(succ)
                hi[succ] = mi[succ]
                lo[not_succ] = mi[not_succ]

            # updating the logger.
            logger.info("Binsearch for batch number {}, success rate is {:.3f}"
                        .format(batch, succ.numpy().astype(np.float).mean()))

            # The lowest the magnitude of the attack, the lowest the distortion. For each image computes the minimum
            # distortion and return the results.
            xs.detach_()

            with torch.no_grad():
                _, labels = self.model(xs)

            succ_model = self._batch_info(ys, ts, labels.to(self.device))

            xs_fail = torch.zeros_like(xs).to(self.device)

            for i in range(self.eff_batch_size):
                if succ_model[i]:
                    rs.append(0.)
                else:
                    if torch.equal(xs_result[i], xs_fail[i]):
                        rs.append(np.nan)
                    else:
                        rs.append(self._distance(xs_result[i].to("cpu") - xs[i].to("cpu")))

            del xs_adv, labels_adv, xs, xs_fail, xs_result

        torch.cuda.empty_cache()

        return rs

    def _run_optimized(self, logger=None):
        """The sub run function for the optimized perturbation attacks : 'deepfool' and 'c&w'.
        """

        # The array of results, each element is the distortion for the corresponding image in the dataset.
        rs = []

        for batch, data in enumerate(self.dataset_iterator):
            xs = data["image"].to(self.device)
            ys, ts = data["label"].to(self.device), data["target"].to(self.device)

            # The effective batch_size() for the attack.
            self.eff_batch_size = xs.size()[0]

            # Creating the adversarial data.
            data_adv = self.attack.batch_attack(xs, ys, ts)
            xs_adv = data_adv["xs_adv"].to(self.device)

            # Feeding the adversarial data and original data to the defense model.
            with torch.no_grad():
                _, labels = self.model(xs)
                _, labels_adv = self.model(xs_adv)

            xs.detach_()

            succ_model = self._batch_info(ys, ts, labels.to(self.device))
            succ_adv = self._batch_info(ys, ts, labels_adv.to(self.device))

            for i in range(self.eff_batch_size):
                if succ_model[i]:
                    rs.append(0.)
                else:
                    if succ_adv[i]:
                        rs.append(self._distance(xs[i].to("cpu") - xs_adv[i].to("cpu")))
                    else:
                        rs.append(np.nan)

                logger.info("For Batch n°{}, and Image n°{} : the minimal distortion value is {}".format(batch, i,
                                                                                                         rs[-1]))

            del xs_adv, labels_adv, xs, labels

        torch.cuda.empty_cache()

        return rs

    def _run_binsearch_single(self, logger=None):
        """The sub run function for the 'nes', 'spsa' and 'nattack' attacks.
        """

        # Rhe array of results, each element is the distortion for the corresponding image in the dataset.
        rs = []

        for batch, data in enumerate(self.dataset_iterator):
            xs = data["image"].to(self.device)
            ys, ts = data["label"].to(self.device), data["target"].to(self.device)

            # The effective batch_size() for the attack.
            self.eff_batch_size = xs.size()[0]

            # lo is the lower bound and hi is the higher bound for the minimal distortion.
            lo = torch.zeros(self.eff_batch_size, dtype=torch.float)
            hi = lo + self.init_distortion

            # Initializing the results value for the batch to zeros, this allows to identify whether the attack failed.
            xs_result = torch.zeros_like(xs.detach()).to(self.device)

            # Initialize the tensor to store the batch of adversarial images.
            xs_adv = torch.zeros_like(xs_result)

            # Computing the grad with respect to the input.
            self._grad(xs, ys, ts)

            # Using exponential search to initialize the 'effective' distortion, a distortion capable of crafting
            # adversarial attacks.
            for i in range(self.search_steps):

                # Creating the adversarial data.
                for k in range(self.eff_batch_size):
                    # Reconfig the attack.
                    self.attack.config(magnitude=hi[k])

                    data_adv = self.attack.attack(xs[k], ys[k], ts[k])
                    x_adv = data_adv["x_adv"].to(self.device)
                    xs_adv[k] = x_adv

                # Feeding the adversarial data to the defense model.
                with torch.no_grad():
                    _, labels_adv = self.model(xs_adv)

                # Checks whether the attack succeeded or not for each image, creates a boolean tensor succ.
                succ = self._batch_info(ys.to("cpu"), ts.to("cpu"), labels_adv.to("cpu"))

                # Update the adversarial examples and smallest distortion.
                xs_result[succ] = xs_adv[succ]

                # If the attack is not successful, use a larger magnitude for next iteration.
                not_succ = torch.logical_not(succ)
                lo[not_succ] = hi[not_succ]
                hi[not_succ] *= 2

                if torch.all(succ):
                    break

            # Updating the logger.
            logger.info("Linsearch for batch number {}, success rate is {:.3f}"
                        .format(batch, succ.numpy().astype(np.float).mean()))

            # Run binsearch to find the minimal adversarial magnitude, for each image, between lo and hi.
            for i in range(self.binsearch_steps):

                # Reconfig the attack.
                mi = (lo + hi) / 2

                # Creating the adversarial data.
                for k in range(self.eff_batch_size):
                    # Reconfig the attack.
                    self.attack.config(magnitude=mi[k])

                    data_adv = self.attack.attack(xs[k], ys[k], ts[k])
                    x_adv = data_adv["x_adv"].to(self.device)
                    xs_adv[k] = x_adv

                # Feeding the adversarial data to the defense model.
                with torch.no_grad():
                    _, labels_adv = self.model(xs_adv)

                # Checks whether the attack succeeded or not for each image, creates a boolean tensor succ.
                succ = self._batch_info(ys.to("cpu"), ts.to("cpu"), labels_adv.to("cpu"))

                # Update the adversarial examples and smallest distortion.
                xs_result[succ] = xs_adv[succ]
                # Update hi (lower, if succeed) or lo (higher, if succeed).
                not_succ = torch.logical_not(succ)
                hi[succ] = mi[succ]
                lo[not_succ] = mi[not_succ]

            # Updating the logger.
            logger.info("Binsearch for batch number {}, success rate is {:.3f}"
                        .format(batch, succ.numpy().astype(np.float).mean()))

            # The lowest the magnitude of the attack, the lowest the distortion. For each image computes the minimum
            # distortion and return the results.
            xs.detach_()

            with torch.no_grad():
                _, labels = self.model(xs)

            succ_model = self._batch_info(ys, ts, labels.to(self.device))

            xs_fail = torch.zeros_like(xs).to(self.device)

            for i in range(self.eff_batch_size):
                if succ_model[i]:
                    rs.append(0.)
                else:
                    if torch.equal(xs_result[i], xs_fail[i]):
                        rs.append(np.nan)
                    else:
                        rs.append(self._distance(xs_result[i].to("cpu") - xs[i].to("cpu")))

            del xs_adv, labels_adv, xs, xs_fail, xs_result

        torch.cuda.empty_cache()

        return rs

    def run(self, logger=None):
        """Run the distortion benchmark attack on the model.

        :param logger: A standard Python logger. If not logger is provided, create a default logger called .__name__.
        :return: A numpy array where each element is the minimal distortion value for each input. If the attack failed
        to create adversarial examples, their values are set to 'np.nan'.
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
        logits.detach_()
        del logits

    def _distance(self, zs):
        """Calculate the distance according to the distance metric with zs = xs - xs_adv.
        """

        zs = zs.flatten(start_dim=0)

        if self.distance_metric == "l_2":
            return torch.linalg.vector_norm(zs, ord=2)
        else:
            return torch.linalg.vector_norm(zs, ord=torch.inf)

    def _batch_info(self, ys, ts, labels_adv):
        """Creates a boolean array where each element checks whether the image is adversarial.
        """

        if self.goal in ("ut", "tm"):
            succ = torch.logical_not(torch.eq(ys.argmax(1), labels_adv.argmax(1)))
        else:
            succ = torch.eq(ts.argmax(1), labels_adv.argmax(1))

        return succ
