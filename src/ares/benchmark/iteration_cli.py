"""Iteration CLI benchmark module.

Module to benchmark adversarial attacks on defense models, returning accuracy and distortion values for each attack
iterations, using the Command Line Interface to call an 'IterationBenchmark' instance.
"""

import argparse
import logging
import numpy as np
import os
import sys

import torch
from torch import nn

from src.ares.benchmark.iteration import IterationBenchmark
from src.ares.dataset import Cifar10Dataset, ImageNetDataset
# importing all example models.
from examples.cifar10.cifar10_test import Cifar10Test
from examples.cifar10.wideresnet import RST
from examples.imagenet.imagenet_test import ImageNetTest
from examples.imagenet.resnet import RST101


_MODELS = {
        "cifar10_test": Cifar10Test,
        "imagenet_test": ImageNetTest,
        "wideresnet": RST,
        "resnet101": RST101,
        }


if __name__ == "__main__":
    _PARSER = argparse.ArgumentParser(description="Interface to run a benchmark attack on a classifier.")

    # Main benchmark required configuration parameters.
    _PARSER.add_argument("--attack", help="The attack method to benchmark.", required=True,
                         choices=["bim", "pgd", "deepfool", "cw", "nes", "spsa", "nattack"],)
    _PARSER.add_argument("--dataset", help="The dataset for the targeted model.", required=True,
                         choices=["cifar10", "imagenet"],)
    _PARSER.add_argument("--model", help="The targeted model for the benchmark.", required=True,
                         choices=["cifar10_test", "imagenet_test", "wideresnet", "resnet101"],)
    _PARSER.add_argument("--save-path", help="The path to the save the benchmark results at. Use the current folder as "
                                             "default.", type=str,)
    _PARSER.add_argument("--device", help="The device to run the computations on. Uses 'cpu' by default", type=str,)

    # Iteration parameters.
    _PARSER.add_argument("--iteration", type=int, help="The number of iterations for the benchmark.",)

    # Attack method initialization parameters.
    _PARSER.add_argument("--magnitude", type=float,)
    _PARSER.add_argument("--alpha", type=float,)
    _PARSER.add_argument("--overshoot", type=float,)
    _PARSER.add_argument("--cs", type=float,)
    _PARSER.add_argument("--kappa", type=float,)
    _PARSER.add_argument("--learning-rate", type=float,)
    _PARSER.add_argument("--search-steps", type=int,)
    _PARSER.add_argument("--binsearch-steps", type=int,)
    _PARSER.add_argument("--iteration-steps", type=int,)
    _PARSER.add_argument("--samples", type=int,)
    _PARSER.add_argument("--sigma", type=float,)

    # Required parameters.
    _PARSER.add_argument("--goal", help="The attack method goal.", required=True, choices=["t", "ut", "tm"],)
    _PARSER.add_argument("--distance-metric", help="The attack method distance metric.", required=True,
                         choices=["l_2", "l_inf"],)
    _PARSER.add_argument("--batch-size", help="The batch size to run batch attacks.", required=True, type=int,)

    # custom name argument for the logger, default is '__main__'.
    _PARSER.add_argument("--logger", help="Name of the custom logger to use. Else create one by default.", type=str,)

    args = _PARSER.parse_args()

    iteration = None
    if args.attack in ("bim", "pgd", "deepfool", "cw", "nes", "spsa", "nattack",):
        iteration = args.iteration

        if iteration is None:
            _PARSER.error("{} attack requires the --iteration parameter.".format(args.attack))

    if args.attack in ("deepfool",):
        overshoot = args.overshoot

        if overshoot is None:
            _PARSER.error("{} attack requires the --overshoot parameter.".format(args.attack))

    if args.attack in ("cw",):
        cs = args.cs
        kappa = args.kappa
        learning_rate = args.learning_rate
        iteration_steps = args.iteration_steps
        search_steps = args.search_steps
        binsearch_steps = args.binsearch_steps

        if cs is None:
            _PARSER.error("{} attack requires the --cs parameter.".format(args.attack))
        if kappa is None:
            _PARSER.error("{} attack requires the --kappa parameter.".format(args.attack))
        if learning_rate is None:
            _PARSER.error("{} attack requires the --learning-rate parameter.".format(args.attack))
        if iteration_steps is None:
            _PARSER.error("{} attack requires the --iteration-steps parameter.".format(args.attack))
        if search_steps is None:
            _PARSER.error("{} attack requires the --search-steps parameter.".format(args.attack))
        if binsearch_steps is None:
            _PARSER.error("{} attack requires the --binsearch-steps parameter.".format(args.attack))

    if args.attack in ("nes", "spsa", "nattack",):
        samples = args.samples
        sigma = args.sigma
        learning_rate = args.learning_rate
        iteration_steps = args.iteration_steps

        if samples is None:
            _PARSER.error("{} attack requires the --samples parameter.".format(args.attack))
        if sigma is None:
            _PARSER.error("{} attack requires the --sigma parameter.".format(args.attack))
        if learning_rate is None:
            _PARSER.error("{} attack requires the --learning-rate parameter.".format(args.attack))
        if iteration_steps is None:
            _PARSER.error("{} attack requires the --iteration-steps parameter.".format(args.attack))

    # Isolate the attack configuration optional arguments.
    config_kwargs = {}
    for kwarg in ("magnitude", "alpha", "overshoot", "cs", "kappa", "learning_rate", "search_steps",
                  "binsearch_steps", "iteration_steps", "sigma", "samples",):
        attr = getattr(args, kwarg)

        if attr is not None:
            config_kwargs[kwarg] = attr

    # Configuring the logger.
    logger_format = "[%(levelname)s] %(asctime)s : %(filename)s Line-%(lineno)s @%(message)s"
    logging.basicConfig(filename=__name__, filemode="w", level=logging.DEBUG, format=logger_format)

    logger_name = args.logger
    if logger_name is not None:
        cli_logger = logging.getLogger(logger_name)
        config_kwargs["logger"] = cli_logger
    else:
        cli_logger = logging.getLogger(__name__)

    # isolate the device argument.
    device = args.device
    if device is None:
        device = "cpu"

    device = torch.device(device)
    torch.cuda.set_device(device)

    print("Loading model...")
    model = _MODELS[args.model]()
    model.eval()
    model = model.to(device)

    print("Loading dataset...")
    transform_image = model.transform_image
    transform_label = model.transform_label
    transform_target = model.transform_target

    if args.dataset == "cifar10":
        dataset = Cifar10Dataset(targets=True, transform=transform_image, transform_labels=transform_label,
                                 transform_targets=transform_target)
    else:
        dataset = ImageNetDataset(targets=True, transform=transform_image, transform_labels=transform_label,
                                  transform_targets=transform_target)

    print("Loading attack...")
    loss_f = nn.CrossEntropyLoss()

    attack_name = args.attack
    batch_size, goal, distance_metric = args.batch_size, args.goal, args.distance_metric

    # Isolate the attack initialization arguments.
    init_kwargs = {}

    benchmark = IterationBenchmark(attack_name, model, loss_f, batch_size, dataset, goal, distance_metric, iteration,
                                   device, **init_kwargs)

    print("Configuring attack...")
    benchmark.config(**config_kwargs)

    print("Running benchmark...")
    rs = benchmark.run(cli_logger)

    cli_logger.debug("Iteration Benchmark Attack ran successfully!\n")

    print("Saving benchmark results...")
    cli_logger.info("Saving the benchmark results main parameters : Benchmark type = 'Iteration Benchmark'")
    cli_logger.info("Attack method : {}, Goal : {}, Distance metric : {}".format(attack_name, goal, distance_metric))
    cli_logger.info("Cmdline used to run the benchmark : \n{}\n".format(" ".join(sys.argv[:])))

    cli_logger.debug("Keys of the results dict are : {}".format(rs.keys()))
    cli_logger.debug("Shape of the accuracy / distortion data for one iteration : {}".
                     format(len(rs["iteration_{}".format(iteration)][0])))
    cli_logger.debug("Example of the adversarial accuracy and distortion on adversarial data for one iteration: \n{}\n"
                     .format(rs["iteration_{}".format(iteration)]))

    save_path = args.save_path
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "iteration_results_{}.npy".format(attack_name))

    np.save(save_path, {
            "type": "iteration",
            "method": attack_name,
            "goal": goal,
            "distance_metric": distance_metric,
            "cmdline": " ".join(sys.argv[:]),
            "result": rs,
            })
