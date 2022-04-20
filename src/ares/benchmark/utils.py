"""Utils for the ares.benchmark module.

Used to facilitate loading attack modules.
"""

import inspect

from src.ares.attack import BIM, CW, DeepFool, FGSM, NAttack, NES, PGD, SPSA


_ATTACKS = {
        "bim": BIM,
        "cw": CW,
        "deepfool": DeepFool,
        "fgsm": FGSM,
        "nattack": NAttack,
        "nes": NES,
        "pgd": PGD,
        "spsa": SPSA,
        }


def load_attack(attack_name, init_kwargs):
    """Loads an attack module using its name. Checks the required initialization parameters and imports them from
    'init_kwargs'. 'init_kwargs' is automatically filled when creating a benchmark instance.

    :param attack_name: The name of the attack module. Valid values are : 'fgsm', 'bim'.
    :param init_kwargs: Keyword arguments to initialize the attack module.
    :return: The loaded attack module, an 'Attack' or 'BatchAttack' module.
    """

    kwargs = {}
    attack_class = _ATTACKS[attack_name]

    sig = inspect.signature(attack_class.__init__)
    for name in sig.parameters:
        if name != "self" and name in init_kwargs:
            kwargs[name] = init_kwargs[name]

    return attack_class(**kwargs)
