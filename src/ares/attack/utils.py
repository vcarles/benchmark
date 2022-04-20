"""Utils for the ares.attack module.

Used during the 'batch_attack()' methods.
"""

import torch


def maybe_to_tensor(x, target_len):
    if isinstance(x, float):
        return torch.repeat_interleave(torch.tensor(x, dtype=torch.float), target_len)
    else:
        return x


def tensor_unit(xs):
    xs_l2 = torch.linalg.vector_norm(xs, ord=2)
    return xs / xs_l2


def norm_clamp(update, x_og, x_adv, distance_metric, magnitude):
    adv_delta = x_adv + update - x_og
    if distance_metric == "l_2":
        norm = torch.linalg.vector_norm(adv_delta, ord=2)
        if norm > magnitude:
            adv_delta = magnitude * adv_delta / norm
    else:
        adv_delta = torch.clamp(adv_delta, -magnitude, magnitude)
    return adv_delta
