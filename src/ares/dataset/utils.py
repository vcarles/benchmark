"""Utils for the ares.dataset module.

Used to facilitate the creation of an iterator for a dataset.
"""

from torch.utils.data import DataLoader


def dataset_to_iterator(dataset, batch_size):
    """A wrapper to get the dataset as an iterator for a given batch size.
    """

    return DataLoader(dataset, batch_size=batch_size)
