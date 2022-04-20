"""Utils for the ares module.

Used to facilitate download resources from an url and accessing resources locally.
"""

import os
import ssl
from urllib.request import urlretrieve

from tqdm import tqdm


context = ssl._create_unverified_context
ssl._create_default_https_context = context


def get_res_path(path):
    """Get the local resources' full path. By default, they are located under "~/.ares/". This location can be
    overridden by setting the "ARES_RES_DIR" environment variable as the desired folder.
    """

    prefix = os.environ.get("ARES_RES_DIR")
    if prefix is None:
        prefix = os.path.expanduser("~/.ares/")

    return os.path.abspath(os.path.join(prefix, path))


def download_res(url, path, show_progress_bar=True):
    """Download the resources from the url and save it to the specified path. If "show_progress_bar" is True, a progress
    bar will be displayed to track the download process.
    """

    hook = None if not show_progress_bar else _download_res_tqdm_hook(tqdm(unit="B", unit_scale=True))
    urlretrieve(url, path, hook)


def _download_res_tqdm_hook(pbar):
    """Wrapper for tqdm progress bar.
    """

    downloaded = [0]

    def update(count, block_size, total_size):
        if total_size is not None:
            pbar.total = total_size

        delta = count * block_size - downloaded[0]
        downloaded[0] = count * block_size
        pbar.update(delta)

    return update
