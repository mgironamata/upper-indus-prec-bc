import os
import shutil
import time
import string, random
import pdb

import slugify
import torch

__all__ =  ['generate_root',
            'save_checkpoint',
            'WorkingDirectory']

def generate_root(name, show_timestamp = False, show_label = True, label_name = None, root = '/data/hpcdata/users/marron31/'):
    """Generate a root path.

    Args:
        name (str): Name of the experiment.

    Returns:

    """
    
    now = time.strftime('%Y-%m-%d_%H-%M-%S')

    # Hack to remove the last part of the name for experiments where I include the predictors as params
    name = name.split('precip_norris')[0]
    
    if label_name is None:
        label = ''.join([random.choice(string.ascii_letters) for i in range(10)])
    else:
        label = label_name

    if show_timestamp and not(show_label):
        return os.path.join(root, '_experiments', f'{now}_{slugify.slugify(name)}')
    elif not(show_timestamp) and show_label: 
        return os.path.join(root, '_experiments', f'{label}_{slugify.slugify(name)}')
    elif show_timestamp and show_label:
        return os.path.join(root, '_experiments', f'{label}_{now}_{slugify.slugify(name)}')

def save_checkpoint(wd, state, is_best):
    """Save a checkpoint.

    Args:
        wd (:class:`.experiment.WorkingDirectory`): Working directory.
        state (dict): State to save.
        is_best (bool): This model is the best so far.
    """
    fn = wd.file('checkpoint.pth.tar')
    torch.save(state, fn,   )
    if is_best:
        fn_best = wd.file('model_best.pth.tar')
        shutil.copyfile(fn, fn_best)

class WorkingDirectory:
    """Working directory.

    Args:
        root (str): Root of working directory.
        override (bool, optional): Delete working directory if it already
            exists. Defaults to `False`.
    """

    def __init__(self, root, override=False):
        self.root = root

        # Delete if the root already exists.
        if os.path.exists(self.root) and override:
            print('Experiment directory already exists. Overwriting.')
            shutil.rmtree(self.root)

        print('Root:', self.root)

        # Create root directory.
        os.makedirs(self.root, exist_ok=True)

    def file(self, *name, exists=False):
        """Get the path of a file.

        Args:
            *name (str): Path to file, relative to the root directory. Use
                different arguments for directories.
            exists (bool): Assert that the file already exists. Defaults to
                `False`.

        Returns:
            str: Path to file.
        """
        path = os.path.join(self.root, *name)

        # Ensure that path exists.
        if exists and not os.path.exists(path):
            raise AssertionError('File "{}" does not exist.'.format(path))
        elif not exists:
            path_dir = os.path.join(self.root, *name[:-1])
            os.makedirs(path_dir, exist_ok=True)

        return path