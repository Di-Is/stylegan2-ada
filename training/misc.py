import glob
import os
import re

from pathlib import Path

import numpy as np
import shutil

def get_parent_dir(run_dir):
    out_dir = Path(run_dir).parent

    return out_dir

def locate_latest_pkl(out_dir):
    all_pickle_names = sorted(glob.glob(os.path.join(out_dir, '0*', 'network-*.pkl')))

    try:
        latest_pickle_name = all_pickle_names[-1]
    except IndexError:
        latest_pickle_name = None

    return latest_pickle_name

def parse_kimg_from_network_name(network_pickle_name):

    if network_pickle_name is not None:
        resume_run_id = os.path.basename(os.path.dirname(network_pickle_name))
        RE_KIMG = re.compile('network-snapshot-(\d+).pkl')
        try:
            kimg = int(RE_KIMG.match(os.path.basename(network_pickle_name)).group(1))
        except AttributeError:
            kimg = 0.0
    else:
        kimg = 0.0

    return float(kimg)

def load_augment_strength(network_pickle_name):
    if network_pickle_name is not None:
        dirpath = os.path.dirname(network_pickle_name)
        aug_strength_path = "{}/augment.dat".format(dirpath)

        try:
            aug_strength = float(np.loadtxt(aug_strength_path))
        except:
            aug_strength = 0.0
        print(aug_strength)
    else:
        aug_strength = 0.0

    return aug_strength

def save_augment_strength(network_pickle_name, aug_strength):

    if network_pickle_name is not None:
        dirpath = os.path.dirname(network_pickle_name)

        aug_strength_path = "{}/augment.dat".format(dirpath)
        np.savetxt(aug_strength_path, np.array([aug_strength], dtype=float))

    else:
        pass

def copy_argument_strength(old_pkl_path, new_pkl_path):

    old_dirpath = os.path.dirname(old_pkl_path)
    new_dirpath = os.path.dirname(new_pkl_path)

    old_fpath = '{}/augment.dat'.format(old_dirpath)
    new_fpath = '{}/augment.dat'.format(new_dirpath)

    shutil.copyfile(old_fpath, new_fpath)
