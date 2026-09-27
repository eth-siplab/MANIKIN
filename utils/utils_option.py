# --------------------------------------------
# Config (YAML) loader.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

import os, yaml
from datetime import datetime
import re, glob


class NoneDict(dict):
    def __missing__(self, key):
        return None


def parse(opt_path, args=None, is_train=True):
    """Load a yaml config. args (optional): task / benchmark / checkpoint overrides from the command line."""
    with open(opt_path, 'r') as f:
        opt = yaml.safe_load(f)
    opt['opt_path'] = opt_path
    opt['is_train'] = is_train

    if args is not None:
        if getattr(args, 'task', None):
            opt['task'] = args.task
        if opt['model'] == 'manikin_l' and getattr(args, 'benchmark', None):   # amass_mixed | cross_cmu | cross_bml | cross_hdm05
            for phase in ('train', 'test'):
                opt['datasets'][phase]['dataset_type'] = args.benchmark
                opt['datasets'][phase]['dataroot'] = './datasets'
            if args.benchmark in ('cross_cmu', 'cross_bml', 'cross_hdm05'):   # cross-dataset: train with heading augmentation
                opt['datasets']['train']['heading_augment'] = True
        if getattr(args, 'checkpoint', None):
            opt['path']['pretrained_netG'] = args.checkpoint

    # defaults
    if 'merge_bn' not in opt:
        opt['merge_bn'] = False
        opt['merge_bn_startpoint'] = -1

    # datasets
    for phase, dataset in opt['datasets'].items():
        dataset['phase'] = phase.split('_')[0]

    # paths
    for key, path in opt['path'].items():
        if path:
            opt['path'][key] = os.path.expanduser(path)
    path_task = os.path.join(opt['path']['root'], opt['task'])
    opt['path']['task'] = path_task
    opt['path']['log'] = path_task
    opt['path']['options'] = os.path.join(path_task, 'options')
    opt['path']['pretrained'] = opt['path']['pretrained_netG']
    if is_train:
        opt['path']['models'] = os.path.join(path_task, 'models')

    # GPU devices
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    # DDP defaults
    if 'find_unused_parameters' not in opt:
        opt['find_unused_parameters'] = True
    if 'dist' not in opt:
        opt['dist'] = False
    opt['num_gpu'] = len(opt['gpu_ids'])
    print('number of GPUs is: ' + str(opt['num_gpu']))

    # optimizer / model-loading defaults
    if 'G_optimizer_reuse' not in opt['train']:
        opt['train']['G_optimizer_reuse'] = False
    if 'G_param_strict' not in opt['train']:
        opt['train']['G_param_strict'] = True

    return opt


def get_timestamp():
    return datetime.now().strftime('_%y%m%d_%H%M%S')

def find_last_checkpoint(save_dir, net_type='G'):
    """Return (init_iter, init_path) of the latest *_<net_type>.pth in save_dir; (0, None) if none."""
    file_list = glob.glob(os.path.join(save_dir, '*_{}.pth'.format(net_type)))
    if file_list:
        iter_exist = []
        for file_ in file_list:
            iter_current = re.findall(r"(\d+)_{}.pth".format(net_type), file_)
            iter_exist.append(int(iter_current[0]))
        init_iter = max(iter_exist)
        init_path = os.path.join(save_dir, '{}_{}.pth'.format(init_iter, net_type))
    else:
        init_iter = 0
        init_path = None
    return init_iter, init_path

def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def save(opt):
    opt_path = opt['opt_path']
    opt_path_copy = opt['path']['options']
    dirname, filename_ext = os.path.split(opt_path)
    filename, ext = os.path.splitext(filename_ext)
    dump_path = os.path.join(opt_path_copy, filename+get_timestamp()+ext)
    with open(dump_path, 'w') as dump_file:
        yaml.dump(opt, dump_file)
