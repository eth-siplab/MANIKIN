#!/usr/bin/env python
# --------------------------------------------
# Training entry point.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Adapted from AvatarPoser and EgoPoser.
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

"""MANIKIN training (both backbones, one loop).

    python main_train.py -opt options/manikin_s_amass.yaml                               # MANIKIN-S
    python main_train.py -opt options/manikin_l_amass.yaml --benchmark amass_mixed --task my_run   # MANIKIN-L (cross_cmu/cross_bml/cross_hdm05 = hold out that set)
"""
import os, sys, math, random, logging, argparse
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model


def main(opt, total_step=None):
    for key, path in opt['path'].items():
        if 'pretrained' not in key and isinstance(path, str):
            os.makedirs(path, exist_ok=True)
    current_step = 0
    if opt['datasets']['train'].get('resume'):                        # continue from the task's last checkpoint
        current_step, opt['path']['pretrained_netG'] = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    option.save(opt)
    opt = option.dict_to_nonedict(opt)
    utils_logger.logger_info('train', os.path.join(opt['path']['log'], 'train.log'))
    logger = logging.getLogger('train')
    seed = opt['train']['manual_seed'] or random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            nw = dataset_opt['dataloader_num_workers']
            logger.info('Number of train sequences: {:,d}, iters/epoch: {:,d}'.format(
                len(train_set), int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))))
            train_loader = DataLoader(train_set, batch_size=dataset_opt['dataloader_batch_size'], shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=nw, drop_last=True, pin_memory=True, persistent_workers=nw > 0)
        elif phase == 'test':
            continue                                                     # evaluation: main_test.py
        else:
            raise NotImplementedError('Phase [%s] is not recognized.' % phase)

    model = define_Model(opt)
    if opt.get('merge_bn') and current_step > opt['merge_bn_startpoint']:
        logger.info('merging bnorm'); model.merge_bnorm_test()
    model.init_train()
    if hasattr(model, 'info_params'):
        logger.info(model.info_params())
    total_step = total_step or opt['train'].get('total_step') or 10 ** 9
    epoch = 0
    while current_step < total_step:
        for train_data in train_loader:
            current_step += 1
            model.feed_data(train_data)
            model.optimize_parameters(current_step)
            model.update_learning_rate(current_step)
            if opt.get('merge_bn') and opt['merge_bn_startpoint'] == current_step:
                logger.info('merging bnorm'); model.merge_bnorm_train()
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()
                logger.info('<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                            + ' '.join('{:s}: {:.3e}'.format(k, v) for k, v in logs.items()))
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.'); model.save(current_step)
            if current_step >= total_step:
                break
        epoch += 1
    logger.info('Saving the final model.'); model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-opt', '--opt', type=str, required=True, help='options/manikin_s_*.yaml (MANIKIN-S) or options/manikin_l_*.yaml (MANIKIN-L)')
    p.add_argument('--task', type=str, default=None, help='run name (results/<task>)')
    p.add_argument('--benchmark', type=str, default=None, help='MANIKIN-L training set: amass_mixed | cross_cmu | cross_bml | cross_hdm05')
    p.add_argument('--total_step', type=int, default=None)
    args = p.parse_args()
    main(option.parse(args.opt, args, is_train=True), args.total_step)
