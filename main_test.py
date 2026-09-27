#!/usr/bin/env python
# --------------------------------------------
# Evaluation entry point.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Adapted from AvatarPoser and EgoPoser.
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

"""MANIKIN evaluation (both backbones, one evaluator).

    python main_test.py -opt options/manikin_s_cross_cmu.yaml     --checkpoint model_zoo/manikin_s_cross_cmu.pth --gpu 0
    python main_test.py -opt options/manikin_l_amass_eval_s2s.yaml --checkpoint model_zoo/manikin_l_amass.pth   --gpu 0 --benchmark amass_mixed

Scores the swivel reconstruction (MANIKIN-S: position_swivel; MANIKIN-L: position).
"""
import os, sys, argparse, logging
from collections import defaultdict
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_option as option
from utils.utils_metric import penetration_error, floating_error, skating_error
from data.select_dataset import define_Dataset
from models.select_model import define_Model

UPPER = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
LOWER = [1, 2, 4, 5, 7, 8, 10, 11]
HANDS = [20, 21]
FPS = 60


def _dist(p, g):                      # (T,22) per-joint L2 in cm
    return torch.sqrt(torch.sum(torch.square(g - p), dim=-1)) * 100



def evaluate(model, test_loader, opt, logger=None, nseq=10 ** 9, save_pred=False, tag=''):
    """Score every sequence; return {metric: mean over sequences}."""
    log = (logger.info if logger else print)
    err = defaultdict(list)
    n = 0
    for index, data in enumerate(test_loader):
        model.feed_data(data, test=True)
        model.test()
        P, G = model.current_prediction(), model.current_gt()
        log('testing the sample {}/{}'.format(index, len(test_loader)))
        gt = G['position'].reshape(-1, 22, 3)
        key = 'position_swivel' if 'position_swivel' in P else 'position'
        p = P[key].reshape(-1, 22, 3)[:gt.shape[0]]
        d = _dist(p, gt)
        err['MPJPE'].append(torch.nanmean(d).item())
        err['U-PE'].append(d[:, UPPER].mean().item())
        err['L-PE'].append(d[:, LOWER].mean().item())
        err['H-PE'].append(d[:, HANDS].mean().item())
        vel = ((p[1:] - p[:-1]) - (gt[1:] - gt[:-1])) * FPS
        err['MPJVE'].append((torch.sqrt(torch.sum(torch.square(vel), dim=-1)) * 100).nanmean().item())
        if 'floor_height' in G:
            pen = penetration_error(p, G['floor_height']) * 100
            flo = floating_error(p, G['floor_height']) * 100
            err['Penetration'].append(float(pen)); err['Floating'].append(float(flo))
            err['Skate'].append(float(skating_error(p, gt) * 100))
        if save_pred:
            out = os.path.join(opt['path']['root'], opt['task'], 'pred_npz'); os.makedirs(out, exist_ok=True)
            T = gt.shape[0]
            aa = torch.cat([P['root_orient'].reshape(-1, 3), P['pose_body'].reshape(-1, 63)], -1)[:T]
            np.savez(os.path.join(out, f'{index}_pred.npz'), poses=aa.detach().cpu().numpy(),     # AMASS format
                     trans=P['trans'].reshape(-1, 3)[:T].detach().cpu().numpy(), betas=np.zeros(16),
                     gender='male', mocap_framerate=FPS)
        # per-sequence lines
        log('length of the sequence: {}'.format(gt.shape[0]))
        log('positional error MPJPE: {}'.format(err['MPJPE'][-1]))
        log('velocity error MPJVE: {}'.format(err['MPJVE'][-1]))
        log('Hand Error: {}'.format(err['H-PE'][-1]))
        log('Upper Error: {}'.format(err['U-PE'][-1]))
        log('Lower Error: {}'.format(err['L-PE'][-1]))
        if 'Penetration' in err: log('Penetration: {}'.format(err['Penetration'][-1]))
        n += 1
        if n >= nseq:
            break
    avg = {k: float(np.mean(v)) for k, v in err.items() if len(v)}
    log(f'[{tag}] {n} sequences: '
        + ', '.join(f'{k}:{v:.4f}' for k, v in avg.items()))
    return avg


def load_model(opt, checkpoint, gpu=0):
    """Build model and load weights on physical GPU `gpu` (both backbones)."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)          # re-assert before CUDA init (parse() set it from gpu_ids)
    opt['gpu_ids'] = [0]; opt['dist'] = False
    opt['path']['pretrained_netG'] = checkpoint; opt['path']['pretrained'] = checkpoint
    opt = option.dict_to_nonedict(opt)
    model = define_Model(opt)
    model.init_test()
    return model, opt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-opt', '--opt', type=str, required=True, help='options/manikin_s_*.yaml (MANIKIN-S) or options/manikin_l_*.yaml (MANIKIN-L)')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--gpu', type=int, default=0, help='physical GPU')
    p.add_argument('--benchmark', type=str, default='amass_mixed', help='MANIKIN-L eval set: amass_mixed | cross_cmu | cross_bml | cross_hdm05')
    p.add_argument('--task', type=str, default=None, help='run name (logs go to results/<task>)')
    p.add_argument('--nseq', type=int, default=10 ** 9, help='stop after N sequences (debug)')
    p.add_argument('--save_pred', action='store_true', help='save the predicted motion per sequence (AMASS npz format)')
    args = p.parse_args()
    if args.task is None: args.task = 'eval_' + os.path.splitext(os.path.basename(args.opt))[0]
    opt = option.parse(args.opt, args, is_train=False)
    model, opt = load_model(opt, args.checkpoint, gpu=args.gpu)
    log_dir = opt['path']['log']; os.makedirs(log_dir, exist_ok=True)
    utils_logger.logger_info('test', os.path.join(log_dir, 'test.log')); logger = logging.getLogger('test')
    test_set = define_Dataset(opt['datasets']['test'])
    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
    avg = evaluate(model, loader, opt, logger, nseq=args.nseq, save_pred=args.save_pred, tag=opt['task'])
    keys = [k for k in ['MPJPE', 'U-PE', 'L-PE', 'H-PE', 'MPJVE', 'Penetration', 'Floating', 'Skate'] if k in avg]
    print('\n' + ' '.join(f'{k:>12s}' for k in keys)); print(' '.join(f'{avg[k]:12.3f}' for k in keys))
    print('done', flush=True)


if __name__ == '__main__':
    main()
