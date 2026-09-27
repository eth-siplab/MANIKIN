#!/usr/bin/env python
# --------------------------------------------
# AMASS -> pickle data preprocessing.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

"""AMASS -> MANIKIN pickles (one pipeline for MANIKIN-S and MANIKIN-L).

    python prepare_data.py --root <AMASS root>     # CMU / BioMotionLab_NTroje / MPI_HDM05, split from datasets/data_split
"""
import os, sys, glob, pickle, argparse
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch
from human_body_prior.body_model.body_model import BodyModel
from utils.utils_transform import sixd2aa
from data.utils_data import process

FLOOR_JOINTS = [7, 8, 10, 11]                                                # ankles and feet


def add_min_foot_height(dst_dirs, body_model, device):
    """Add the floor height of each sequence (lowest ankle / foot z) to its pickle."""
    files = sorted(f for d in dst_dirs for f in glob.glob(os.path.join(d, '*.pkl')))
    for f in files:
        with open(f, 'rb') as fh:
            d = pickle.load(fh)
        rot = d['rotation_local_full_gt_list'].float().to(device)          # (T,132) 6D
        trans = d['body_parms_list']['trans'].float().to(device)           # (T,3)
        go = sixd2aa(rot[:, :6]).reshape(-1, 3)                            # global orientation (T,3)
        jr = sixd2aa(rot[:, 6:].reshape(-1, 6)).reshape(rot.shape[0], -1)   # joint rotations (T,63)
        with torch.no_grad():
            jp = body_model(root_orient=go, pose_body=jr).Jtr[:, :22]      # mean shape, root-relative
        n = min(jp.shape[0], trans.shape[0])                               # rot/trans can differ by one frame
        d['min_foot_height'] = (jp[:n, FLOOR_JOINTS, 2] + trans[:n, 2:3]).min().item()
        with open(f + '.tmp', 'wb') as fh:
            pickle.dump(d, fh)
        os.replace(f + '.tmp', f)
    print(f'min_foot_height: {len(files)} sequences', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, required=True, help='AMASS root containing the dataset folders')
    p.add_argument('--datasets', type=str, default='MPI_HDM05,BioMotionLab_NTroje,CMU')
    p.add_argument('--out', type=str, default='./datasets')
    p.add_argument('--support_data', type=str, default='./support_data')
    p.add_argument('--data_split', type=str, default='./datasets/data_split')
    a = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bms = {g: BodyModel(bm_fname=os.path.join(a.support_data, f'body_models/smplh/{g}/model.npz'), num_betas=16, num_dmpls=8,
                        dmpl_fname=os.path.join(a.support_data, f'body_models/dmpls/{g}/model.npz')).to(device) for g in ['male', 'female']}
    dst_dirs = []
    for ds in a.datasets.split(','):
        for phase in ['train', 'test']:
            split_file = os.path.join(a.data_split, ds, phase + '_split.txt')
            dst = os.path.join(a.out, ds, phase)
            process(os.path.join(a.root, ds), dst, bms, split_file if os.path.exists(split_file) else None)
            dst_dirs.append(dst)
    add_min_foot_height(dst_dirs, bms['male'], device)
