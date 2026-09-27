# --------------------------------------------
# MANIKIN-L dataset.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

import numpy as np
import random
import glob
import pickle
import torch
from torch.utils.data import Dataset

# male SMPL rest root joint J0 (shared with dataset_s pelvis_offset); SMPL rotates about it.
_J0 = (-0.0022, -0.2408, 0.0286)


def _rot_vecs(v, Ry):                                         # rotate last-dim-3 vectors
    return torch.matmul(v, Ry.to(v.dtype).t())


def _yaw_augment(input_signal, rotation_local_full, pos_pelvis_gt):
    """Per-sample heading (yaw) augmentation by a random angle. Rotated: global 6D rotations, positions, position
    velocities, GT root orient/trans. NOT rotated: rotation velocities (heading-invariant) and
    local joint rotations."""
    ang = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(ang), np.sin(ang)
    Ry = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

    T = input_signal.shape[0]
    rot = _rot_vecs(input_signal[:, 0:132].reshape(T, 22, 2, 3), Ry).reshape(T, 132)   # 22 joints x 6D
    pos = _rot_vecs(input_signal[:, 264:330].reshape(T, 22, 3), Ry).reshape(T, 66)
    velpos = _rot_vecs(input_signal[:, 330:396].reshape(T, 22, 3), Ry).reshape(T, 66)
    inp = torch.cat([rot, input_signal[:, 132:264], pos, velpos], dim=1)               # velrot kept

    g = rotation_local_full.clone()
    g[:, 0:6] = _rot_vecs(g[:, 0:6].reshape(T, 2, 3), Ry).reshape(T, 6)                # root orient only

    J0 = torch.tensor(_J0, dtype=pos_pelvis_gt.dtype)
    pel = _rot_vecs(pos_pelvis_gt + J0, Ry) - J0
    return inp, g, pel


class AMASS_Dataset(Dataset):
    """Motion Capture dataset"""
    def __init__(self, opt):
        self.opt = opt
        self.window_size = opt['window_size']
        self.batch_size = opt['dataloader_batch_size']

        phase = self.opt['phase']
        dataroot = opt['dataroot']
        dataset_type = opt['dataset_type']
        assert dataset_type in ['amass_mixed', 'cross_cmu', 'cross_bml', 'cross_hdm05']

        held = {'cross_cmu': 'CMU', 'cross_bml': 'BioMotionLab_NTroje', 'cross_hdm05': 'MPI_HDM05'}

        if dataset_type == 'amass_mixed':
            self.filename_list = glob.glob(f'{dataroot}/*/{phase}/*.pkl')
        else:                                                # cross-dataset hold-out
            if phase == 'train':                             # train on the other two datasets (all their data)
                self.filename_list = []
                for k, ds in held.items():
                    if k != dataset_type:
                        self.filename_list += glob.glob(f'./{dataroot}/{ds}/*/*.pkl')
            else:                                            # test on the held-out dataset
                self.filename_list = glob.glob(f'./{dataroot}/{held[dataset_type]}/test/*.pkl')

        print('-------------------------------number of {} data is {}'.format(phase, len(self.filename_list)))


    def __len__(self):
        return max(len(self.filename_list), self.batch_size)


    def __getitem__(self, idx):
        filename = self.filename_list[idx]
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        if self.opt['phase'] == 'train':
            while data['rotation_local_full_gt_list'].shape[0] < self.window_size:
                idx = random.randint(0, idx)
                filename = self.filename_list[idx]
                with open(filename, 'rb') as f:
                    data = pickle.load(f)

        seq_len = data['hmd_position_global_full_gt_list'].shape[0]

        if self.opt['phase'] == 'train':
            start = np.random.randint(0, seq_len - self.window_size + 1)
            end = start + self.window_size
            input_signal = data['hmd_position_global_full_gt_list'][start:end, ...].reshape(self.window_size, -1).float()
            rotation_local_full = data['rotation_local_full_gt_list'][start:end, ...].reshape(self.window_size, -1).float()
            pos_pelvis_gt = data['body_parms_list']['trans'][start:end]
            if self.opt.get('heading_augment'):              # cross-dataset training
                input_signal, rotation_local_full, pos_pelvis_gt = _yaw_augment(input_signal, rotation_local_full, pos_pelvis_gt)
            return {'input_signal': input_signal,
                    'rotation_local_full': rotation_local_full,
                    'body_param_list': 0,
                    'global_head_trans': 0,
                    'pos_pelvis_gt': pos_pelvis_gt,
                    'floor_height': data['offset_floor_height'],
                    'foot_contact': data['contacts'][:, [10, 11]][start:end] # left, right
                    }
        else:
            return {'input_signal': data['hmd_position_global_full_gt_list'].reshape(seq_len, -1).float(),
                    'rotation_local_full': data['rotation_local_full_gt_list'],
                    'body_param_list': data['body_parms_list'],
                    'global_head_trans':data['head_global_trans_list'],
                    'pos_pelvis_gt':data['body_parms_list']['trans'],
                    'floor_height': data['offset_floor_height'],
                    'foot_contact': data['contacts'][:, [10, 11]], # left, right
                    'filepath': data['filepath']
                    }

