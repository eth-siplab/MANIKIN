# --------------------------------------------
# MANIKIN-S dataset.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

import torch
import numpy as np
import os
import glob
import pickle
import random
from torch.utils.data import Dataset


def _to_tracker_format(data):
    """Adapt a 22-joint MANIKIN pickle to 3 trackers x 18 feats (head, L/R hand), plus body params."""
    hmd = data['hmd_position_global_full_gt_list']
    J = [15, 20, 21]; T = hmd.shape[0]
    r = hmd[:, 0:132].reshape(T, 22, 6)[:, J].reshape(T, 18); rv = hmd[:, 132:264].reshape(T, 22, 6)[:, J].reshape(T, 18)
    p = hmd[:, 264:330].reshape(T, 22, 3)[:, J].reshape(T, 9); pv = hmd[:, 330:396].reshape(T, 22, 3)[:, J].reshape(T, 9)
    data = dict(data); data['hmd_position_global_full_gt_list'] = torch.cat([r, rv, p, pv], -1)
    bp = {}
    for k, v in data['body_parms_list'].items():
        v = torch.as_tensor(v)
        bp[k] = torch.cat([v[0:1], v], 0) if v.dim() >= 1 and v.shape[0] == T else v
    data['body_parms_list'] = bp
    return data


class AMASS_Dataset(Dataset):
    """Motion Capture dataset"""

    def __init__(self, opt):
        self.opt = opt
        self.window_size = opt['window_size']
        self.batch_size = opt['dataloader_batch_size']
        dataroot_list = opt['dataroot']

        # heading_rotate_velocity: input dims [18:36] are body-frame rotation velocities and are
        # heading-invariant; set false so the heading augmentation leaves them unchanged.
        self.heading_rotate_velocity = opt.get('heading_rotate_velocity', True)

        self.filename_list = []
        for dataroot in dataroot_list:
            self.filename_list += glob.glob(os.path.join(dataroot, '*.pkl'))
        if self.opt['phase'] != 'train':
            self.filename_list.sort()
            print('-------------------------------number of test data is {}'.format(len(self.filename_list)))

    def __len__(self):

        return max(len(self.filename_list), self.batch_size)

    def __getitem__(self, idx):

        filename = self.filename_list[idx]
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        if self.opt['phase'] == 'train':
            while data['rotation_local_full_gt_list'].shape[0] <= self.window_size:
                idx = random.randint(0,idx)
                filename = self.filename_list[idx]
                with open(filename, 'rb') as f:
                    data = pickle.load(f)

        floor = data['min_foot_height']
        data = _to_tracker_format(data)
        rotation_local_full_gt_list = data['rotation_local_full_gt_list']
        hmd_position_global_full_gt_list = data['hmd_position_global_full_gt_list']
        body_parms_list = data['body_parms_list']
        head_global_trans_list = data['head_global_trans_list']

        # SMPL pelvis offset (mean-shape T-pose, constant)
        pelvis_offset = torch.tensor([-0.0022, -0.2408, 0.0286], dtype=torch.float32)

        def apply_heading_rotation(R_h, input_hmd, output_gt, head_trans, pelvis_trans):
            """Apply heading rotation; SMPL FK rotates around pelvis, so trans needs compensation."""
            def rot_sixd(sixd):
                return torch.cat([(R_h @ sixd[:, :3].T).T, (R_h @ sixd[:, 3:6].T).T], dim=-1)

            inp = input_hmd.clone()
            # [0:18] orientations rotate with the world. [18:36] body-frame rotation velocities are
            # heading-invariant; rotated only when heading_rotate_velocity is set.
            rot_blocks = [0, 6, 12] + ([18, 24, 30] if self.heading_rotate_velocity else [])
            for s in rot_blocks:
                inp[:, s:s+6] = rot_sixd(inp[:, s:s+6])
            for s in [36, 39, 42, 45, 48, 51]:                # positions and velocities rotate
                inp[:, s:s+3] = (R_h @ inp[:, s:s+3].T).T

            out = output_gt.clone()
            out[:, :6] = rot_sixd(out[:, :6])                 # rotate GT root orient

            R_h_4x4 = torch.eye(4); R_h_4x4[:3, :3] = R_h
            ht = R_h_4x4 @ head_trans

            p = pelvis_offset.to(pelvis_trans.device)         # pelvis trans + SMPL offset compensation
            pt = (R_h @ pelvis_trans.T).T - p + (R_h @ p)

            return inp, out, ht, pt

        if self.opt['phase'] == 'train':

            frame = np.random.randint(hmd_position_global_full_gt_list.shape[0] - self.window_size)
            input_hmd  = hmd_position_global_full_gt_list[frame:frame + self.window_size,...].reshape(self.window_size, -1).float()
            input_hmd = input_hmd.clone()
            input_hmd[:, [38, 41, 44]] -= floor               # world-z of head/lhand/rhand

            output_gt = rotation_local_full_gt_list[frame:frame + self.window_size,...].float()
            head_trans_window = head_global_trans_list[frame:frame + self.window_size,...]
            pelvis_trans_window = body_parms_list['trans'][frame+1:frame + self.window_size+1,...]

            # heading (yaw) augmentation
            rand_angle = np.random.uniform(-np.pi, np.pi)
            cos_a, sin_a = np.cos(rand_angle), np.sin(rand_angle)
            R_aug = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=torch.float32)
            input_hmd, output_gt, head_trans_window, pelvis_trans_window = apply_heading_rotation(
                R_aug, input_hmd, output_gt, head_trans_window, pelvis_trans_window)

            return {'L': input_hmd,
                    'H': output_gt,
                    'Head_trans_global':head_trans_window[[-1],...],
                    'pos_pelvis_gt':pelvis_trans_window[[-1],...],
                    }

        else:

            input_hmd  = hmd_position_global_full_gt_list.reshape(hmd_position_global_full_gt_list.shape[0], -1)[1:].float()
            input_hmd = input_hmd.clone()
            input_hmd[:, [38, 41, 44]] -= floor               # world-z of head/lhand/rhand, as in training

            return {'L': input_hmd,
                    'H': rotation_local_full_gt_list[1:].float(),
                    'Head_trans_global':head_global_trans_list[1:],
                    'pos_pelvis_gt':body_parms_list['trans'][2:],
                    }
