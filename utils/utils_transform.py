# --------------------------------------------
# Rotation and transform utilities.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

from torch.nn import functional as F
from human_body_prior.tools import tgm_conversion as tgm
from human_body_prior.tools.rotation_tools import aa2matrot, matrot2aa

import torch


def bgs(d6s):
    d6s = d6s.reshape(-1, 2, 3).permute(0, 2, 1)
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:,:,0], p=2, dim=1)
    a2 = d6s[:,:,1]
    c = torch.bmm(b1.view(bsz,1,-1),a2.view(bsz,-1,1)).view(bsz,1)*b1
    b2 = F.normalize(a2-c,p=2,dim=1)
    b3=torch.cross(b1,b2,dim=1)
    return torch.stack([b1,b2,b3],dim=-1)

def matrot2sixd(pose_matrot):
    ''' Nx3x3 -> Nx6 '''
    pose_6d = torch.cat([pose_matrot[:,:3,0], pose_matrot[:,:3,1]], dim=1)
    return pose_6d


def aa2sixd(pose_aa):
    ''' Nx3 -> Nx6 '''
    pose_matrot = aa2matrot(pose_aa)
    pose_6d = matrot2sixd(pose_matrot)
    return pose_6d

def sixd2matrot(pose_6d):
    ''' Nx6 -> Nx3x3, Gram-Schmidt orthonormalised '''
    return bgs(pose_6d)

def sixd2aa(pose_6d, batch = False):
    ''' Nx6 -> Nx3 '''
    if batch:
        B,J,C = pose_6d.shape
        pose_6d = pose_6d.reshape(-1,6)
    pose_matrot = sixd2matrot(pose_6d)
    pose_aa = matrot2aa(pose_matrot)
    if batch:
        pose_aa = pose_aa.reshape(B,J,3)
    return pose_aa

def sixd2quat(pose_6d):
    ''' Nx6 -> Nx4 quaternion '''
    pose_mat = sixd2matrot(pose_6d)
    pose_mat_34 = torch.cat((pose_mat, torch.zeros(pose_mat.size(0), pose_mat.size(1), 1)), dim=-1)
    pose_quaternion = tgm.rotation_matrix_to_quaternion(pose_mat_34)
    return pose_quaternion

def quat2aa(pose_quat):
    ''' Nx4 -> Nx3 '''
    return tgm.quaternion_to_angle_axis(pose_quat)

def aa2quat(aa):
    ''' Nx3 -> Nx4 '''
    return tgm.angle_axis_to_quaternion(aa)


def matrot2mat4x4(R, t):
    ''' Batch transform matrices from R (BxJx3x3) and t (BxJx3) -> T (BxJx4x4). '''
    return torch.cat([F.pad(R, [0, 0, 0, 1, 0, 0]),
                      F.pad(t[:, :, :, None], [0, 0, 0, 1, 0, 0], value=1)], dim=-1)


def local2global_pose(local_pose, kintree):
    """Local -> global joint rotations along the kinematic tree. List-based (no in-place writes) so it's
    autograd-safe. `local_pose` (B,J,9) or (B,J*9) -> (B,J,3,3)."""
    bs = local_pose.shape[0]
    lp = local_pose.reshape(bs, -1, 3, 3)
    g = [lp[:, j] for j in range(lp.shape[1])]
    for j in range(lp.shape[1]):
        p = int(kintree[j])
        if p >= 0:
            g[j] = torch.matmul(g[p], g[j])
    return torch.stack(g, dim=1)


def fk_module(global_orientation, joint_rotation, body_model):
    """Forward kinematics: 6D root orient + 6D joint rotations -> joint positions (SMPL Jtr)."""
    global_orientation = sixd2aa(global_orientation.reshape(-1, 6)).reshape(global_orientation.shape[0], -1).float()
    joint_rotation = sixd2aa(joint_rotation.reshape(-1, 6)).reshape(joint_rotation.shape[0], -1).float()
    body_pose = body_model(**{'pose_body': joint_rotation, 'root_orient': global_orientation})
    return body_pose.Jtr
