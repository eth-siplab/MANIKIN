# --------------------------------------------
# Loss functions.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

import torch


def velocityLoss(loss_func, pred, gt, interval=1):
    # pred/gt: (batch, seq_len, joint_num, 3) -> loss on frame-to-frame joint velocity
    batch, seq_len = pred.shape[0], pred.shape[1]
    pred = pred.reshape(batch, seq_len, -1)
    gt = gt.reshape(batch, seq_len, -1)
    target_vel = gt[:, interval::interval, :22*3] - gt[:, :-interval:interval, :22*3]
    pred_vel = pred[:, interval::interval, :22*3] - pred[:, :-interval:interval, :22*3]
    return loss_func(target_vel, pred_vel)


def footContactLoss(loss_func, pred, gt):
    # Penalise motion of the ankle/toe joints on frames where the GT foot is static (contact).
    batch, seq_len = pred.shape[0], pred.shape[1]
    pred = pred.reshape(batch, seq_len, -1)
    gt = gt.reshape(batch, seq_len, -1)
    pred = pred[:, :, :22*3].reshape(batch, seq_len, 22, 3)
    gt = gt[:, :, :22*3].reshape(batch, seq_len, 22, 3)

    relevant_joints = [7, 10, 8, 11]                          # L/R ankle and toe
    gt_joint_xyz = gt[:, :, relevant_joints, :]
    gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, 1:, :, :] - gt_joint_xyz[:, :-1, :, :], dim=-1)
    fc_mask = torch.unsqueeze((gt_joint_vel <= 0.01), dim=-1).repeat(1, 1, 1, 3)
    pred_joint_xyz = pred[:, :, relevant_joints, :]
    pred_vel = pred_joint_xyz[:, 1:, :, :] - pred_joint_xyz[:, :-1, :, :]
    pred_vel[~fc_mask] = 0
    return loss_func(torch.zeros(pred_vel.shape, device=pred_vel.device), pred_vel)


def penetrationLoss(pred_mesh, floor_height):
    # pred_mesh: (batch*seq, v_num, 3) -> penalise vertices below the floor
    seq_len = pred_mesh.shape[0] // floor_height.shape[0]
    floor_height = floor_height[:, None].repeat(1, seq_len).view(-1).float()
    lowest_z = pred_mesh.min(1)[0][:, 2]
    lowest_z_filtered = torch.where(lowest_z >= floor_height, torch.zeros_like(lowest_z), lowest_z)
    floor_height_filtered = torch.where(lowest_z >= floor_height, torch.zeros_like(floor_height), floor_height)
    return torch.abs(floor_height_filtered - lowest_z_filtered).mean()


def footHeightLoss(pred_mesh, floor_height, foot_contact):
    # Keep the toes at floor height on contact frames.
    seq_len = pred_mesh.shape[0] // floor_height.shape[0]
    floor_height = floor_height[:, None].repeat(1, seq_len).view(-1).float()[:, None].repeat(1, 2)
    foot_height = pred_mesh[:, [10, 11], 2]
    foot_contact = foot_contact.reshape(-1, 2)
    return (torch.abs(foot_height - floor_height) * foot_contact).mean()
