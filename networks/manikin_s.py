# --------------------------------------------
# MANIKIN-S network.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

import torch
import torch.nn as nn


# MANIKIN-S: a compact three-layer transformer backbone following AvatarPoser (Jiang et al., ECCV 2022),
# with the output head replaced by the swivel parameterisation.
#   Input  : B*40*54 (3 trackers x 18)
#   Output : root orientation (6D), torso + limb-root rotations (6D, other joints identity),
#            swivel angles (cos/sin x 4 limbs), ankle positions and foot-bone vectors


class ManikinNet(nn.Module):
    def __init__(self, input_dim, num_layer, embed_dim, nhead, rotation_keep_joints, dropout=0.1):
        super(ManikinNet, self).__init__()

        self.linear_embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)

        # rotations are predicted for the torso and limb-root joints only; the others are filled with identity
        self.keep_joints = list(rotation_keep_joints)
        self.register_buffer('ident_pose', torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(21))
        self.register_buffer('keep_slots', torch.tensor([j - 1 for j in self.keep_joints]))

        def _head(out_dim):
            return nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, out_dim))

        self.stabilizer = _head(6)                                     # root orientation
        self.swivel_decoder = _head(8)                                 # swivel cos(4) | sin(4)
        self.joint_rotation_decoder = _head(6 * len(self.keep_joints))
        self.ankle_position_decoder = _head(12)                        # ankle position (6) | foot bone (6)

    def forward(self, input_tensor):
        x = self.linear_embedding(input_tensor)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)[:, -1]
        root_orient = self.stabilizer(x)
        raw_rot = self.joint_rotation_decoder(x)
        swivel = self.swivel_decoder(x)
        ankle = self.ankle_position_decoder(x)
        B = x.shape[0]
        pose_body = self.ident_pose.unsqueeze(0).expand(B, -1).clone().reshape(B, 21, 6)
        pose_body[:, self.keep_slots, :] = raw_rot.reshape(B, len(self.keep_joints), 6)
        return root_orient, pose_body.reshape(B, 126), swivel, ankle
