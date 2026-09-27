# --------------------------------------------
# MANIKIN-L network.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

import collections
import math
import warnings
import torch
import torch.nn as nn
from utils.utils_transform import sixd2matrot, matrot2sixd, sixd2aa, fk_module, local2global_pose
from networks import swivel as SR


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Fill tensor in-place with a truncated N(mean, std^2), values clamped to [a, b] (from PyTorch)."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


# MANIKIN-L: a coarse per-frame pose refined by a spatio-temporal transformer. The backbone follows
# AvatarJLM (Zheng et al., ICCV 2023) -- joint-level tokenization and coarse-to-fine refinement; its
# spatio-temporal transformer follows MixSTE (Zhang et al., CVPR 2022). MANIKIN replaces the output
# head with the swivel parameterisation and its analytic limb reconstruction.
#   Input  : B*T*22*18  (per joint: rot6 | rot-velocity6 | pos3 | pos-velocity3)
#   Output : per-frame swivel parameterisation (root6 + torso/limb-root rotations + swivel(8) + ankle/toe
#            offsets), turned into a full-body pose by the Analytic Arm/Leg Solver (networks/swivel.py).


class CoarsePoseNet(nn.Module):
    """Map the static per-frame embedding to a coarse SMPL pose (6D root + 21x6 body) and its FK positions."""
    def __init__(self, body_model, joint_regressor_dim=1024, embed_dim=1024):
        super(CoarsePoseNet, self).__init__()
        self.regressor = nn.Sequential(
                            nn.Linear(joint_regressor_dim, embed_dim),
                            nn.LeakyReLU(0.1),
                            nn.Linear(embed_dim, 22*6))
        self.body_model = body_model
    
    def forward(self, x):
        params = self.regressor(x)
        global_orientation = params[:, :, :6]
        joint_rotation = params[:, :, 6:]
        # Torso FK: forward kinematics on the predicted torso angles -> joint positions (incl. the base
        # joints, shoulder/hip, that the Analytic Arm/Leg Solver needs).
        joint_position = fk_module(global_orientation.reshape(-1, 6), joint_rotation.reshape(-1, 21*6), self.body_model)
        joint_position = joint_position[:, :22]
        return joint_position, params

class SpatioTemporalTransformer(nn.Module):
    """Mixed spatio-temporal encoder, following MixSTE (Zhang et al., CVPR 2022). `repeat_time` blocks,
    each a spatial encoder over joint tokens within a frame (STEblocks) then a temporal encoder over frames
    per joint (TTEblocks). Built on nn.TransformerEncoderLayer; made causal at eval via the padding mask."""
    def __init__(self, repeat_time=1, s_layer=2, t_layer=2, embed_dim=256, nhead=8):
        super(SpatioTemporalTransformer, self).__init__()
        self.num_layer = repeat_time
        self.STEblocks = nn.ModuleList()
        self.TTEblocks = nn.ModuleList()
        for _ in range(repeat_time):
            spatial_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead, batch_first=True)
            self.STEblocks.append(nn.TransformerEncoder(spatial_layer, num_layers=s_layer))
            temporal_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead, batch_first=True)
            self.TTEblocks.append(nn.TransformerEncoder(temporal_layer, num_layers=t_layer))

    def forward(self, feat, time_pad_mask=None):
        # time_pad_mask: (batch, seq_len) bool, True = padded frame. Only temporal blocks mix frames, so as
        # src_key_padding_mask it lets eval batch the streaming prefixes into one forward.
        batch, seq_len, joint_num, feat_dim = feat.shape
        kpm = None if time_pad_mask is None else time_pad_mask.repeat_interleave(joint_num, dim=0)   # (batch*joint_num, seq_len)
        # Last token = raw per-frame input feature. Held fixed and re-written after each block so the other
        # tokens attend to it as a constant conditioning signal.
        input_feature = feat[:, :, -1].clone().detach()
        for i in range(self.num_layer):
            feat = self.STEblocks[i](feat.reshape(batch*seq_len, joint_num, -1)).reshape(batch, seq_len, joint_num, -1)
            feat[:, :, -1] = input_feature
            feat = self.TTEblocks[i](feat.reshape(batch, seq_len, joint_num, -1).permute(0, 2, 1, 3).reshape(batch*joint_num, seq_len, -1), src_key_padding_mask=kpm).reshape(batch, joint_num, seq_len, -1).permute(0, 2, 1, 3)
            feat[:, :, -1] = input_feature
        return feat

class PoseRefiner(nn.Module):
    """Add positional embeddings, run the spatio-temporal transformer and regress the per-frame swivel
    parameterisation, which the Analytic Arm/Leg Solver (networks.swivel) turns into a full-body pose."""
    def __init__(self, body_model, keep_joints, num_layer=6, s_layer=1, t_layer=1, joint_feat_dim=256, node_num=22,
                 reg_hidden_dim=1024, nhead=8):
        super(PoseRefiner, self).__init__()
        self.body_model = body_model
        self.transformer = SpatioTemporalTransformer(repeat_time=num_layer, s_layer=s_layer, t_layer=t_layer, embed_dim=joint_feat_dim, nhead=nhead)
        max_seq_len = 200
        self.temp_embed = nn.Parameter(torch.zeros(1, max_seq_len, 1, joint_feat_dim))
        trunc_normal_(self.temp_embed, std=.02)
        self.joint_position_embed = nn.Parameter(torch.zeros(1, 1, node_num, joint_feat_dim))
        trunc_normal_(self.joint_position_embed, std=.02)

        # Regression head: root orientation, rotations of the torso and limb-root joints (keep_joints; the others
        # are identity), swivel angles, ankle/toe offsets, plus the full 21x6 rotations as an auxiliary training
        # target (unused at test). Knee/elbow flexion and limb-root twist come from the Analytic Arm/Leg Solver.
        self.keep_joints = list(keep_joints)
        out_dim = 6 + 6 * len(self.keep_joints) + 8 + 6 + 6 + 21 * 6   # root + rot + swivel + ankle + toe + aux
        self.register_buffer('ident_pose', torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(21))
        self.register_buffer('keep_slots', torch.tensor([j-1 for j in self.keep_joints]))
        self.register_buffer('J_rest', body_model().Jtr[0, :22].detach())
        self.parents = body_model.kintree_table[0][:22].long().tolist()
        self.register_buffer('kintree22', body_model.kintree_table[0][:22].long())
        self.param_regressor = nn.Sequential(
                            nn.Linear(joint_feat_dim * node_num, reg_hidden_dim),
                            nn.GroupNorm(8, reg_hidden_dim),
                            nn.LeakyReLU(0.1),
                            nn.Linear(reg_hidden_dim, out_dim)
            )


    def anchor_to_head(self, joint_position, head_position):
        head2root = joint_position[:, :, 15].clone()
        root_transl = head_position - head2root                       # head on the tracked head
        global_position = root_transl[:, :, None] + joint_position
        return global_position, root_transl


    def forward(self, tokens, head_position, wrist_world, time_pad_mask=None):
        batch, seq_len = tokens.shape[0], tokens.shape[1]
        tokens = tokens + self.joint_position_embed                    # spatial positional embedding
        tokens = tokens + self.temp_embed[:,:seq_len,:,:]              # temporal positional embedding
        refined_feat = self.transformer(tokens, time_pad_mask=time_pad_mask)
        param = self.param_regressor(refined_feat.reshape(batch * seq_len, -1)).reshape(batch, seq_len, -1)
        return self._analytic_limb_solver(param, head_position, wrist_world, batch, seq_len)

    def _analytic_limb_solver(self, param, head_position, wrist_world, batch, seq_len):
        """Reconstruct full-body positions from the swivel parameterisation (see networks/swivel.py)."""
        BT = batch * seq_len
        p = param.reshape(BT, -1)
        nkeep = len(self.keep_joints)
        root6 = p[:, :6]
        raw_rot = p[:, 6:6 + 6*nkeep]
        swivel = p[:, 6 + 6*nkeep: 6 + 6*nkeep + 8]                 # cos(4) | sin(4)
        ankle_off = p[:, 6 + 6*nkeep + 8: 6 + 6*nkeep + 14]        # L/R ankle offset (rel head)
        _s = 6 + 6*nkeep + 14
        toe_off = p[:, _s:_s+6]                                    # L/R toe offset (rel ankle)
        aux_rot = p[:, _s+6:_s+6+126]                              # auxiliary full rotations (training)
        # fill reduced rotations -> full 126 (dropped joints = identity 6D)
        pose_body = self.ident_pose.unsqueeze(0).expand(BT, -1).clone().reshape(BT, 21, 6)
        pose_body[:, self.keep_slots, :] = raw_rot.reshape(BT, nkeep, 6)
        pose_body = pose_body.reshape(BT, 126)
        # FK, then place the root so the head matches the tracker
        joint_position = fk_module(root6, pose_body, self.body_model)[:, :22].reshape(batch, seq_len, 22, 3)
        global_position, root_transl = self.anchor_to_head(joint_position, head_position)   # (B,T,22,3),(B,T,3)
        gp = global_position.reshape(BT, 22, 3)
        # predicted end-effectors (world): ankle rel head(15), toe rel ankle, wrist = tracker
        head_w = gp[:, 15, :]
        ankle_w = torch.stack([head_w + ankle_off[:, :3], head_w + ankle_off[:, 3:6]], 1)    # (BT,2,3)
        toe_w = torch.stack([ankle_w[:, 0] + toe_off[:, :3], ankle_w[:, 1] + toe_off[:, 3:6]], 1)
        wrist_w = wrist_world.reshape(BT, 2, 3)
        sw_ang = torch.atan2(swivel[:, 4:8], swivel[:, :4])                                  # (BT,4)
        aa = torch.cat([sixd2aa(root6), sixd2aa(pose_body.reshape(-1, 6)).reshape(BT, 63)], -1)
        extra = {'swivel': swivel.reshape(batch, seq_len, 8),
                 'ankle_off': ankle_off.reshape(batch, seq_len, 6),
                 'toe_off': toe_off.reshape(batch, seq_len, 6),
                 'ankle_world': ankle_w.reshape(batch, seq_len, 2, 3),
                 'toe_world': toe_w.reshape(batch, seq_len, 2, 3),
                 'aux_rot': aux_rot.reshape(batch, seq_len, 126)}
        if getattr(self, 'skip_recon', False):
            # Eval streaming: skip the per-forward reconstruction, return raw ingredients so test()
            # reconstructs once on the accumulated sequence.
            extra['recon_ingredients'] = (
                aa.reshape(batch, seq_len, 66), gp.reshape(batch, seq_len, 22, 3),
                wrist_w.reshape(batch, seq_len, 2, 3), ankle_w.reshape(batch, seq_len, 2, 3),
                toe_w.reshape(batch, seq_len, 2, 3), sw_ang.reshape(batch, seq_len, 4),
                root_transl.reshape(batch, seq_len, 3))
            zero = torch.zeros(batch, seq_len, 66, device=param.device)
            return root6.reshape(batch, seq_len, 6), raw_rot.reshape(batch, seq_len, -1), extra, zero
        recon_pos, recon_aa = SR.analytic_limb_solver(
            aa, gp, wrist_w, ankle_w, toe_w, sw_ang,
            self.J_rest, self.parents, self.kintree22, self.body_model, root_transl.reshape(BT, 3))
        recon_pos = recon_pos.reshape(batch, seq_len, 22, 3)
        # pelvis-relative, like the joint_position_loss target (FK without translation)
        extra['recon_pos_local'] = (recon_pos - recon_pos[:, :, 0:1]).reshape(batch, seq_len, -1)
        return root6.reshape(batch, seq_len, 6), raw_rot.reshape(batch, seq_len, -1), \
               extra, recon_pos.reshape(batch, seq_len, -1)


class ManikinL(nn.Module):
    """MANIKIN-L network: regress a coarse pose, refine it with the transformer, return a dict of per-frame
    predictions for ModelManikinL. MANIKIN-LN is the same weights evaluated seq2seq instead of causal-online."""
    def __init__(
        self,
        body_model,
        keep_joints,
        nhead=8,
        input_dim=22*18,
        embed_dim=1024,
        single_frame_feat_dim=1024,
        joint_regressor_dim=1024,
        joint_embed_dim=256,
        ):
        super(ManikinL, self).__init__()
        self.body_model = body_model

        # Coarse-pose regressor: static per-frame embedding -> initial full-body pose; the 3 tracked joints
        # (head/hands) are then overwritten with the input tracker signal.
        self.linear_embedding_static = nn.Linear(input_dim, embed_dim)
        self.joint_regressor = CoarsePoseNet(self.body_model, joint_regressor_dim)

        # Tokens for the transformer: a position and a rotation token per joint (22 each) plus one input token.
        self.token_num = 22 + 22 + 1
        self.position_embedding = nn.Linear(3, joint_embed_dim * 2)
        self.rotation_embedding = nn.Linear(6, joint_embed_dim * 2)
        self.input_token_embedding = nn.Linear(single_frame_feat_dim, joint_embed_dim * 2)
        self.motion_net = PoseRefiner(body_model, keep_joints, joint_feat_dim=joint_embed_dim * 2, node_num=self.token_num, nhead=nhead)


    def prepare_input(self, input_tensor):
        # tracker context: head/hands rotation + position, head-relative positions
        tracker_ctx = {}
        tracker_ctx['tracker_rot'] = input_tensor[:, :, [15, 20, 21], 0:6].clone()
        tracker_ctx['tracker_pos'] = input_tensor[:, :, [15, 20, 21], 12:15].clone()
        tracker_ctx['head_pos'] = tracker_ctx['tracker_pos'][:, :, 0].clone()
        tracker_ctx['tracker_rel_pos'] = tracker_ctx['tracker_pos'] - tracker_ctx['head_pos'][:, :, None]
        # mask out non-tracked joints
        selected_joint_index = list(range(0, 14+1)) + list(range(16, 19+1))
        input_tensor[:, :, selected_joint_index] = torch.ones_like(input_tensor[:, :, selected_joint_index]) * 0.01
        return input_tensor, tracker_ctx


    def build_tokens(self, coarse_position, coarse_rotation, frame_embedding):
        batch, seq_len = frame_embedding.shape[0], frame_embedding.shape[1]
        position_token = self.position_embedding(coarse_position.reshape(batch * seq_len, 22, -1))
        rotation_token = self.rotation_embedding(coarse_rotation.reshape(batch * seq_len, 22, -1))
        input_token = self.input_token_embedding(frame_embedding.detach()).reshape(batch*seq_len, -1)[:, None]
        return torch.cat([position_token, rotation_token, input_token], dim=-2)


    def forward(self, input_tensor, time_pad_mask=None):
        outputs = collections.defaultdict(list)
        batch, seq_len, _, _ = input_tensor.shape

        # --- coarse pose: per-frame regressor -> initial full-body pose; head/hand joints overwritten with
        # the tracker signal. coarse_position is also supervised directly, hence returned as pred_init_pose.
        input_tensor, tracker_ctx = self.prepare_input(input_tensor)
        frame_embedding = self.linear_embedding_static(input_tensor.reshape(batch, seq_len, -1))
        coarse_position, coarse_rotation = self.joint_regressor(frame_embedding)
        coarse_position = coarse_position.reshape(batch, seq_len, 22, 3)
        coarse_position = coarse_position - coarse_position[:, :, 15:16]              # head-relative
        coarse_rotation = coarse_rotation.reshape(batch, seq_len, 22, 6).clone()
        rotation_local_matrot = sixd2matrot(coarse_rotation.reshape(-1, 6)).reshape(batch * seq_len, 22, 9)
        rotation_global_matrot = local2global_pose(rotation_local_matrot, self.body_model.kintree_table[0].long())  # joint rot rel. origin
        coarse_rotation = matrot2sixd(rotation_global_matrot.reshape(-1, 3, 3)).reshape(batch, seq_len, 22, 6)
        coarse_position[:, :, [15, 20, 21]] = tracker_ctx['tracker_rel_pos']
        coarse_rotation[:, :, [15, 20, 21]] = tracker_ctx['tracker_rot']
        coarse_position = coarse_position.reshape(batch, seq_len, -1)

        # --- refine with the spatio-temporal transformer
        tokens = self.build_tokens(coarse_position, coarse_rotation, frame_embedding)
        wrist_world = tracker_ctx['tracker_pos'][:, :, 1:3]                                  # tracker hands
        global_orientation, joint_rotation, extra, global_position = self.motion_net(
            tokens.reshape(batch, seq_len, self.token_num, -1), tracker_ctx['head_pos'], wrist_world, time_pad_mask=time_pad_mask)

        outputs['pred_init_pose'].append(coarse_position)
        outputs['pred_global_orientation'].append(global_orientation)
        outputs['pred_joint_rotation'].append(joint_rotation)
        outputs['pred_swivel_extra'].append(extra)                   # swivel/ankle/toe predictions for the loss
        if 'recon_pos_local' in extra:                               # absent when skip_recon (eval streaming)
            outputs['pred_joint_position'].append(extra['recon_pos_local'])
        outputs['pred_global_position'].append(global_position)
        return outputs
