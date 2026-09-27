# --------------------------------------------
# MANIKIN-S and MANIKIN-L/LN models.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

"""MANIKIN pose models.
  ModelManikinS  -- MANIKIN-S (compact transformer backbone + swivel head)
  ModelManikinL    -- MANIKIN-L / MANIKIN-LN (transformer backbone + swivel head)
Both subclass ModelBase; selected by models/select_model.py from the config.
"""
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from models.select_model import define_G, define_bm
from models.model_base import ModelBase
from models.loss import velocityLoss, footContactLoss, penetrationLoss, footHeightLoss
from utils.utils_transform import fk_module
from utils import utils_transform as utils_transform
from utils.utils_transform import local2global_pose
from networks import swivel as SR
from networks.swivel import positions_to_swivel, swivel_to_mid, solve_limb_angles


# MANIKIN-S: compact transformer backbone + swivel head
class ModelManikinS(ModelBase):
    def __init__(self, opt):
        super(ModelManikinS, self).__init__(opt)
        self.opt_train = self.opt['train']
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)

        self.opt_bm = self.opt['body_model']
        self.bm_dict = define_bm(opt)
        default_gender = self.opt_bm['default_gender']
        self.bm = self.bm_dict[default_gender]
        self.w_joint_pos = self.opt.get('w_joint_pos', 5)   # torso joint positions
        self.w_swivel = self.opt.get('w_swivel', 0.1)
        self.w_mid = self.opt.get('w_mid', 1)               # swivel-reconstructed knee/elbow
        self.w_ankle = self.opt.get('w_ankle', 5)           # predicted ankle position
        self.w_toe_pos = self.opt.get('w_toe_pos', 5.0)     # predicted foot bone
        # w_consistency: heading-equivariance loss. Forward again under a random Z rotation R and penalise
        # ||root-rel pred(R.x) - R . root-rel pred(x)||, encouraging consistent predictions across headings.
        self.w_consistency = self.opt.get('w_consistency', 0.0)
        # The network predicts rotations of the torso and limb-root joints (rotation_keep_joints). Positions are
        # supervised on the joints whose ancestors all have a predicted rotation, including the hips/shoulders
        # that define the swivel reference.
        _keepj = self.opt['netG']['rotation_keep_joints']
        self.rot_keep_dims = torch.tensor([(j-1)*6+k for j in _keepj for k in range(6)]).long()
        _par = self.bm.kintree_table[0][:22].long().tolist()   # [-1,0,0,0,1,2,...]
        def _pos_valid(j):
            a = _par[j]
            while a > 0:
                if a not in _keepj: return False
                a = _par[a]
            return True
        self.pos_sup_joints = torch.tensor([j for j in range(22) if _pos_valid(j)]).long()

        with torch.no_grad():
            self.J_rest = self.bm().Jtr[0, :22].detach().to(self.device)    # (22,3) rest joints, mean shape
        self.parents = self.bm.kintree_table[0][:22].long().tolist()
        # knee/elbow flexion axes (local frame), used as the 1-DoF hinge axes
        self.hinge_ref = {4: torch.tensor([0.985,0.003,-0.174]), 5: torch.tensor([0.995,-0.029,0.096]),
                          18: torch.tensor([0.189,-0.966,0.179]), 19: torch.tensor([0.085,0.970,-0.227])}
        self.hinge_ref = {k: (v/v.norm()).to(self.device) for k, v in self.hinge_ref.items()}

    """ Preparation before training; save model during training. """

    # initialize training
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # train mode
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    def init_test(self):
        self.load(test=True)                  # load model
        self.log_dict = OrderedDict()         # log

    # load pre-trained G model
    def load(self, test=False):
        load_path_G = self.opt['path']['pretrained_netG'] if test == False else self.opt['path']['pretrained']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')

    # load optimizer
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # save model / optimizer(optional)
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # define loss
    def define_loss(self):
        self.G_lossfn = nn.L1Loss().to(self.device)

    # define optimizer
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)

    # define scheduler (MultiStepLR)
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))

    """ Optimization during training; testing/evaluation. """

    # feed L/H data
    def feed_data(self, data, need_H=True, test=False):

        self.L = data['L'].to(self.device)
        self.Head_trans_global = data['Head_trans_global'].to(self.device)
        self.H_global_orientation = data['H'][...,:6].to(self.device)
        self.H_joint_rotation = data['H'][...,6:].to(self.device)
        self.H_pos_pelvis = data['pos_pelvis_gt'].to(self.device)
        self.H_joint_position = self.fk(self.bm, self.H_global_orientation, self.H_joint_rotation)

    # feed L to netG
    def netG_forward(self):
        self.netG.eval()
        self.E_global_orientation, self.E_joint_rotation, self.E_swivel, E_ankle = self.netG(self.L)
        self.E_ankle_pos, self.E_toe_pos = E_ankle[:, :6], E_ankle[:, 6:12]     # ankle rel. head | foot bone
        self.E_joint_position = self.fk(self.bm, self.E_global_orientation, self.E_joint_rotation)

    # update parameters and get loss
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()

        kd = self.rot_keep_dims.to(self.E_joint_rotation.device)
        joint_rotation_loss = self.G_lossfn(self.E_joint_rotation[:, kd], self.H_joint_rotation[:,-1,...][:, kd])

        self.predicted_angle_aa = utils_transform.sixd2aa(torch.cat([self.E_global_orientation, self.E_joint_rotation],dim=-1).reshape(-1,6)).reshape(self.E_global_orientation.shape[0],-1)
        self.predicted_position = self.E_joint_position
        rotation_local_matrot = utils_transform.aa2matrot(self.predicted_angle_aa.reshape(-1,3)).reshape(self.predicted_angle_aa.shape[0],-1,9)
        rotation_global_matrot = local2global_pose(rotation_local_matrot, self.bm.kintree_table[0][:22].long()) # rotation of joints relative to the origin

        # swivel angles and swivel-reconstructed mid joints (knee/elbow) against the ground truth
        gt_joint_position = self.H_joint_position[:,-1,...]
        pdist = torch.nn.PairwiseDistance(p=2,keepdim=True)
        mid_loss = 0
        swivel_loss = 0
        for ja_l, jb_l, jc_l, cos_idx, sin_idx in [
            (16, 18, 20, 0, 4),  # left arm
            (17, 19, 21, 1, 5),  # right arm
            (1, 4, 7, 2, 6),     # left leg
            (2, 5, 8, 3, 7),     # right leg
        ]:
            ref_world = rotation_global_matrot[:,ja_l,:].reshape(-1,3,3)[:,:,1].unsqueeze(1)   # swivel reference: limb-root Y axis
            a_gt_l = gt_joint_position[:,ja_l:ja_l+1,:]
            b_gt_l = gt_joint_position[:,jb_l:jb_l+1,:]
            c_gt_l = gt_joint_position[:,jc_l:jc_l+1,:]
            d1_l = pdist(a_gt_l, b_gt_l)
            d2_l = pdist(b_gt_l, c_gt_l)
            d3_l = pdist(a_gt_l, c_gt_l).clamp(min=1e-4)
            E_swivel_l = torch.atan2(self.E_swivel[:,sin_idx], self.E_swivel[:,cos_idx]).unsqueeze(-1)
            mid_swivel_l = a_gt_l + swivel_to_mid(a_gt_l-a_gt_l, c_gt_l-a_gt_l, ref_world, E_swivel_l.unsqueeze(-1), d1_l, d2_l, d3_l)
            mid_gt_l = gt_joint_position[:,jb_l,:].squeeze()
            swivel_gt_l = positions_to_swivel(a_gt_l-a_gt_l, b_gt_l-a_gt_l, c_gt_l-a_gt_l, ref_world)
            mid_loss = mid_loss + self.G_lossfn(mid_swivel_l.squeeze(), mid_gt_l)
            swivel_loss = swivel_loss + self.G_lossfn(self.E_swivel[:,cos_idx], torch.cos(swivel_gt_l).squeeze()) + self.G_lossfn(self.E_swivel[:,sin_idx], torch.sin(swivel_gt_l).squeeze())

        global_orientation_loss = self.G_lossfn(self.E_global_orientation, self.H_global_orientation[:,-1,...])
        sj = self.pos_sup_joints.to(self.predicted_position.device)
        joint_position_loss = (self.predicted_position[:, sj, :] - self.H_joint_position[:,-1,sj,:]).abs().mean(dim=(0, 2)).mean()

        # ankle position relative to the head, and the foot bone (toe relative to its ankle)
        gt_head_pos = self.H_joint_position[:,-1,15,:]
        gt_ankle_pos = torch.cat([self.H_joint_position[:,-1,7,:] - gt_head_pos,
                                  self.H_joint_position[:,-1,8,:] - gt_head_pos], dim=-1)   # (B, 6)
        ankle_pos_loss = self.G_lossfn(self.E_ankle_pos, gt_ankle_pos)
        gt_toe = torch.cat([self.H_joint_position[:,-1,10,:] - self.H_joint_position[:,-1,7,:],
                            self.H_joint_position[:,-1,11,:] - self.H_joint_position[:,-1,8,:]], dim=-1)
        toe_pos_loss = self.G_lossfn(self.E_toe_pos, gt_toe)   # (B,6)

        loss =  0.02*global_orientation_loss + joint_rotation_loss  + self.w_joint_pos*joint_position_loss + self.w_swivel*swivel_loss + self.w_mid* mid_loss + self.w_ankle*ankle_pos_loss
        loss = loss + self.w_toe_pos*toe_pos_loss

        # ---- heading-equivariance consistency (optional) ----
        if self.w_consistency > 0:
            B = self.L.shape[0]
            ang = torch.rand(B, device=self.L.device) * (2 * 3.141592653589793)
            c, s = torch.cos(ang), torch.sin(ang)
            z, o = torch.zeros_like(c), torch.ones_like(c)
            R = torch.stack([torch.stack([c, -s, z], -1),
                             torch.stack([s,  c, z], -1),
                             torch.stack([z,  z, o], -1)], -2)          # (B,3,3)
            rot_vel = self.opt['datasets']['train'].get('heading_rotate_velocity', True)
            Lr = self.L.clone()
            blocks = [0, 6, 12] + ([18, 24, 30] if rot_vel else [])
            for b0 in blocks:                                          # 6D orientations: rotate both columns
                for k in (0, 3):
                    Lr[..., b0+k:b0+k+3] = torch.einsum('bij,btj->bti', R, Lr[..., b0+k:b0+k+3])
            for b0 in [36, 39, 42, 45, 48, 51]:                        # positions and their world deltas
                Lr[..., b0:b0+3] = torch.einsum('bij,btj->bti', R, Lr[..., b0:b0+3])
            out_r = self.netG(Lr)
            go_r, jr_r = out_r[0], out_r[1]
            P_r = self.fk(self.bm, go_r, jr_r)                         # (B,22,3) FK, pelvis-anchored
            P = self.E_joint_position
            P_rel = P - P[:, [0]]                                      # root-relative
            Pr_rel = P_r - P_r[:, [0]]
            P_rel_rot = torch.einsum('bij,bnj->bni', R, P_rel)         # rotated prediction
            consistency_loss = (Pr_rel - P_rel_rot).norm(dim=-1).mean()
            loss = loss + self.w_consistency * consistency_loss
            self.log_dict['consistency_loss'] = consistency_loss.item()

        if not torch.isnan(loss):
            loss.backward()
            self.G_optimizer.step()

        self.log_dict['total_loss'] = loss.item()
        self.log_dict['global_orientation_loss'] = global_orientation_loss.item()
        self.log_dict['joint_rotation_loss'] = joint_rotation_loss.item()
        self.log_dict['joint_position_loss'] = joint_position_loss.item()
        self.log_dict['swivel_loss'] = swivel_loss.item()
        self.log_dict['mid_loss'] = mid_loss.item()
        self.log_dict['ankle_pos_loss'] = ankle_pos_loss.item()
        self.log_dict['toe_pos_loss'] = toe_pos_loss.item()


    # test / inference
    def test(self):
        self.netG.eval()

        self.L = self.L.squeeze()
        self.Head_trans_global = self.Head_trans_global.squeeze()
        window_size = self.opt['datasets']['test']['window_size']
        test_batch = self.opt['datasets']['test']['test_batch']

        with torch.no_grad():
            num_frames = self.L.shape[0]
            # the first window_size-1 frames see the growing prefix; then full windows, test_batch at a time
            outs = [self.netG(self.L[0:frame_idx+1].unsqueeze(0)) for frame_idx in range(min(num_frames, window_size-1))]
            if num_frames >= window_size:
                for idx_batch in range((num_frames-window_size)//test_batch+1):
                    L_segment = self.L[test_batch*idx_batch:test_batch*(idx_batch+1)+window_size-1]
                    outs.append(self.netG(torch.stack([L_segment[f-window_size+1:f+1]
                                                       for f in range(window_size-1, L_segment.shape[0])])))
            E_global_orientation_tensor, E_joint_rotation_tensor, E_swivel_tensor, E_ankle_tensor = \
                (torch.cat([o[k] for o in outs], dim=0) for k in range(4))

        E = torch.cat([E_global_orientation_tensor, E_joint_rotation_tensor],dim=-1).to(self.device).squeeze()
        predicted_angle = utils_transform.sixd2aa(E[:,:132].reshape(-1,6).detach()).reshape(E[:,:132].shape[0],-1).float()
        # root translation: puts the head joint on the tracked head position
        t_head2world = self.Head_trans_global[:,:3,3]
        body_pose_local = self.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3]})
        self.predicted_translation = -body_pose_local.Jtr[:,15,:]+t_head2world
        self.predicted_position = self.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3], 'trans': self.predicted_translation}).Jtr[:,:22,:]
        rotation_local_matrot = utils_transform.aa2matrot(predicted_angle.reshape(-1,3)).reshape(predicted_angle.shape[0],-1,9)
        rotation_global_matrot = local2global_pose(rotation_local_matrot, self.bm.kintree_table[0][:22].long()) # rotation of joints relative to the origin

        # Analytic Arm/Leg Solver: from each limb's base joint (shoulder/hip, from torso FK), its swivel angle
        # and its end joint (tracked wrist / predicted ankle), reconstruct the mid joint (elbow/knee).
        # Tracked wrists in the world frame (the input positions are floor-normalised, the head transform is not).
        to_world = self.Head_trans_global[:, :3, 3] - self.L[:, 36:39]
        wrist_world = {20: self.L[:, 39:42] + to_world, 21: self.L[:, 42:45] + to_world}
        head_world = self.predicted_position[:, 15, :]
        pdist = torch.nn.PairwiseDistance(p=2,keepdim=True)
        n_fr = self.predicted_position.shape[0]
        swivel_targets = []   # per limb (base, mid, end, toe, swivel direction), solved into joint angles below
        for ja_l, jb_l, jc_l, cos_idx, sin_idx in [
            (16, 18, 20, 0, 4), (17, 19, 21, 1, 5),     # arms
            (1, 4, 7, 2, 6), (2, 5, 8, 3, 7),           # legs
        ]:
            ref_sw = rotation_global_matrot[:,ja_l,:].reshape(-1,3,3)[:,:,1].unsqueeze(1)   # swivel reference: limb-root Y axis
            a_sw = self.predicted_position[:,ja_l,:]
            if jc_l in [20, 21]:
                c_sw = wrist_world[jc_l]
            else:
                ai = 0 if jc_l == 7 else 1
                c_sw = E_ankle_tensor[:, ai*3:ai*3+3] + head_world          # ankle, predicted relative to the head
            d1_sw = (self.J_rest[jb_l] - self.J_rest[ja_l]).norm().reshape(1, 1, 1).expand(n_fr, 1, 1)
            d2_sw = (self.J_rest[jc_l] - self.J_rest[jb_l]).norm().reshape(1, 1, 1).expand(n_fr, 1, 1)
            d3_sw = pdist(a_sw.unsqueeze(1), c_sw.unsqueeze(1))
            E_swi_sw = torch.atan2(E_swivel_tensor[:,sin_idx], E_swivel_tensor[:,cos_idx]).unsqueeze(-1)
            mid_sw = a_sw.unsqueeze(1) + swivel_to_mid(a_sw.unsqueeze(1)-a_sw.unsqueeze(1), (c_sw-a_sw).unsqueeze(1), ref_sw, E_swi_sw.unsqueeze(-1), d1_sw, d2_sw, d3_sw)
            # swivel direction k (world), same (u, v) frame as swivel_to_mid; sets the limb twist, also for a straight limb
            _n = (c_sw - a_sw) / ((c_sw - a_sw).norm(dim=-1, keepdim=True) + 1e-8)
            _v0 = ref_sw.reshape(-1, 3).expand_as(_n)
            _u = -_v0 + (_v0 * _n).sum(-1, keepdim=True) * _n; _u = _u / (_u.norm(dim=-1, keepdim=True) + 1e-8)
            kdir_sw = _u * torch.cos(E_swi_sw) + torch.cross(_u, _n, dim=-1) * torch.sin(E_swi_sw)
            toe_world = None
            if jc_l in [7, 8]:
                toe_world = c_sw + E_ankle_tensor[:, 6+ai*3:6+ai*3+3]          # predicted foot bone (toe - ankle)
            swivel_targets.append((ja_l, jb_l, jc_l, mid_sw.squeeze(1), c_sw, toe_world, kdir_sw))

        # joint angles from the swivel targets, then FK
        self.predicted_position_swivel, self.predicted_angle_swivel = \
            self._swivel_positions_via_angles(predicted_angle, swivel_targets)

        self.netG.train()


    def current_log(self):
        return self.log_dict

    # predictions for evaluation
    def current_prediction(self,):
        body_parms = OrderedDict()
        pose = self.predicted_angle_swivel         # limbs solved from the swivel angles
        body_parms['pose_body'] = pose[...,3:66]
        body_parms['root_orient'] = pose[...,:3]
        body_parms['trans'] = self.predicted_translation
        body_parms['position'] = self.predicted_position                  # torso FK
        body_parms['position_swivel'] = self.predicted_position_swivel    # after the limb solver

        return body_parms

    def current_gt(self, ):
        num_frames = self.H_joint_rotation.squeeze().shape[0]
        body_parms = OrderedDict()
        body_parms['pose_body'] = utils_transform.sixd2aa(self.H_joint_rotation.squeeze().reshape(num_frames,-1,6),batch=True).reshape(num_frames,-1)
        body_parms['root_orient'] = utils_transform.sixd2aa(self.H_global_orientation.squeeze()) 
        body_parms['trans'] = self.H_pos_pelvis.squeeze()
        body_parms['position'] = self.H_joint_position + self.H_pos_pelvis.squeeze().unsqueeze(1)
        return body_parms


    """ Information of netG. """

    # print network
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # print params
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # network information
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # params information
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg

    # forward kinematics
    def fk(self, body_model, global_orientation, joint_rotation, trans=None, betas=None):

        bs = global_orientation.shape[0]
        len_seq = global_orientation.shape[1] if len(global_orientation.shape) == 3 else 1
        global_orientation = utils_transform.sixd2aa(global_orientation.reshape(-1,6)).reshape(-1,3).float()
        joint_rotation = utils_transform.sixd2aa(joint_rotation.reshape(-1,6)).reshape(-1,63).float()
        body_pose = body_model(**{'pose_body':joint_rotation, 'root_orient':global_orientation, 'trans': trans, 'betas':betas})
        joint_position = body_pose.Jtr[...,:22,:].reshape(bs,len_seq,-1,3).squeeze()
        return joint_position

    def _swivel_positions_via_angles(self, predicted_angle, targets):
        """Recover each limb's local rotations from the swivel reconstruction, then FK.

        Knee/elbow are 1-DoF hinges about the flexion axis; the hip/shoulder twist comes from the swivel direction.
        targets: list of (ja, jb, jc, mid, end, toe_or_None, swivel_dir) in world coordinates.
        """
        _kt = self.bm.kintree_table[0][:22].long()
        # wrist orientation = tracked hand rotation (input 6D at dims 6:18), used to align the hands
        _wrist_ori = utils_transform.sixd2matrot(self.L[:, 6:18].reshape(-1, 6)).reshape(-1, 2, 3, 3)
        # same Analytic Arm/Leg Solver as MANIKIN-L (networks/swivel.py)
        return solve_limb_angles(predicted_angle.clone(), self.predicted_position, targets, self.J_rest, self.parents,
                                 _kt, self.bm, self.predicted_translation, hinge_ref=self.hinge_ref,
                                 wrist_ori=_wrist_ori, temporal=True)   # inference on one time-ordered sequence


# MANIKIN-L / MANIKIN-LN: transformer backbone + swivel head
class ModelManikinL(ModelBase):
    def __init__(self, opt):
        super(ModelManikinL, self).__init__(opt)
        self.opt_train = self.opt['train']
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        self.bm = self.netG.body_model
        self._keep = list(self.opt['netG']['keep_joints'])   # joints with a predicted rotation (torso + limb roots)
        self._keep_slots = [j-1 for j in self._keep]         # into the 21-joint pose_body
        self._kintree22 = self.bm.kintree_table[0][:22].long()
        self.w_swivel = self.opt['netG'].get('w_swivel', 1.0)
        self.w_ankle  = self.opt['netG'].get('w_ankle', 5.0)
        self.w_toe    = self.opt['netG'].get('w_toe', 5.0)
        self.w_recon  = self.opt['netG'].get('w_recon', 10.0)   # world positions after the limb solver
        self.w_aux    = self.opt['netG'].get('w_aux', 2.0)      # auxiliary full-rotation loss
        self.w_root   = self.opt['netG'].get('w_root_orient', 0.02)


    """ Preparation before training; save model during training. """

    # initialize training
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # train mode
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    def init_test(self):
        self.load(test=True)                  # load model
        self.log_dict = OrderedDict()         # log

    # load pre-trained G model
    def load(self, test=False):
        load_path_G = self.opt['path']['pretrained_netG'] if test == False else self.opt['path']['pretrained']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')

    # load optimizer
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # save model / optimizer(optional)
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # define loss
    def define_loss(self):
        self.G_lossfn = nn.L1Loss().to(self.device)

    # define optimizer
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)

    # define scheduler (MultiStepLR)
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))

    """ Optimization during training; testing/evaluation. """

    # feed L/H data
    def feed_data(self, data, test=False):
        # (batch, window_size, 22*18): per joint rotation, rotation velocity, position, position velocity
        self.input_signal = data['input_signal'].to(self.device)
        batch, seq_len = self.input_signal.shape[0], self.input_signal.shape[1]
        rotation = self.input_signal[:, :, :22*6].reshape(batch, seq_len, 22, 6)
        velocity_rotation = self.input_signal[:, :, 22*6:22*6*2].reshape(batch, seq_len, 22, 6)
        position = self.input_signal[:, :, 22*6*2:22*6*2+3*22].reshape(batch, seq_len, 22, 3)
        velocity_position = self.input_signal[:, :, 22*6*2+3*22:].reshape(batch, seq_len, 22, 3)
        self.input_signal = torch.cat((rotation, velocity_rotation, position, velocity_position), dim=3)

        self.H_floor_height = data['floor_height'].to(self.device)
        self.H_global_orientation = data['rotation_local_full'][:, :, :6].to(self.device)
        self.H_joint_rotation = data['rotation_local_full'][:, :, 6:].to(self.device)
        # heading (yaw) augmentation is applied in the dataloader (data/dataset_l.py)
        self.H_joint_position = fk_module(self.H_global_orientation.reshape(batch * seq_len, -1), self.H_joint_rotation.reshape(batch * seq_len, -1), self.bm).reshape(batch, seq_len, -1)
        self.H_global_root_trans = data['pos_pelvis_gt'].to(self.device)

        # training only
        if not test:
            self.H_floor_contact = data['foot_contact'].to(self.device)

        # testing only
        if test:
            self.H_global_head_trans = data['global_head_trans'].to(self.device)
            self.H_body_param_list = data['body_param_list']
            for k,v in self.H_body_param_list.items():
                self.H_body_param_list[k] = v.squeeze().to(self.device)

    def _compute_gt_swivel(self, gt_gp, batch, seq_len):
        """GT swivel angle per limb from GT world positions, with the limb-root global Y axis as reference
        (same convention as networks/swivel.py)."""
        BT = batch * seq_len
        go = self.H_global_orientation.reshape(BT, 6)
        jr = self.H_joint_rotation.reshape(BT, 21*6)
        loc = utils_transform.sixd2matrot(torch.cat([go, jr], -1).reshape(-1, 6)).reshape(BT, 22, 9)
        Rg = local2global_pose(loc, self._kintree22).reshape(BT, 22, 3, 3)
        gp = gt_gp.reshape(BT, 22, 3)
        sw = torch.zeros(BT, 4, device=gp.device)
        for li, (ja, jb, jc) in enumerate(SR.LIMBS):
            v = Rg[:, ja, :, 1].unsqueeze(1)
            a, b, c = gp[:, ja].unsqueeze(1), gp[:, jb].unsqueeze(1), gp[:, jc].unsqueeze(1)
            sw[:, li] = SR.positions_to_swivel(a - a, b - a, c - a, v).squeeze(-1).squeeze(-1)
        return sw.reshape(batch, seq_len, 4)

    # feed L to netG
    def netG_forward(self):
        self.predictions = self.netG(self.input_signal)

    # update parameters and get loss
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        loss = 0
        batch, seq_len, _ = self.predictions['pred_joint_position'][-1].shape

        for i in range(len(self.predictions['pred_global_orientation'])):
            global_orientation_loss = self.G_lossfn(self.predictions['pred_global_orientation'][i], self.H_global_orientation)
            self.log_dict[f'global_orientation_loss_{i}'] = global_orientation_loss.item()
            loss += global_orientation_loss * self.w_root

        for i in range(len(self.predictions['pred_joint_rotation'])):
            rotation_error_rate = 2
            # only the kept (torso + limb-root) joints are predicted
            gt_rot = self.H_joint_rotation.reshape(batch, seq_len, 21, 6)[:, :, self._keep_slots].reshape(batch, seq_len, -1)
            joint_rotation_loss = self.G_lossfn(self.predictions['pred_joint_rotation'][i], gt_rot) * rotation_error_rate
            self.log_dict[f'joint_rotation_loss_{i}'] = joint_rotation_loss.item()
            loss += joint_rotation_loss

        # swivel angles (GT from GT positions), ankle/toe world positions, auxiliary full rotations
        gt_gp = self.H_joint_position[:, :, :22*3].reshape(batch, seq_len, 22, 3) + self.H_global_root_trans[:, :, None]
        gt_sw = self._compute_gt_swivel(gt_gp, batch, seq_len)   # (B,T,4) angle
        for extra in self.predictions['pred_swivel_extra']:
            sw = extra['swivel'].reshape(batch, seq_len, 8)
            pred_ang = torch.atan2(sw[..., 4:8], sw[..., :4])
            # angle diff wrapped to (-pi,pi]
            d = torch.atan2(torch.sin(pred_ang - gt_sw), torch.cos(pred_ang - gt_sw))
            swivel_loss = d.abs().mean() * self.w_swivel
            loss += swivel_loss; self.log_dict['swivel_loss'] = swivel_loss.item()
            ankle_loss = self.G_lossfn(extra['ankle_world'].reshape(batch, seq_len, 2, 3),
                                       gt_gp[:, :, [7, 8]]) * self.w_ankle
            toe_loss = self.G_lossfn(extra['toe_world'].reshape(batch, seq_len, 2, 3),
                                     gt_gp[:, :, [10, 11]]) * self.w_toe
            loss += ankle_loss + toe_loss
            self.log_dict['ankle_loss'] = ankle_loss.item(); self.log_dict['toe_loss'] = toe_loss.item()
            aux_loss = self.G_lossfn(extra['aux_rot'], self.H_joint_rotation) * self.w_aux
            loss += aux_loss; self.log_dict['aux_rot_loss'] = aux_loss.item()
        # world positions of all joints after the limb solver
        for gpos in self.predictions['pred_global_position']:
            recon_pos_loss = self.G_lossfn(gpos.reshape(batch, seq_len, 22, 3), gt_gp) * self.w_recon
            loss += recon_pos_loss
            self.log_dict['recon_pos_loss'] = recon_pos_loss.item()

        for i in range(len(self.predictions['pred_joint_position'])):
            pose_error_rate = 5
            joint_position_loss = self.G_lossfn(self.predictions['pred_joint_position'][i][:, :, :22*3], self.H_joint_position[:, :, :22*3]) * pose_error_rate
            loss += joint_position_loss
            self.log_dict[f'joint_position_loss_{i}'] = joint_position_loss.item()

        for i in range(len(self.predictions['pred_global_position'])):
            # global position
            gt_global_position = self.H_joint_position[:, :, :22*3].reshape(batch, seq_len, 22, 3) + self.H_global_root_trans[:, :, None]
            pred_global_position = self.predictions['pred_global_position'][i].reshape(batch, seq_len, 22, 3)

            velocity_error_scale = 50
            # velocity loss
            global_velocity_loss = velocityLoss(loss_func=self.G_lossfn, pred=pred_global_position, gt=gt_global_position) * velocity_error_scale
            loss += global_velocity_loss
            self.log_dict[f'global_velocity_loss_{i}'] = global_velocity_loss.item()

            global_velocity_loss = velocityLoss(loss_func=self.G_lossfn, pred=pred_global_position, gt=gt_global_position, interval=3) * velocity_error_scale / 3
            loss += global_velocity_loss
            self.log_dict[f'global_velocity_loss_i3_{i}'] = global_velocity_loss.item()

            global_velocity_loss = velocityLoss(loss_func=self.G_lossfn, pred=pred_global_position, gt=gt_global_position, interval=5) * velocity_error_scale / 5 
            loss += global_velocity_loss
            self.log_dict[f'global_velocity_loss_i5_{i}'] = global_velocity_loss.item()

            # foot contact loss
            foot_concat_error_scale = 20
            global_foot_concat_loss = footContactLoss(loss_func=self.G_lossfn, pred=pred_global_position, gt=gt_global_position) * foot_concat_error_scale
            loss += global_foot_concat_loss
            self.log_dict[f'global_foot_concat_loss_{i}'] = global_foot_concat_loss.item()

            # penetration loss
            pene_error_scale = 1
            pene_loss = penetrationLoss(self.predictions['pred_global_position'][i].reshape(batch * seq_len, -1, 3), self.H_floor_height) * pene_error_scale
            loss += pene_loss
            self.log_dict[f'penetration_loss_{i}'] = pene_loss.item()

            # foot-height loss
            foot_height_loss_scale = 0.5
            fh_loss = footHeightLoss(self.predictions['pred_global_position'][i].reshape(batch * seq_len, -1, 3), self.H_floor_height, self.H_floor_contact) * foot_height_loss_scale
            loss += fh_loss
            self.log_dict[f'foot_height_loss_{i}'] = fh_loss.item()

            # hand positions
            hand_alignment_loss_scale = 5
            hand_alignment_loss = self.G_lossfn(self.predictions['pred_global_position'][i].reshape(batch, seq_len, 22, 3)[:, :, [20, 21]], gt_global_position[:, :, [20, 21]]) * hand_alignment_loss_scale
            loss += hand_alignment_loss
            self.log_dict[f'hand_alignment_loss_{i}'] = hand_alignment_loss.item()


        for i, init_pose in enumerate(self.predictions['pred_init_pose']):
            init_pose_scale = 1
            gt_22 = self.H_joint_position[:, :, :22*3].reshape(batch, seq_len, 22, 3)
            gt_22_head_centered = gt_22 - gt_22[:, :, 15:16]
            pose_loss = self.G_lossfn(init_pose, gt_22_head_centered.reshape(batch, seq_len, -1)) * init_pose_scale
            loss += pose_loss
            self.log_dict[f'joint_regress_loss_{i}'] = pose_loss.item()

        self.log_dict['total_loss'] = loss.item()
        loss.backward()
        self.G_optimizer.step()

    # test / inference
    def test(self):
        self.netG.eval()
        self.input_signal = self.input_signal.squeeze()
        self.H_global_head_trans = self.H_global_head_trans.squeeze()
        x = self.input_signal; T = x.shape[0]
        W = self.opt['datasets']['test']['window_size']
        part = self.opt['datasets']['test'].get('test_batch') or 512       # windows per forward

        # the network returns the limb-solver inputs; the solver then runs once on the whole sequence
        _mn = self.netG.module.motion_net if hasattr(self.netG, 'module') else self.netG.motion_net
        _mn.skip_recon = True
        seq2seq = self.opt['netG'].get('fuse_windows') == 'seq2seq'

        with torch.no_grad():
            if seq2seq:
                # MANIKIN-LN: non-overlapping windows (the last one ends at the last frame); frames covered by
                # two windows are averaged
                starts = list(range(0, T - W + 1, W)) if T > W else [0]
                if T > W and starts[-1] != T - W: starts.append(T - W)
                Weff = min(T, W)
                sum_go = sum_jr = sum_ex = None; cnt = torch.zeros(T, 1, device=x.device)
                for p0 in range(0, len(starts), part):
                    st = starts[p0:p0 + part]
                    outputs = self.netG(torch.stack([x[s0:s0 + Weff] for s0 in st]))
                    go = outputs['pred_global_orientation'][-1]                   # (B,Weff,6)
                    jr = outputs['pred_joint_rotation'][-1]
                    ex = outputs['pred_swivel_extra'][-1]
                    exb = torch.cat([ex['swivel'], ex['ankle_off'], ex['toe_off']], dim=-1)  # (B,Weff,8+6+6)
                    if sum_go is None:
                        sum_go = torch.zeros(T, go.shape[-1], device=x.device)
                        sum_jr = torch.zeros(T, jr.shape[-1], device=x.device)
                        sum_ex = torch.zeros(T, exb.shape[-1], device=x.device)
                    for bi, s0 in enumerate(st):
                        sum_go[s0:s0 + Weff] += go[bi]; sum_jr[s0:s0 + Weff] += jr[bi]
                        sum_ex[s0:s0 + Weff] += exb[bi]; cnt[s0:s0 + Weff] += 1
                pred_global_orientation_tensor = sum_go / cnt
                pred_joint_rotation_tensor = sum_jr / cnt
                fused_ex = sum_ex / cnt
            else:
                # online: frame t is the last frame of the window ending at t. The first W frames use their
                # growing prefix, in one forward (right-padded, with a key-padding mask).
                ing_list = []     # limb-solver inputs, one per frame
                n = min(T, W)
                pref = x.new_zeros((n, n) + tuple(x.shape[1:]))
                mask = torch.ones(n, n, dtype=torch.bool, device=x.device)
                for k in range(n):
                    pref[k, :k+1] = x[:k+1]; mask[k, :k+1] = False
                outputs = self.netG(pref, time_pad_mask=mask)
                idx = torch.arange(n, device=x.device)                            # row k -> frame k
                go_list = [outputs['pred_global_orientation'][-1][idx, idx]]
                jr_list = [outputs['pred_joint_rotation'][-1][idx, idx]]
                ing_list.append(tuple(t[idx, idx] for t in outputs['pred_swivel_extra'][-1]['recon_ingredients']))
                if T > W:
                    wins = torch.stack([x[f-W+1:f+1] for f in range(W, T)])
                    for p0 in range(0, wins.shape[0], part):
                        outputs = self.netG(wins[p0:p0 + part])
                        go_list.append(outputs['pred_global_orientation'][-1][:, -1])
                        jr_list.append(outputs['pred_joint_rotation'][-1][:, -1])
                        ing_list.append(tuple(t[:, -1] for t in outputs['pred_swivel_extra'][-1]['recon_ingredients']))
                pred_global_orientation_tensor = torch.cat(go_list, dim=0)
                pred_joint_rotation_tensor = torch.cat(jr_list, dim=0)

        # reduced (torso + limb-root) rotations, the other joints filled with identity
        ident6d = torch.tensor([1.,0.,0.,0.,1.,0.], device=self.device)
        full = ident6d.repeat(21).unsqueeze(0).expand(T, -1).clone().reshape(T, 21, 6)
        full[:, self._keep_slots, :] = pred_joint_rotation_tensor.reshape(T, len(self._keep), 6)
        full = full.reshape(T, 126)
        pred = torch.cat([pred_global_orientation_tensor, full], dim=-1).to(self.device)
        predicted_angle = utils_transform.sixd2aa(pred[:,:132].reshape(-1,6).detach()).reshape(pred[:,:132].shape[0],-1).float()

        # root translation: puts the head joint on the tracked head position
        t_head2world = self.H_global_head_trans[:,:3,3]
        body_pose_local = self.bm(**{'pose_body':predicted_angle[...,3:66], 'root_orient':predicted_angle[...,:3]})
        self.predicted_translation = -body_pose_local.Jtr[:,15,:]+t_head2world

        H_body = self.bm(**{k:v for k,v in self.H_body_param_list.items() if k in ['pose_body','trans', 'root_orient']})
        self.H_position = H_body.Jtr[:,:22,:]

        with torch.no_grad():
            if seq2seq:
                # limb-solver inputs from the averaged predictions (as in the network's swivel forward)
                jp = fk_module(pred_global_orientation_tensor, full, self.bm)[:, :22]   # (T,22,3)
                head_w = x[:, 15, 12:15]
                trans = head_w - jp[:, 15]
                gp = trans[:, None] + jp
                sw, ank, toe = fused_ex[:, :8], fused_ex[:, 8:14], fused_ex[:, 14:20]
                ankle_w = torch.stack([head_w + ank[:, :3], head_w + ank[:, 3:6]], 1)
                toe_w = torch.stack([ankle_w[:, 0] + toe[:, :3], ankle_w[:, 1] + toe[:, 3:6]], 1)
                wrist_w = x[:, [20, 21], 12:15]
                sw_ang = torch.atan2(sw[:, 4:8], sw[:, :4])
                aa = torch.cat([utils_transform.sixd2aa(pred_global_orientation_tensor.reshape(-1,6)).reshape(T,3),
                                utils_transform.sixd2aa(full.reshape(-1, 6)).reshape(T, 63)], -1)
            else:
                aa, gp, wrist_w, ankle_w, toe_w, sw_ang, trans = (torch.cat([g[k] for g in ing_list], dim=0) for k in range(7))
            ankle_w, toe_w = self._floor_clamp(ankle_w, toe_w)
            # wrist orientation = tracked hand rotation (input joints 20/21, dims 0:6), used to align the hands
            wrist_ori = utils_transform.sixd2matrot(x[:, [20, 21], 0:6].reshape(-1, 6)).reshape(T, 2, 3, 3)
            recon_pos, recon_aa = SR.analytic_limb_solver(aa, gp, wrist_w, ankle_w, toe_w, sw_ang,
                _mn.J_rest, _mn.parents, _mn.kintree22, self.bm, trans, wrist_ori=wrist_ori, temporal=True)
        self.predicted_angle_swivel = recon_aa
        self.predicted_position = recon_pos.reshape(-1, 22, 3)
        _mn.skip_recon = False

        self.netG.train()

    def _floor_clamp(self, ankle_w, toe_w):
        """Raise predicted ankles that are below the floor onto it; each toe moves with its ankle."""
        gtp = self.H_position.reshape(-1, 22, 3)
        ankle_w, toe_w = ankle_w.clone(), toe_w.clone()
        for k, j in enumerate([7, 8]):
            newz = torch.maximum(ankle_w[:, k, 2], gtp[:, j, 2].min())
            toe_w[:, k, 2] = toe_w[:, k, 2] + (newz - ankle_w[:, k, 2])
            ankle_w[:, k, 2] = newz
        return ankle_w, toe_w

    # get log_dict
    def current_log(self):
        return self.log_dict

    # predictions for evaluation
    def current_prediction(self,):
        body_parms = OrderedDict()
        pose = self.predicted_angle_swivel         # limbs solved from the swivel angles
        body_parms['pose_body'] = pose[...,3:66]
        body_parms['root_orient'] = pose[...,:3]
        body_parms['trans'] = self.predicted_translation
        body_parms['position'] = self.predicted_position
        return body_parms

    def current_gt(self, ):
        body_parms = OrderedDict()
        body_parms['pose_body'] = self.H_body_param_list['pose_body']
        body_parms['root_orient'] = self.H_body_param_list['root_orient']
        body_parms['trans'] = self.H_body_param_list['trans']
        body_parms['position'] = self.H_position
        body_parms['floor_height'] = self.H_floor_height
        return body_parms


    """ Information of netG. """

    # print network
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # print params
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # network information
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # params information
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
