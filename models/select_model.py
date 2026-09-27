# --------------------------------------------
# Model and network factory.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

import os, importlib, functools
import torch
from torch.nn import init
from human_body_prior.body_model.body_model import BodyModel

"""Unified factories for both backbones: the config's `model` (manikin_s | manikin_l) selects the model,
netG.net_type the network."""

def define_Model(opt):
    model = opt['model']
    if model == 'manikin_s':
        from models.model_manikin import ModelManikinS as M
    elif model == 'manikin_l':
        from models.model_manikin import ModelManikinL as M
    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))
    m = M(opt)
    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m


def define_G(opt):
    return define_G_s(opt) if opt['model'] == 'manikin_s' else define_G_l(opt)


def define_G_s(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']
    network_path = 'networks.' + net_type.lower()
    net = getattr(importlib.import_module(network_path), 'ManikinNet')
    # flat netG: constructor args are everything except the meta/init keys
    _meta = ('net_type', 'init_type', 'init_bn_type', 'init_gain')
    netG = net(**{k: v for k, v in opt_net.items() if k not in _meta})

    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG


def define_G_l(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']
    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    support_dir = opt['support_dir']
    subject_gender = "male"
    bm_fname = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
    dmpl_fname = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))
    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters
    body_model = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)


    if net_type == 'ManikinL':
        from networks.manikin_l import ManikinL as net
        netG = net(body_model = body_model,
                   input_dim=opt_net['input_dim'],
                   nhead = opt_net['nhead'],
                   embed_dim=opt_net['embed_dim'],
                   single_frame_feat_dim=opt_net['single_frame_feat_dim'],
                   joint_regressor_dim = opt_net['joint_regressor_dim'],
                   joint_embed_dim=opt_net['joint_embed_dim'],
                   keep_joints=opt_net['keep_joints'])
    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG


def define_bm(opt):
    opt_bm = opt['body_model']
    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    smpl_path = opt_bm['smpl_path']
    gender_list = ["male", "female", "neutral"]
    body_model_dict={}
    for subject_gender in gender_list:
        bm_fname = os.path.join(smpl_path, 'body_models/smplh/{}/model.npz'.format(subject_gender))
        dmpl_fname = os.path.join(smpl_path, 'body_models/dmpls/{}/model.npz'.format(subject_gender))
        num_betas = 16 # number of body parameters
        num_dmpls = 8 # number of DMPL parameters
        body_model = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
        body_model_dict[subject_gender] = body_model
    return body_model_dict


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """Weight init (Kai Zhang, https://github.com/cszn/KAIR).
    init_type: default/none | normal | xavier_normal/uniform | kaiming_normal/uniform | orthogonal.
    init_bn_type: uniform | constant.
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                pass
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))


        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network defination!')
