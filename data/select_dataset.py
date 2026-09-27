# --------------------------------------------
# Dataset factory.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    if dataset_type == 'amass_swivel':                       # MANIKIN-S pipeline
        from data.dataset_s import AMASS_Dataset as D
    elif dataset_type in ['amass_mixed', 'cross_cmu', 'cross_bml', 'cross_hdm05']:   # MANIKIN-L pipeline
        from data.dataset_l import AMASS_Dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))
    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
