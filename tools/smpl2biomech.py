#!/usr/bin/env python
# --------------------------------------------
# SMPL motion -> anatomical DoF model (7-DoF limbs) and biomechanical joint angles.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

"""Convert the arms and legs of SMPL(-H) motion (e.g. AMASS) to our anatomical DoF model: the knee/elbow becomes a
1-DoF hinge and the limb-root twist is fixed by the swivel angle, while the joint positions are kept. The converted
pose is also given as biomechanical joint angles.

    python tools/smpl2biomech.py --input <npz file or directory> --out <out dir>
    # every *.npz with poses / trans / betas (/ gender) under --input -> the same relative path under --out

Each output npz keeps the input fields (`poses` converted) and adds:
  swivel      (T,4)    swivel angle phi per limb [L-arm, R-arm, L-leg, R-leg]   (Fig. 3a)
  swivel_vref (T,4,3)  reference vector of each swivel angle (networks/swivel.py), fixed to the body: the
                       thorax's lateral axis for the arms (swivel 0: elbow pointing outward) and the pelvis's
                       forward axis for the legs (swivel 0: knee pointing forward)
  flexion     (T,4)    mid-joint (elbow/knee) flexion theta_flexion             (Eq. 9); 0 at full extension
  joint_angles (T,32) degrees, joint_angle_names (32,)   pelvis, lumbar (thorax w.r.t. pelvis), hip, knee, ankle,
                       subtalar, shoulder (w.r.t. thorax), elbow, forearm (pro_sup) and wrist angles
  joint_angle_residuals (T,8) degrees, joint_angle_residual_names (8,)   rotation left over after decomposing knee,
                       ankle + subtalar, elbow + pro_sup and wrist
All quantities use the subject's body shape (betas).

Add --verify to report the FK position error before/after.
Add --print_angles to print the per-limb swivel/flexion angles of the first sequence's mid-frame.

Joint angles follow the names, rotation sequences, signs, joint axes and zero posture of the OpenSim full-body model of
Rajagopal et al. (2016); all 0 = upright, straight limbs, arms at the sides, palms facing forward. The zero posture is
written as a SMPL pose -- the rest pose with both arms lowered from the T-pose to the sides and the forearms turned
palm-forward -- and each segment frame (x anterior, y superior, z to the right) is the upright body frame in that
posture, moving rigidly with its SMPL joint; the humerus is turned about its long axis so that the model's elbow axis
coincides with the elbow flexion axis of the anatomical DoF model. Pelvis, lumbar (thorax = SMPL spine3), hip and
shoulder are Z-X-Y sequences (left limbs mirrored onto the right-side axes); the knee flexes about the DoF model's
axis; ankle + subtalar, elbow + pro_sup and wrist are decomposed onto the model's joint axes. The rotation left over is
reported as a residual.
"""
import os, sys, glob, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np
import torch
import torch.nn.functional as F
from human_body_prior.body_model.body_model import BodyModel
from utils.utils_transform import aa2matrot, matrot2aa, local2global_pose
from networks import swivel as SR


# ============================================================ joint angles
# joint angle names, in output order
ANGLE_NAMES = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
          'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r',
          'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l',
          'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 'pro_sup_r', 'wrist_flex_r', 'wrist_dev_r',
          'arm_flex_l', 'arm_add_l', 'arm_rot_l', 'elbow_flex_l', 'pro_sup_l', 'wrist_flex_l', 'wrist_dev_l']
RESIDUAL_NAMES = ['knee_r', 'ankle_r', 'elbow_r', 'wrist_r', 'knee_l', 'ankle_l', 'elbow_l', 'wrist_l']

# right-side joint axes of the Rajagopal et al. (2016) model, in the parent segment frame
A_KNEE = np.array([0.0, -0.0707, -0.9975])       # walker knee, knee_angle (flexion positive)
A_ANKLE = np.array([-0.105, -0.174, 0.9791])     # talocrural, ankle_angle (dorsiflexion positive)
A_SUBTALAR = np.array([0.7872, 0.6047, -0.1209])
A_ELBOW = np.array([0.226, 0.0223, 0.9739])      # elbow_flex (flexion positive)
A_PROSUP = np.array([0.0564, 0.9984, 0.002])     # pro_sup (pronation positive, 0 = palm forward)
A_WRIST_FLEX = np.array([0.0, 0.0, 1.0])         # universal joint: wrist_flex, then wrist_dev
A_WRIST_DEV = np.array([1.0, 0.0, 0.0])
MIRROR = np.diag([1.0, 1.0, -1.0])               # sagittal-plane reflection (segment frame coordinates)
# segment axes (anterior, superior, right) in SMPL rest coordinates (SMPL: x left, y up, z forward)
C = np.stack([np.array([0.0, 0, 1]), np.array([0.0, 1, 0]), np.array([-1.0, 0, 0])], 1)
LEGS = {'r': (2, 5, 8), 'l': (1, 4, 7)}          # SMPL (hip, knee, ankle)
ARMS = {'r': (17, 19, 21), 'l': (16, 18, 20)}    # SMPL (shoulder, elbow, wrist)


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _rot(axis, q):
    """Rotation by angle q (T,) about a fixed unit axis (3,) -> (T,3,3)."""
    k = _unit(np.asarray(axis, dtype=np.float64))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    q = np.asarray(q, dtype=np.float64)[..., None, None]
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)


def _angle_about(p, q, k):
    """Angle t with rot(k, t) p closest to q; p, q (T,3), k (3,)."""
    pp = p - np.outer(p @ k, k); qp = q - np.outer(q @ k, k)
    return np.arctan2(np.cross(pp, qp) @ k, np.sum(pp * qp, -1))


def _rot_angle(R):
    return np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1))


def _perp(a):
    return _unit(np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0.0, 1, 0]))


def zxy(R):
    """R = Rz(a) Rx(b) Ry(c) -> (a, b, c)."""
    b = np.arcsin(np.clip(R[:, 2, 1], -1, 1))
    return np.arctan2(-R[:, 0, 1], R[:, 1, 1]), b, np.arctan2(-R[:, 2, 0], R[:, 2, 2])


def one_axis(R, a):
    """R ~ rot(a, q) -> (q, residual angle)."""
    p = _perp(a)
    q = _angle_about(np.broadcast_to(p, (len(R), 3)), R @ p, a)
    return q, _rot_angle(np.swapaxes(_rot(a, q), 1, 2) @ R)


def two_axis(R, a1, a2):
    """R ~ rot(a1, q1) rot(a2, q2) -> (q1, q2, residual angle)."""
    a1, a2 = _unit(a1), _unit(a2)
    q1 = _angle_about(np.broadcast_to(a2, (len(R), 3)), R @ a2, a1)
    R1 = np.swapaxes(_rot(a1, q1), 1, 2) @ R
    p = _perp(a2)
    q2 = _angle_about(np.broadcast_to(p, (len(R), 3)), R1 @ p, a2)
    return q1, q2, _rot_angle(np.swapaxes(_rot(a2, q2), 1, 2) @ R1)


def _side(a, s):
    """Right-side joint axis -> the axis for side s (mirrored, same sign meaning)."""
    return a if s == 'r' else -MIRROR @ a


def _min_rot(a, b):
    """Smallest rotation taking direction a to direction b."""
    a, b = _unit(a), _unit(b)
    ax = np.cross(a, b)
    return _rot(ax, np.array(np.arctan2(np.linalg.norm(ax), a @ b)))


def rest_frames(J_rest, hinge_ref):
    """Segment-frame offsets K (frame = SMPL global joint rotation @ K) and joint axes (in the parent segment frame),
    from the rest joints (22,3) and the knee / elbow flexion axes (dict: SMPL mid joint -> axis in its local frame).
    The humerus is turned about its long axis so that the model's elbow axis coincides with the elbow flexion axis of
    the anatomical DoF model; the knee flexes about the DoF model's own axis."""
    ey = np.array([0.0, 1, 0])
    turn = lambda a, h: _rot(ey, _angle_about(a[None], h[None], ey)[0])      # about y, taking axis a onto h
    K = {'pelvis': C, 'torso': C}
    for s, (hj, kj, aj) in LEGS.items():                                     # legs: zero posture = SMPL rest
        K['femur_' + s] = K['tibia_' + s] = K['foot_' + s] = C
        K['knee_axis_' + s] = C.T @ _unit(np.asarray(hinge_ref[kj], dtype=np.float64))
    for s, (sj, ej, wj) in ARMS.items():
        S = _min_rot(J_rest[ej] - J_rest[sj], [0.0, -1, 0])                 # arm lowered from the T-pose to the side
        u = _unit(J_rest[wj] - J_rest[ej])
        p = u if np.cross(u, [0.0, 0, 1]) @ [0.0, -1, 0] > 0 else -u         # pronation: T-pose thumb (forward) turns down
        P = _rot(p, np.array(-np.pi / 2))                                    # T-pose palm down -> palm forward
        a = _side(A_ELBOW, s)
        K['humerus_' + s] = S.T @ C @ turn(a, C.T @ S @ _unit(np.asarray(hinge_ref[ej], dtype=np.float64)))
        K['radius_' + s] = K['hand_' + s] = (S @ P).T @ C
        K['elbow_axis_' + s], K['forearm_axis_' + s] = a, _side(A_PROSUP, s)
        K['wrist_axes_' + s] = (_side(A_WRIST_FLEX, s), _side(A_WRIST_DEV, s))
    return K


def angles_from_rotations(Rg, K, up=2):
    """Rg: (T,22,3,3) SMPL global joint rotations (world frame, axis `up` vertical); K: rest_frames().
    Returns (T, len(ANGLE_NAMES)) angles and (T, len(RESIDUAL_NAMES)) residual angles, both in degrees, and the ground
    frame G (world, columns x forward, y up, z right). G faces the sequence's mean heading, so pelvis_rotation is the
    yaw relative to it."""
    Rg = np.asarray(Rg, dtype=np.float64)
    F = {'pelvis': Rg[:, 0] @ K['pelvis'], 'torso': Rg[:, 9] @ K['torso']}
    for s, (hj, kj, aj) in LEGS.items():
        F['femur_' + s], F['tibia_' + s], F['foot_' + s] = (Rg[:, j] @ K[n + s] for j, n in ((hj, 'femur_'), (kj, 'tibia_'), (aj, 'foot_')))
    for s, (sj, ej, wj) in ARMS.items():
        F['humerus_' + s], F['radius_' + s], F['hand_' + s] = (Rg[:, j] @ K[n + s] for j, n in ((sj, 'humerus_'), (ej, 'radius_'), (wj, 'hand_')))
    rel = lambda p, c: np.swapaxes(F[p], 1, 2) @ F[c]
    mirrored = lambda R, s: MIRROR @ R @ MIRROR if s == 'l' else R

    # ground: vertical = world `up`, x = mean horizontal heading of the pelvis
    fwd = F['pelvis'][:, :, 0].copy(); fwd[:, up] = 0
    h = _unit(fwd.mean(0)); v = np.eye(3)[up]
    G = np.stack([h, v, np.cross(h, v)], 1)
    out, res = {}, {}
    out['pelvis_tilt'], out['pelvis_list'], out['pelvis_rotation'] = zxy(G.T @ F['pelvis'])
    out['lumbar_extension'], out['lumbar_bending'], out['lumbar_rotation'] = zxy(rel('pelvis', 'torso'))  # thorax (SMPL spine3)
    for s in ('r', 'l'):
        out['hip_flexion_' + s], out['hip_adduction_' + s], out['hip_rotation_' + s] = zxy(mirrored(rel('pelvis', 'femur_' + s), s))
        out['knee_angle_' + s], res['knee_' + s] = one_axis(rel('femur_' + s, 'tibia_' + s), K['knee_axis_' + s])
        out['ankle_angle_' + s], out['subtalar_angle_' + s], res['ankle_' + s] = two_axis(
            rel('tibia_' + s, 'foot_' + s), _side(A_ANKLE, s), _side(A_SUBTALAR, s))
        out['arm_flex_' + s], out['arm_add_' + s], out['arm_rot_' + s] = zxy(mirrored(rel('torso', 'humerus_' + s), s))
        out['elbow_flex_' + s], out['pro_sup_' + s], res['elbow_' + s] = two_axis(
            rel('humerus_' + s, 'radius_' + s), K['elbow_axis_' + s], K['forearm_axis_' + s])
        out['wrist_flex_' + s], out['wrist_dev_' + s], res['wrist_' + s] = two_axis(
            rel('radius_' + s, 'hand_' + s), *K['wrist_axes_' + s])
    angles = np.degrees(np.stack([out[n] for n in ANGLE_NAMES], 1))
    lo = np.array([-90.0 if n.startswith('pro_sup') else -180.0 for n in ANGLE_NAMES])   # pro_sup in [-90, 270)
    angles = (angles - lo) % 360 + lo
    residuals = np.degrees(np.stack([res[n] for n in RESIDUAL_NAMES], 1))
    return angles.astype(np.float32), residuals.astype(np.float32), G


# ============================================================ anatomical DoF model
def smooth_twist(R_g, n, r, lam):
    """Twist regularisation over ONE time-ordered sequence. Each root orientation R_t may turn about its
    root->end axis n_t by psi_t, which keeps the end effector and moves the mid joint by about r_t * psi_t (r_t: its
    distance from the axis). psi minimises  sum_t (r_t psi_t)^2 + lam * sum_t (psi_t - psi_{t-1} + phi_t)^2,  where
    phi_t is the twist of R_t R_{t-1}^T about n_t: mid-joint displacement against the frame-to-frame change of the
    limb-root twist, so the twist follows the data where the limb is bent and stays smooth where it is nearly
    straight. Solved as one tridiagonal linear system."""
    Rn = R_g.detach().cpu().double().numpy(); nn = n.detach().cpu().double().numpy()
    a = r.detach().cpu().double().numpy() ** 2 + 1e-12
    T = len(Rn)
    if T < 2:
        return R_g
    e = np.where(np.abs(nn[:, :1]) < 0.9, [[1.0, 0, 0]], [[0.0, 1, 0]])
    p = np.cross(nn, e); p /= np.linalg.norm(p, axis=-1, keepdims=True)
    q = np.einsum('tij,tj->ti', Rn[1:] @ np.swapaxes(Rn[:-1], 1, 2), p[1:])   # previous-frame direction, carried along
    q -= np.sum(q * nn[1:], -1, keepdims=True) * nn[1:]
    phi = np.zeros(T)
    phi[1:] = np.arctan2(np.sum(np.cross(p[1:], q) * nn[1:], -1), np.sum(p[1:] * q, -1))
    diag, off, rhs = a.copy(), np.full(T - 1, -lam), np.zeros(T)
    diag[1:] += lam; diag[:-1] += lam
    rhs[1:] -= lam * phi[1:]; rhs[:-1] += lam * phi[1:]
    for t in range(1, T):                                                    # Thomas algorithm
        m = off[t - 1] / diag[t - 1]
        diag[t] -= m * off[t - 1]; rhs[t] -= m * rhs[t - 1]
    psi = np.zeros(T); psi[-1] = rhs[-1] / diag[-1]
    for t in range(T - 2, -1, -1):
        psi[t] = (rhs[t] - off[t] * psi[t + 1]) / diag[t]
    K = np.zeros((T, 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -nn[:, 2], nn[:, 1], nn[:, 2], -nn[:, 0], -nn[:, 1], nn[:, 0]
    s, c = np.sin(psi)[:, None, None], np.cos(psi)[:, None, None]
    out = (np.eye(3) + s * K + (1 - c) * K @ K) @ Rn
    return torch.as_tensor(out, dtype=R_g.dtype, device=R_g.device)


def to_biomechanical(poses_aa, trans, betas, body_model, device, fps=30.0, twist_tau=0.05, chunk=4096):
    """poses_aa: (T, >=66) axis-angle SMPL pose of one sequence; trans: (T,3); betas: (>=16,) body shape;
    fps: its frame rate. Returns:
      corrected: (T,66) body pose converted to the anatomical DoF model (limb joints solved as
                 1-DoF hinge + swivel twist),
      swivel:    (T,4) swivel angle phi per limb [L-arm, R-arm, L-leg, R-leg] (radians),
      vref:      (T,4,3) swivel reference vector per limb (thorax lateral axis / pelvis forward axis),
      flexion:   (T,4) mid-joint (elbow/knee) flexion angle theta_flexion per limb, same order (radians),
                 as defined in the paper Eq. (9): theta_flexion = pi - arccos((lbm^2 + lme^2 - ||pe-pb||^2)
                 / (2 lbm lme)) -- 0 for a fully extended limb, increasing as it bends.

    Every joint position is known (from FK of the input pose), so the mid joint is given directly and the
    Analytic Arm/Leg Solver needs no swivel angle -- the input base/mid/end positions are fed straight to it.
    swivel (base swing+twist DoF, Fig. 3a) and flexion (mid DoF, Eq. 9) are the two internal scalar DoF of
    the limb (base position and end-effector pose are given/predicted); both are derived from
    the input positions (swivel via positions_to_swivel).

    Where a limb is nearly straight, the joint positions hardly determine the twist of its root; the twist is then
    regularised over the sequence (smooth_twist) with a time constant of twist_tau seconds where
    the mid joint lies 3 cm off the base->end axis (shorter where the limb is more bent)."""
    T = poses_aa.shape[0]
    aa = torch.as_tensor(np.asarray(poses_aa)[:, :66]).float().to(device)
    tr = torch.as_tensor(np.asarray(trans)).float().to(device)
    kintree = body_model.kintree_table[0][:22].long().to(device)
    parents = kintree.tolist()
    bb = torch.as_tensor(np.asarray(betas)[:body_model.num_betas]).float().to(device)[None]
    J_rest = body_model(betas=bb).Jtr[0, :22].detach()
    # joint world positions (forward kinematics in chunks) and per-joint global rotations (wrist orientation)
    gp = torch.cat([body_model(root_orient=aa[s:s + chunk, :3], pose_body=aa[s:s + chunk, 3:66], trans=tr[s:s + chunk],
                               betas=bb.expand(len(aa[s:s + chunk]), -1)).Jtr[:, :22].detach() for s in range(0, T, chunk)])
    Rg = local2global_pose(aa2matrot(aa.reshape(-1, 3)).reshape(T, 22, 9), kintree).reshape(T, 22, 3, 3)
    wrist_ori = Rg[:, [20, 21]]                                                 # known wrist global rotation
    # Targets: the input base/mid/end (and toe for legs) joint positions -- known directly, no swivel needed.
    targets = []
    sw = torch.zeros(T, 4, device=device)
    fx = torch.zeros(T, 4, device=device)
    vref_all = torch.zeros(T, 4, 3, device=device)
    for li, (ja, jb, jc) in enumerate(SR.LIMBS):
        toe = None if jc in (20, 21) else gp[:, jc + 3]                         # legs: toe joint (7->10, 8->11)
        # Swivel direction k = the input mid joint's side of the base->end axis. A nearly straight limb (< 10 deg
        # flexion) whose mid joint lies behind is hyperextended: solve with the negative root. "Behind" is judged
        # against the foot direction for the knee (it flexes towards the toes) and against the hinge's flexion side
        # for the elbow.
        n_ax = F.normalize(gp[:, jc] - gp[:, ja], dim=-1)
        km = gp[:, jb] - gp[:, ja]; km = km - (km * n_ax).sum(-1, keepdim=True) * n_ax
        a_h = F.normalize(torch.tensor(SR.HINGE_REF[jb], device=device, dtype=gp.dtype), dim=0)
        h = torch.einsum('tij,j->ti', Rg[:, jb], a_h); h = h - (h * n_ax).sum(-1, keepdim=True) * n_ax
        kh = F.normalize(torch.cross(h, n_ax, dim=-1), dim=-1) * SR.hinge_side_sign(J_rest[jb] - J_rest[ja], J_rest[jc] - J_rest[jb], a_h)
        r = km.norm(dim=-1) / (J_rest[jb] - J_rest[ja]).norm()
        kdir = torch.where((r > 1e-6).unsqueeze(-1), F.normalize(km, dim=-1), kh)
        ref = kh
        if toe is not None:
            ft = toe - gp[:, jc]; fp = ft - (ft * n_ax).sum(-1, keepdim=True) * n_ax
            ref = torch.where((fp.norm(dim=-1) > 0.3 * ft.norm(dim=-1)).unsqueeze(-1), F.normalize(fp, dim=-1), kh)
        u = F.normalize(gp[:, ja] - gp[:, jb], dim=-1)
        v = F.normalize(gp[:, jc] - gp[:, jb], dim=-1)
        fx[:, li] = np.pi - torch.acos((u * v).sum(-1).clamp(-1.0, 1.0))         # included-angle flexion (Eq. 9)
        bend = torch.where(((kdir * ref).sum(-1) < 0) & (fx[:, li] < np.radians(10)), -1.0, 1.0)
        targets.append((ja, jb, jc, gp[:, jb], gp[:, jc], toe, kdir, bend))
        # swivel annotation: angle of the (given) mid joint about the base->end axis, from a body-fixed reference
        if jc in (20, 21):                                                      # arms: thorax lateral axis
            vref = Rg[:, 9, :, 0] * (-1.0 if jc == 20 else 1.0)                 # (0: elbow pointing outward)
        else:                                                                   # legs: pelvis forward axis
            vref = -Rg[:, 0, :, 2]                                              # (0: knee pointing forward)
        vref = vref.unsqueeze(1)
        vref_all[:, li] = vref.squeeze(1)
        a, b, c = gp[:, ja].unsqueeze(1), gp[:, jb].unsqueeze(1), gp[:, jc].unsqueeze(1)
        sw[:, li] = SR.positions_to_swivel(a - a, b - a, c - a, vref).squeeze(-1).squeeze(-1)
    # Analytic Arm/Leg Solver -> anatomically correct local rotations, whole sequence at once
    lam = (twist_tau * fps * 0.03) ** 2
    _, corrected = SR.solve_limb_angles(aa, gp, targets, J_rest, parents, kintree, body_model, tr, wrist_ori=wrist_ori,
                                        limits=False, twist_fn=lambda R, n, r: smooth_twist(R, n, r, lam), fk=False)
    # The ankle keeps its 3 DoF: the foot keeps its input world orientation, expressed relative to the new shank.
    Rg_c = local2global_pose(aa2matrot(corrected.reshape(-1, 3)).reshape(T, 22, 9), kintree).reshape(T, 22, 3, 3)
    cols = list(corrected.split(3, dim=-1))
    for (ja, jb, jc) in SR.LIMBS:
        if jc in (20, 21):                                                      # arms keep the solver's result
            continue
        cols[jc] = matrot2aa(torch.bmm(Rg_c[:, jb].transpose(1, 2), Rg[:, jc]))  # new_shank^{-1} @ orig_foot_world
    corrected = torch.cat(cols, dim=-1)
    return corrected.detach().cpu(), sw.detach().cpu(), vref_all.detach().cpu(), fx.detach().cpu()


def _fk_positions(body_model, aa66, trans, betas, device):
    aa66 = torch.as_tensor(aa66).float().to(device); trans = torch.as_tensor(trans).float().to(device)
    bb = torch.as_tensor(np.asarray(betas)[:body_model.num_betas]).float().to(device)[None].expand(aa66.shape[0], -1)
    return body_model(root_orient=aa66[:, :3], pose_body=aa66[:, 3:66], trans=trans, betas=bb).Jtr[:, :22].detach()


def joint_angles(aa66, betas, body_model):
    """Joint angles, residuals (degrees) and ground frame of a (T,66) axis-angle pose; world z up (AMASS)."""
    aa = torch.as_tensor(aa66).float().cpu()
    kt = body_model.kintree_table[0][:22].long().cpu()
    Rg = local2global_pose(aa2matrot(aa.reshape(-1, 3)).reshape(aa.shape[0], 22, 9), kt).reshape(-1, 22, 3, 3)
    bb = torch.as_tensor(np.asarray(betas)[:body_model.num_betas]).float().to(body_model.shapedirs.device)[None]
    J_rest = body_model(betas=bb).Jtr[0, :22].detach().cpu().numpy().astype(np.float64)
    return angles_from_rotations(Rg.numpy(), rest_frames(J_rest, SR.HINGE_REF), up=2)


def _print_angles(swivel, flexion, t):
    """Pretty-print the per-limb swivel and flexion angles of one frame (degrees)."""
    r2d = 180.0 / np.pi
    print(f'\nAnatomical limb angles (frame {t}, degrees):')
    for li, nm in enumerate(['L-arm', 'R-arm', 'L-leg', 'R-leg']):
        print(f'  {nm}:  swivel={swivel[t, li] * r2d:7.1f}   flexion={flexion[t, li] * r2d:6.1f}')
    print()


# ============================================================ command line
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, required=True, help='SMPL(-H) npz file, or a directory searched recursively for *.npz')
    p.add_argument('--out', type=str, required=True, help='output directory (mirrors the input layout)')
    p.add_argument('--support_data', type=str, default=os.path.join(ROOT, 'support_data'))
    p.add_argument('--verify', action='store_true', help='report FK position error before/after')
    p.add_argument('--fps', type=float, default=30.0, help='frame rate of npz files without mocap_framerate')
    p.add_argument('--print_angles', action='store_true',
                   help='print the per-limb swivel and flexion angles of the first sequence, mid-frame')
    a = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bms = {g: BodyModel(bm_fname=os.path.join(a.support_data, f'body_models/smplh/{g}/model.npz'), num_betas=16,
                        num_dmpls=8, dmpl_fname=os.path.join(a.support_data, f'body_models/dmpls/{g}/model.npz')).to(device)
           for g in ['male', 'female', 'neutral']}

    if os.path.isfile(a.input):
        files, base = [a.input], os.path.dirname(a.input)
    else:
        files, base = sorted(glob.glob(os.path.join(a.input, '**', '*.npz'), recursive=True)), a.input
    print(f'{len(files)} npz files under {a.input}')
    max_err, n = 0.0, 0
    for f in files:
        bdata = np.load(f, allow_pickle=True)
        if not {'poses', 'trans', 'betas'} <= set(bdata.files):
            continue                                                            # e.g. AMASS shape.npz
        gender = str(bdata['gender']) if 'gender' in bdata else 'neutral'
        bm = bms.get(gender if gender in bms else 'neutral', bms['neutral'])
        poses, trans, betas = bdata['poses'], bdata['trans'], bdata['betas']
        fps = next((float(bdata[k]) for k in ('mocap_framerate', 'mocap_frame_rate') if k in bdata.files), a.fps)
        corrected, swivel, vref, flexion = to_biomechanical(poses, trans, betas, bm, device, fps)  # (T,66), (T,4), (T,4,3), (T,4)
        angles, residuals, _ = joint_angles(corrected.numpy(), betas, bm)

        if a.verify:
            p0 = _fk_positions(bm, poses[:, :66], trans, betas, device)
            p1 = _fk_positions(bm, corrected.numpy(), trans, betas, device)
            max_err = max(max_err, (p0 - p1).norm(dim=-1).max().item())

        if a.print_angles and n == 0:
            _print_angles(swivel, flexion, t=len(corrected) // 2)

        new_poses = np.array(bdata['poses'], copy=True)
        new_poses[:, :66] = corrected.numpy()                                   # keep hands/face poses as-is
        dst = os.path.join(a.out, os.path.relpath(f, base))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        save = {k: bdata[k] for k in bdata.files}
        save['poses'] = new_poses
        save['swivel'] = swivel.numpy()                                         # (T,4) swivel phi [L-arm,R-arm,L-leg,R-leg]
        save['swivel_vref'] = vref.numpy()                                      # (T,4,3) swivel reference vectors
        save['flexion'] = flexion.numpy()                                       # (T,4) mid flexion theta_flexion (Eq. 9)
        save['joint_angles'] = angles                                           # (T,32) degrees
        save['joint_angle_names'] = np.array(ANGLE_NAMES)
        save['joint_angle_residuals'] = residuals                               # (T,8) degrees
        save['joint_angle_residual_names'] = np.array(RESIDUAL_NAMES)
        np.savez(dst, **save)
        n += 1
        if n % 50 == 0:
            print(f'  {n} done')
    if a.verify:
        print(f'max FK position error (original vs biomechanical): {max_err * 100:.4f} cm')
    print(f'done: {n} sequences')


if __name__ == '__main__':
    main()
