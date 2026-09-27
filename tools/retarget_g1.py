#!/usr/bin/env python
# --------------------------------------------
# Retarget SMPL motion to the Unitree G1 humanoid through the swivel parameterization.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

"""Retarget SMPL(-H) motion to the Unitree G1 humanoid (29 DoF) in closed form, through the swivel parameterization.

    python tools/smpl2biomech.py --input <motion npz or dir> --out <biomech dir>
    python tools/retarget_g1.py --input <biomech dir> --g1_xml <mujoco_menagerie>/unitree_g1/g1.xml --out <g1 dir>

Input: outputs of tools/smpl2biomech.py (poses, trans, betas, gender, swivel, swivel_vref; world z up). Per limb
(2 arms + 2 legs), without iterative IK:
  1. elbow / knee angle  from the robot's base->end distance
  2. swivel              the human swivel angle and its reference vector give the elbow / knee direction around the
                         base->end axis, which is copied to the robot unchanged
  3. base rotation       the rotation taking the robot's (base->end, elbow side) onto (that axis, that direction)
  4. base joints         that rotation decomposed onto G1's three shoulder / hip joint axes (Paden-Kahan subproblems)
The limb is solved on a copy of the G1 model whose three shoulder / hip axes meet in one point; one closed-form pass
then shifts the target by the end-point offset of the real model. Wrist (3 DoF), ankle (2 DoF) and waist (3 DoF) are
rotation decompositions; limb targets are scaled by the robot / human limb-length ratio.

Output npz per sequence: qpos (T, 36) in MuJoCo order (root position, root quaternion wxyz, 29 joints), joint_names
(29,), fps, end_error (T, 4) the end-point error (m) of [L-arm, R-arm, L-leg, R-leg], and leg_scale the robot / human
leg length ratio.
Requires `pip install mujoco` and the G1 model of MuJoCo Menagerie (https://github.com/google-deepmind/mujoco_menagerie).
"""
import os, re, glob, time, argparse
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = np.array([[0., 0, 1], [1, 0, 0], [0, 1, 0]])       # SMPL canonical (left, up, fwd) -> G1 canonical (fwd, left, up)

# name: (human base, mid, end, swivel idx, robot base body, 4 limb joints, robot mid body, robot end body,
#        distal joints (wrist 3 / ankle 2), distal link)
LIMBS = {
    'L_arm': (16, 18, 20, 0, 'torso_link', ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint'],
              'left_elbow_link', 'left_wrist_pitch_link', ['left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint'], 'left_wrist_yaw_link'),
    'R_arm': (17, 19, 21, 1, 'torso_link', ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'],
              'right_elbow_link', 'right_wrist_pitch_link', ['right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'], 'right_wrist_yaw_link'),
    'L_leg': (1, 4, 7, 2, 'pelvis', ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint'],
              'left_knee_link', 'left_ankle_pitch_link', ['left_ankle_pitch_joint', 'left_ankle_roll_joint'], 'left_ankle_roll_link'),
    'R_leg': (2, 5, 8, 3, 'pelvis', ['right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint'],
              'right_knee_link', 'right_ankle_pitch_link', ['right_ankle_pitch_joint', 'right_ankle_roll_joint'], 'right_ankle_roll_link'),
}
WAIST = ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']


# ----------------------------------------------------------------------------- rotation helpers
def unit(x): return x / (np.linalg.norm(x) + 1e-12)


def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def rot(k, q):
    """Rodrigues: rotation by q about unit axis k."""
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def aa2mat(aa):
    th = np.linalg.norm(aa)
    return np.eye(3) if th < 1e-12 else rot(aa / th, th)


def sp1(p, q, k):
    """Subproblem 1: angle t with rot(k,t) p ~ q (least squares when |p_perp| != |q_perp|)."""
    pp, qp = p - k * (k @ p), q - k * (k @ q)
    return np.arctan2(k @ np.cross(pp, qp), pp @ qp)


def sp2(p, q, k1, k2):
    """Subproblem 2: all (t1, t2) with rot(k1,t1) rot(k2,t2) p = q  (i.e. rot(k2,t2)p = c = rot(k1,-t1)q).
    Least-squares (1 solution) when the two cones do not intersect."""
    k12 = k1 @ k2
    a1, a2 = k1 @ q, k2 @ p                                     # c.k1 = q.k1 ,  c.k2 = p.k2
    den = 1 - k12 ** 2
    al = (a1 - k12 * a2) / den; be = (a2 - k12 * a1) / den
    k3 = np.cross(k1, k2)
    g2 = (p @ p - al ** 2 - be ** 2 - 2 * al * be * k12) / (k3 @ k3)
    gs = [0.0] if g2 <= 0 else [np.sqrt(g2), -np.sqrt(g2)]
    out = []
    for g in gs:
        c = al * k1 + be * k2 + g * k3
        out.append((-sp1(q, c, k1), sp1(p, c, k2)))
    return out


def euler3(R, a1, a2, a3):
    """All (q1,q2,q3) with rot(a1,q1) rot(a2,q2) rot(a3,q3) = R for arbitrary (non-parallel) unit axes."""
    sols = []
    for t1, t2 in sp2(a3, R @ a3, a1, a2):                      # rot(a3,q3) fixes a3
        x = unit(np.cross(a3, [1., 0, 0] if abs(a3[0]) < 0.9 else [0., 1, 0]))
        t3 = sp1(x, rot(a2, -t2) @ rot(a1, -t1) @ R @ x, a3)
        sols.append(np.array([t1, t2, t3]))
    return sols


def pick(sols, rng, prev):
    """Branch choice: least joint-limit violation, then closest to the previous frame."""
    def cost(q):
        viol = np.sum(np.maximum(rng[:, 0] - q, 0) + np.maximum(q - rng[:, 1], 0))
        return (round(viol, 4), 0 if prev is None else np.sum(np.abs(wrap(q - prev))))
    best = min(sols, key=cost)
    return np.clip(best, rng[:, 0], rng[:, 1]), cost(best)[0] > 0


# ----------------------------------------------------------------------------- human (SMPL-H) kinematics
def smplh_kinematics(aa, trans, betas, model_path):
    """(T,66) axis-angle, (T,3) translation, betas -> joint positions (T,22,3), local and global rotations
    (T,22,3,3), rest joints (22,3)."""
    m = np.load(model_path, allow_pickle=True)
    parents = m['kintree_table'][0][:22].astype(np.int64); parents[0] = -1
    b = np.zeros(m['shapedirs'].shape[-1]); bb = np.asarray(betas).reshape(-1)[:len(b)]; b[:len(bb)] = bb
    J = (m['J_regressor'] @ (m['v_template'] + m['shapedirs'] @ b))[:22]
    T = len(aa)
    loc = np.stack([[aa2mat(aa[t, 3 * j:3 * j + 3]) for j in range(22)] for t in range(T)])
    glob_ = np.zeros_like(loc); pos = np.zeros((T, 22, 3))
    for j in range(22):
        if parents[j] < 0:
            glob_[:, j], pos[:, j] = loc[:, j], J[j]
        else:
            glob_[:, j] = glob_[:, parents[j]] @ loc[:, j]
            pos[:, j] = pos[:, parents[j]] + np.einsum('tij,j->ti', glob_[:, parents[j]], J[j] - J[parents[j]])
    return pos + np.asarray(trans)[:, None], loc, glob_, J


# ----------------------------------------------------------------------------- robot models
def load_models(g1_xml):
    """The G1 model and a copy whose three shoulder / hip axes meet in one point."""
    xml = open(g1_xml).read()
    def setpos(x, body, new):
        return re.sub(rf'(<body name="{body}" pos=")[^"]*(")', rf'\g<1>{new}\g<2>', x)
    ideal = xml
    for s, sg in (('left', 1), ('right', -1)):
        ideal = setpos(ideal, f'{s}_shoulder_roll_link', f'0 {sg * 0.038} 0')
        ideal = setpos(ideal, f'{s}_shoulder_yaw_link', f'0 0 {-0.1032 - 0.013831}')
        ideal = setpos(ideal, f'{s}_hip_roll_link', f'0 {sg * 0.052} 0')
        ideal = setpos(ideal, f'{s}_hip_yaw_link', f'0 0 {-0.12412 - 0.030465}')
    cwd = os.getcwd(); os.chdir(os.path.dirname(os.path.abspath(g1_xml)))    # mesh paths are relative to the xml
    try:
        mr, mi = mujoco.MjModel.from_xml_string(xml), mujoco.MjModel.from_xml_string(ideal)
    finally:
        os.chdir(cwd)
    return mr, mi


def zero_data(m):
    d = mujoco.MjData(m); d.qpos[:] = 0; d.qpos[3] = 1; mujoco.mj_kinematics(m, d); return d


class Limb:
    """Closed-form limb solver built from the concurrent-axes model at q = 0, everything in the limb's base-body frame."""

    def __init__(self, name, mi, mr):
        (self.hb, self.hm, self.he, self.si, base, joints, mid, end, distal, dlink) = LIMBS[name]
        self.name = name
        d = zero_data(mi)
        b = mi.body(base).id
        Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b].copy()
        to_b = lambda x: Rb.T @ (x - pb)
        jid = [mi.joint(n).id for n in joints]
        self.ax = [Rb.T @ d.xaxis[j] for j in jid]
        self.O = to_b(d.xanchor[jid[1]])                         # concurrency point (roll anchor)
        for j, a in zip(jid[:3], self.ax[:3]):
            r = to_b(d.xanchor[j]) - self.O
            assert np.linalg.norm(r - a * (a @ r)) < 1e-6, f'{name}: unexpected g1.xml ({mi.joint(j).name})'
        M0, P0 = to_b(d.xpos[mi.body(mid).id]), to_b(d.xpos[mi.body(end).id])
        self.u, self.v = M0 - self.O, P0 - M0                    # O->elbow, elbow->end at q = 0
        self.d1, self.d2 = np.linalg.norm(self.u), np.linalg.norm(self.v)
        self.qadr = np.array([mr.jnt_qposadr[mr.joint(n).id] for n in joints])
        self.rng = mr.jnt_range[[mr.joint(n).id for n in joints]].copy()
        # hinge: |u + rot(a4,q) v|^2 = |u|^2 + |v|^2 + 2 (A + B cos q + C sin q)
        a4, v = self.ax[3], self.v
        vpar = a4 * (a4 @ v)
        self.A, self.B, self.Cc = self.u @ vpar, self.u @ (v - vpar), self.u @ np.cross(a4, v)
        self.q_straight = np.arctan2(self.Cc, self.B)
        self.bend = np.sign(wrap(self.rng[3].mean() - self.q_straight))   # which side of "straight" flexes
        # elbow-side sign (constant per limb): elbow direction vs (hinge x base->end) at a bent pose
        w0 = self._at_hinge(self.q_straight + self.bend * 1.0)
        s0 = self.u - (self.u @ unit(w0)) * unit(w0)
        self.sig = np.sign(s0 @ self._hinge_side(w0))
        # distal joints (wrist / ankle), axes in the mid-link frame at q = 0
        md = mi.body(mid).id; Rm = d.xmat[md].reshape(3, 3)
        self.dax = [Rm.T @ d.xaxis[mi.joint(n).id] for n in distal]
        self.Q0 = Rm.T @ d.xmat[mi.body(dlink).id].reshape(3, 3)
        self.dadr = np.array([mr.jnt_qposadr[mr.joint(n).id] for n in distal])
        self.drng = mr.jnt_range[[mr.joint(n).id for n in distal]].copy()
        self.R_dist0 = d.xmat[mi.body(dlink).id].reshape(3, 3).copy()
        self.p_fwd = Rm.T @ np.array([1., 0, 0])                 # foot forward (world +x at q = 0) in the knee-link frame
        self.mid_body, self.end_body, self.base_body = mid, end, base
        self.prev, self.dprev = None, None

    def _at_hinge(self, q4):
        return self.u + rot(self.ax[3], q4) @ self.v

    def _hinge_side(self, w0):
        e1 = unit(w0); h = self.ax[3] - (self.ax[3] @ e1) * e1
        return unit(np.cross(unit(h), e1))

    def set_human_rest(self, J_rest):
        """Neutral correspondences. Arms: G1 at q = 0 (arm hanging, elbow flexed 90 deg forward, palm medial) <->
        the SMPL hand with fingers forward, palm medial, thumb up.  Legs: SMPL rest toe direction <-> G1 foot forward."""
        if self.he in (20, 21):
            H_q0 = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]]) if self.he == 20 else np.array([[0., -1, 0], [0, 0, 1], [-1, 0, 0]])
            self.K = H_q0.T @ C.T @ self.R_dist0                   # R_hand_robot = R_hand_human @ K  (world)
        else:
            t = unit(J_rest[self.he + 3] - J_rest[self.he])        # SMPL rest ankle->toe (canonical: y up, z fwd)
            self.toe_elev = np.arctan2(-t[1], t[2])                # how far the rest toe bone points below horizontal

    def solve(self, P, k):
        """P: end target, k: unit elbow direction (perp to O->P); both in the base frame. -> 4 joint angles."""
        w = P - self.O; d = np.linalg.norm(w); n = w / d
        # 1. hinge from distance
        K = (d * d - self.u @ self.u - self.v @ self.v) / 2 - self.A
        Rr = np.hypot(self.B, self.Cc)
        acs = np.arccos(np.clip(K / Rr, -1, 1))
        q4 = np.clip(wrap(self.q_straight + self.bend * acs), *self.rng[3])
        w0 = self._at_hinge(q4)
        # 2-3. base rotation: (w0-direction, elbow side) -> (n, k)
        e1s = unit(w0)
        s0 = self.u - (self.u @ e1s) * e1s                      # elbow side, blended with the hinge side as the limb straightens
        wgt = np.clip((np.linalg.norm(s0) / self.d1 - 0.05) / 0.10, 0, 1)
        e2s = unit(wgt * unit(s0) + (1 - wgt) * self.sig * self._hinge_side(w0))
        k = unit(k - (k @ n) * n)
        Rs = np.stack([n, k, np.cross(n, k)], 1) @ np.stack([e1s, e2s, np.cross(e1s, e2s)], 1).T
        # 4. decompose onto the three base joints
        q123, clipped = pick(euler3(Rs, *self.ax[:3]), self.rng[:3], self.prev)
        q = np.r_[q123, q4]; self.prev = q123
        return q, clipped or not (self.rng[3, 0] < q4 < self.rng[3, 1])

    def solve_distal(self, R_mid_world, R_dist_h, pos_h):
        """Wrist: match the human global hand orientation (3 DoF). Ankle: point the foot along the human ankle->toe
        direction (2 DoF)."""
        if len(self.dax) == 3:
            Mt = R_mid_world.T @ (R_dist_h @ self.K) @ self.Q0.T
            q, _ = pick(euler3(Mt, *self.dax), self.drng, self.dprev)
        else:
            t = unit(pos_h[self.he + 3] - pos_h[self.he])
            l = unit(np.cross([0., 0, 1], t))
            f = rot(l, -self.toe_elev) @ t                         # rest toe elevation removed -> foot forward
            q, _ = pick([np.array(x) for x in sp2(self.p_fwd, R_mid_world.T @ f, *self.dax)], self.drng, self.dprev)
        self.dprev = q
        return q


# ----------------------------------------------------------------------------- retarget one sequence
def retarget(pos, Rg, sw, vref, J, mr, mi, limbs, stand_height):
    """pos (T,22,3), Rg (T,22,3,3): human joint positions / global rotations (world, z up); sw (T,4), vref (T,4,3):
    swivel angles and their reference vectors; J (22,3): human rest joints. Returns qpos (T, nq), end-point error (T,4),
    fraction of frames clipped at a joint limit per limb (4,), the solve time per frame (ms) and the robot / human leg
    length ratio that scales the root path."""
    T = len(pos)
    for L in limbs.values():
        L.set_human_rest(J); L.prev = L.dprev = None
    # scale factors (robot / human limb length); legs also scale the root trajectory
    s = {n: (L.d1 + L.d2) / (np.linalg.norm(J[L.hm] - J[L.hb]) + np.linalg.norm(J[L.he] - J[L.hm])) for n, L in limbs.items()}
    s_leg = 0.5 * (s['L_leg'] + s['R_leg'])
    waist_rng = mr.jnt_range[[mr.joint(n).id for n in WAIST]].copy()
    waist_adr = np.array([mr.jnt_qposadr[mr.joint(n).id] for n in WAIST])
    dr = mujoco.MjData(mr)
    qpos = np.zeros((T, mr.nq)); wprev = None
    targets = np.zeros((T, 4, 3)); err = np.zeros((T, 4)); clip_cnt = np.zeros(4)
    hipc_i = zero_data(mi); hipc0 = np.mean([hipc_i.xanchor[mi.joint(j).id] for j in ('left_hip_roll_joint', 'right_hip_roll_joint')], 0)
    t0 = time.perf_counter()
    for t in range(T):
        q = np.zeros(mr.nq); q[3] = 1
        # root: pelvis orientation, hip-centre position scaled by the leg ratio
        Rp = Rg[t, 0] @ C.T
        q[:3] = s_leg * 0.5 * (pos[t, 1] + pos[t, 2]) - Rp @ hipc0
        mujoco.mju_mat2Quat(q[3:7], Rp.flatten())
        # waist
        Rw = C @ (Rg[t, 0].T @ Rg[t, 9]) @ C.T
        wq, _ = pick(euler3(Rw, np.array([0., 0, 1]), np.array([1., 0, 0]), np.array([0., 1, 0])), waist_rng, wprev)
        wprev = wq; q[waist_adr] = wq
        dr.qpos[:] = q; mujoco.mj_kinematics(mr, dr)
        base = {b: (dr.xmat[mr.body(b).id].reshape(3, 3).copy(), dr.xpos[mr.body(b).id].copy()) for b in ('pelvis', 'torso_link')}
        # limbs: solve on the concurrent-axes model, then one closed-form compensation pass on the real model
        tgt_b = {}
        for n, L in limbs.items():
            Rb, pb = base[L.base_body]
            S, W = pos[t, L.hb], pos[t, L.he]
            nvec = unit(W - S)
            vr = vref[t, L.si]
            u_ = unit(-vr + (vr @ nvec) * nvec); v_ = np.cross(u_, nvec)          # swivel frame (networks/swivel.py)
            k_w = u_ * np.cos(sw[t, L.si]) + v_ * np.sin(sw[t, L.si])              # elbow / knee direction, world
            P_w = Rb @ L.O + pb + s[n] * (W - S)
            targets[t, L.si] = P_w
            tgt_b[n] = (Rb.T @ (P_w - pb), Rb.T @ k_w)
            qq, _ = L.solve(*tgt_b[n]); q[L.qadr] = qq
        dr.qpos[:] = q; mujoco.mj_kinematics(mr, dr)
        for n, L in limbs.items():
            Rb, pb = base[L.base_body]
            Pr = dr.xpos[mr.body(L.end_body).id]
            P_b, k_b = tgt_b[n]
            qq, c = L.solve(P_b - Rb.T @ (Pr - targets[t, L.si]), k_b); q[L.qadr] = qq
            clip_cnt[L.si] += c
        dr.qpos[:] = q; mujoco.mj_kinematics(mr, dr)
        for n, L in limbs.items():                                                 # wrists / ankles
            q[L.dadr] = L.solve_distal(dr.xmat[mr.body(L.mid_body).id].reshape(3, 3), Rg[t, L.he], pos[t])
            err[t, L.si] = np.linalg.norm(dr.xpos[mr.body(L.end_body).id] - targets[t, L.si])
        qpos[t] = q
    ms = 1000 * (time.perf_counter() - t0) / T
    # ground: lowest ankle (5th percentile) at the ankle height of the standing robot
    dz = zero_data(mr)
    ankle0 = dz.xpos[mr.body('left_ankle_pitch_link').id][2] - dz.xpos[mr.body('pelvis').id][2] + stand_height
    qpos[:, 2] += ankle0 - np.percentile(np.minimum(targets[:, 2, 2], targets[:, 3, 2]), 5)
    return qpos, err, clip_cnt / T, ms, s_leg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='output npz (or directory) of tools/smpl2biomech.py')
    p.add_argument('--g1_xml', required=True, help='<mujoco_menagerie>/unitree_g1/g1.xml')
    p.add_argument('--out', required=True, help='output directory (mirrors the input layout)')
    p.add_argument('--support_data', default=os.path.join(ROOT, 'support_data'))
    a = p.parse_args()

    mr, mi = load_models(a.g1_xml)
    limbs = {n: Limb(n, mi, mr) for n in LIMBS}
    stand_height = float(mr.key('stand').qpos[2]) if mr.nkey else 0.79
    joint_names = [mr.joint(j).name for j in range(mr.njnt) if mr.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    if os.path.isfile(a.input):
        files, base = [a.input], os.path.dirname(a.input)
    else:
        files, base = sorted(glob.glob(os.path.join(a.input, '**', '*.npz'), recursive=True)), a.input
    for f in files:
        z = np.load(f, allow_pickle=True)
        if not {'poses', 'trans', 'swivel', 'swivel_vref'} <= set(z.files):
            continue
        g = str(z['gender']) if 'gender' in z.files and str(z['gender']) in ('male', 'female') else 'neutral'
        betas = z['betas'] if 'betas' in z.files else np.zeros(16)
        pos, _, Rg, J = smplh_kinematics(z['poses'][:, :66], z['trans'], betas,
                                         os.path.join(a.support_data, f'body_models/smplh/{g}/model.npz'))
        qpos, err, clip, ms, s_leg = retarget(pos, Rg, z['swivel'], z['swivel_vref'], J, mr, mi, limbs, stand_height)
        fps = next((float(z[k]) for k in ('mocap_framerate', 'mocap_frame_rate') if k in z.files), 30.0)
        dst = os.path.join(a.out, os.path.relpath(f, base))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.savez(dst, qpos=qpos, joint_names=np.array(joint_names), fps=fps, end_error=err, leg_scale=s_leg)
        e = err * 1000
        print(f'{os.path.relpath(f, base)}: {len(qpos)} frames, {ms:.1f} ms/frame | end-point error mean {e.mean():.2f} '
              f'mm, max {e.max():.1f} mm | frames at a joint limit: arms {100 * clip[:2].mean():.0f}%, legs {100 * clip[2:].mean():.0f}%',
              flush=True)


if __name__ == '__main__':
    main()
