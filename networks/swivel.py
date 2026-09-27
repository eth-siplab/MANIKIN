# --------------------------------------------
# Swivel geometry and 7-DoF limb reconstruction.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

"""Swivel geometry and the Analytic Arm/Leg Solver, shared by MANIKIN-S and MANIKIN-L.

Each arm/leg is a two-bone chain of a base, mid and end joint (pb, pm, pe). Instead of predicting all
joint rotations, we parameterise a limb by its end-effector position plus a single SWIVEL angle: with the
base and end fixed, the mid joint (elbow/knee) is free on a circle about the base->end axis, and the swivel
angle is its position on that circle. This is the anatomical DoF model -- 7 DoF per limb rather than SMPL's
3-per-joint. Bone lengths are lbm
(base->mid) and lme (mid->end); vref is the swivel reference vector.

Pipeline:
  positions_to_swivel  : positions -> swivel angle           (forward map, builds GT swivel targets)
  swivel_to_mid        : swivel angle -> mid-joint position   (inverse, used by the Analytic Arm/Leg Solver)
  analytic_limb_solver : swivel params -> per-limb targets -> solve_limb_angles (the Analytic Arm/Leg Solver)
  solve_limb_angles    : targets -> SMPL LOCAL rotations (knee/elbow: 1-DoF hinge about the flexion axis;
                         limb-root twist from the swivel direction, within the joint range) -> FK.

Limb table (SMPL-H 22-joint): arms use the WRIST (tracker-known); legs use the predicted ANKLE.
  L-arm (16,18,20)  R-arm (17,19,21)  L-leg (1,4,7)  R-leg (2,5,8)   as (base, mid, end)
"""
import math
import numpy as np
import torch
from utils.utils_transform import local2global_pose
from utils import utils_transform



LIMBS = [(16, 18, 20), (17, 19, 21), (1, 4, 7), (2, 5, 8)]     # (base, mid, end) joint per limb
# Internal/external rotation range of the limb root about its own upper bone, relative to SMPL's rest twist (deg).
TWIST_LIMIT = {1: 40.0, 2: 40.0, 16: 90.0, 17: 90.0}
# Knee/elbow flexion axes (local frame), keyed by the mid (hinge) joint.
HINGE_REF = {18: [0.189, -0.966, 0.179], 19: [0.085, 0.970, -0.227],
             4:  [0.985, 0.003, -0.174], 5:  [0.995, -0.029, 0.096]}


def positions_to_swivel(base, mid, end, vertical):
    """Swivel angle of a limb from its three joint positions (the FORWARD map used to build GT targets).

    With the limb root (`base`) and end effector (`end`) fixed and the two bone lengths constant,
    the middle joint (`mid`, the elbow/knee) is free to rotate on a circle about the base->end axis;
    the swivel angle is its position on that circle, measured from a `vertical` reference direction.

    base/mid/end/vertical: (...,1,3). Returns the angle (...,1,1) in radians.
    """
    pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)
    d1 = pdist(base, mid); d2 = pdist(mid, end); d3 = pdist(base, end)  # bone1, bone2, base->end
    # Triangle (base, mid, end): interior angle at `base` from the law of cosines.
    cos_alpha = torch.clamp((d1**2 + d3**2 - d2**2) / (2 * d1 * d3), -1, 1)
    # Orthonormal frame in the circle's plane: n = base->end axis; (u,v) span the plane perp to n,
    # with u the in-plane component of the `vertical` reference (defines swivel = 0).
    e = end - base
    n = e / (torch.norm(e, dim=-1, keepdim=True) + 1e-8)
    u_ = -vertical + torch.sum(vertical * n, dim=-1, keepdim=True) * n
    u = u_ / (torch.norm(u_, dim=-1, keepdim=True) + 1e-8)
    v = torch.cross(u, n, dim=-1)
    c = n * d1 * cos_alpha                                            # foot of perpendicular from mid onto the axis
    cm = mid - c                                                     # radius vector (mid off the axis), lies in (u,v)
    k = cm / torch.norm(cm, dim=-1, keepdim=True)
    return torch.atan2(torch.sum(k * v, dim=-1, keepdim=True), torch.sum(k * u, dim=-1, keepdim=True))


def swivel_to_mid(base, end, vertical, swivel, d1, d2, d3):
    """Inverse of positions_to_swivel: reconstruct the middle joint from the swivel angle.

    Given base, end, the two bone lengths (d1=base->mid, d2=mid->end, d3=base->end) and the swivel
    angle, place `mid` on its circle. Returns mid RELATIVE to `base` (caller adds the base position).
    """
    # Same triangle geometry as the forward map: angle at base -> axial offset (cos) and circle radius (sin).
    cos_tmp = torch.clamp((d1**2 + d2**2 - d3**2) / (2 * d1 * d2), -1 + 1e-6, 1 - 1e-6)
    v3 = math.pi - torch.arccos(cos_tmp)
    cos_alpha = torch.clamp((d1**2 + d3**2 - d2**2) / (2 * d1 * d3), -1, 1)
    sin_alpha = torch.clamp(d2 * torch.sin(math.pi - v3) / d3, -1, 1)
    # Rebuild the same (n,u,v) frame, then walk c along the axis and r around the circle by `swivel`.
    e = end - base
    n = e / (torch.norm(e, dim=-1, keepdim=True) + 1e-8)
    u_ = -vertical + torch.sum(vertical * n, dim=-1, keepdim=True) * n
    u = u_ / (torch.norm(u_, dim=-1, keepdim=True) + 1e-8)
    v = torch.cross(u, n, dim=-1)
    c = n * d1 * cos_alpha                                            # along the axis
    r = d1 * sin_alpha                                               # circle radius
    return c + r * (u * torch.cos(swivel) + v * torch.sin(swivel))


def _rot_about(axis, theta):
    """Rotation matrix about a unit `axis` by `theta` (Rodrigues' formula). axis:(N,3), theta:(N,) -> (N,3,3)."""
    N = theta.shape[0]; a = axis
    K = torch.zeros(N, 3, 3, device=a.device, dtype=a.dtype)         # skew(axis): K @ x == axis x x
    K[:, 0, 1] = -a[:, 2]; K[:, 0, 2] = a[:, 1]; K[:, 1, 0] = a[:, 2]
    K[:, 1, 2] = -a[:, 0]; K[:, 2, 0] = -a[:, 1]; K[:, 2, 1] = a[:, 0]
    I = torch.eye(3, device=a.device, dtype=a.dtype).expand(N, 3, 3)
    return I + torch.sin(theta).view(-1, 1, 1) * K + (1 - torch.cos(theta)).view(-1, 1, 1) * torch.bmm(K, K)


def _rot_align(u, v, eps=1e-8):
    """Minimal rotation R with R @ (u/|u|) == v/|v|. u,v: (T,3) -> (T,3,3)."""
    a = u / (u.norm(dim=-1, keepdim=True) + eps)
    b = v / (v.norm(dim=-1, keepdim=True) + eps)
    c = torch.cross(a, b, dim=-1)                            # rotation axis (unnormalised): |c| = sin, direction a x b
    d = (a * b).sum(-1, keepdim=True).clamp(-1., 1.)         # cos of the angle
    s2 = (c * c).sum(-1, keepdim=True)                       # sin^2
    T = u.shape[0]

    def skew(w):
        K = torch.zeros(T, 3, 3, device=u.device, dtype=u.dtype)
        K[:, 0, 1] = -w[:, 2]; K[:, 0, 2] = w[:, 1]; K[:, 1, 0] = w[:, 2]
        K[:, 1, 2] = -w[:, 0]; K[:, 2, 0] = -w[:, 1]; K[:, 2, 1] = w[:, 0]
        return K
    I = torch.eye(3, device=u.device, dtype=u.dtype).expand(T, 3, 3)
    K = skew(c)
    # Rodrigues via the (unnormalised) axis: I + K + K^2*(1-cos)/sin^2 (the closed form for R that sends a->b).
    R = I + K + torch.bmm(K, K) * ((1. - d) / (s2 + eps)).unsqueeze(-1)
    # Antipodal case (a ~ -b): axis is undefined above; build an explicit 180 deg rotation about any axis perp to a.
    bad = (s2.squeeze(-1) < 1e-12) & (d.squeeze(-1) < 0)
    if bad.any():
        e1 = torch.zeros_like(a); e1[:, 0] = 1.
        p = torch.cross(a, e1, dim=-1)
        e2 = torch.zeros_like(a); e2[:, 1] = 1.
        p = torch.where((p.norm(dim=-1, keepdim=True) < 1e-6), torch.cross(a, e2, dim=-1), p)
        p = p / (p.norm(dim=-1, keepdim=True) + eps)
        Kp = skew(p)
        Rpi = I + 2. * torch.bmm(Kp, Kp)
        R = torch.where(bad.view(-1, 1, 1), Rpi, R)
    return R


def _swing_twist_first(R, a):
    """Swing-twist decomposition R = R_twist @ R_swing, twist PROXIMAL (about unit axis `a`).
    R,a: (T,3,3),(T,3). Twist about `a` preserves points on `a` (the wrist), so end-effector
    position is unchanged. Derivation: decompose M = R^T twist-LAST (M = Sm @ Tm), then
    R = Tm^T @ Sm^T -> R_twist = Tm^T (about a), R_swing = Sm^T (perp a).
    """
    M = R.transpose(-1, -2)
    b = torch.bmm(M, a.unsqueeze(-1)).squeeze(-1)            # M @ a
    Sm = _rot_align(a, b)                                    # swing of M (axis perp a)
    # Gram-Schmidt re-orthonormalise so both outputs remain exactly in SO(3).
    x = Sm[..., 0]; x = x / (x.norm(dim=-1, keepdim=True) + 1e-9)
    y = Sm[..., 1]; y = y - (x * y).sum(-1, keepdim=True) * x; y = y / (y.norm(dim=-1, keepdim=True) + 1e-9)
    Sm = torch.stack([x, y, torch.cross(x, y, dim=-1)], dim=-1)
    Tm = torch.bmm(Sm.transpose(-1, -2), M)                 # twist of M (about a)
    return Tm.transpose(-1, -2), Sm.transpose(-1, -2)       # R_twist (about a), R_swing (perp a)


def twist_about(R, b):
    """Twist angle (rad) of rotations R (T,3,3) about unit axes b (T,3): R = swing @ twist, twist about b applied first."""
    u = lambda x: x / (x.norm(dim=-1, keepdim=True) + 1e-12)
    Tw = torch.bmm(_rot_align(b, torch.bmm(R, b.unsqueeze(-1)).squeeze(-1)).transpose(1, 2), R)
    e = torch.where((b[:, :1].abs() < 0.9), torch.tensor([1., 0, 0], device=b.device, dtype=b.dtype).expand_as(b),
                    torch.tensor([0., 1, 0], device=b.device, dtype=b.dtype).expand_as(b))
    x = u(torch.cross(b, e, dim=-1)); y = torch.bmm(Tw, x.unsqueeze(-1)).squeeze(-1)
    return torch.atan2((torch.cross(x, y, dim=-1) * b).sum(-1), (x * y).sum(-1))


def _limit_twist(R_g, R_pa, n, b, lim, steps=72, refine=12):
    """Rotate R_g about the root->end axis n by the smallest angle psi such that the root's local twist about its
    upper bone b stays within [-lim, lim]. Only violating frames are touched; the end effector is invariant."""
    tw = lambda Rg_: twist_about(torch.bmm(R_pa.transpose(1, 2), Rg_), b)
    bad = tw(R_g).abs() > lim
    if not bool(bad.any()):
        return R_g
    i = bad.nonzero(as_tuple=True)[0]; m = i.numel()
    Rg_i, Rp_i, n_i, b_i = R_g[i], R_pa[i], n[i], b[i]
    psis = torch.linspace(-math.pi, math.pi, steps + 1, device=R_g.device, dtype=R_g.dtype)[:-1]   # (P,)
    P = psis.numel()
    Rrot = _rot_about(n_i.repeat_interleave(P, 0), psis.repeat(m))                                 # (m*P,3,3)
    Rc = torch.bmm(Rrot, Rg_i.repeat_interleave(P, 0))
    tau = twist_about(torch.bmm(Rp_i.repeat_interleave(P, 0).transpose(1, 2), Rc), b_i.repeat_interleave(P, 0)).reshape(m, P)
    viol = (tau.abs() - lim).clamp(min=0)
    cost = viol * 1e3 + psis.abs().unsqueeze(0)                         # feasible first, then the smallest change
    j = cost.argmin(1); psi = psis[j]
    # refine towards psi = 0 by bisection on feasibility (the optimum lies on the range boundary)
    lo, hi = torch.zeros_like(psi), psi.clone()                         # lo: infeasible side (0), hi: feasible
    ok0 = viol[torch.arange(m), j] <= 0
    for _ in range(refine):
        mid_ = 0.5 * (lo + hi)
        Rm = torch.bmm(_rot_about(n_i, mid_), Rg_i)
        feas = twist_about(torch.bmm(Rp_i.transpose(1, 2), Rm), b_i).abs() <= lim
        hi = torch.where(feas & ok0, mid_, hi); lo = torch.where(feas & ok0, lo, mid_)
    out = R_g.clone()
    out[i] = torch.bmm(_rot_about(n_i, hi), Rg_i)
    return out


def _temporal_twist(R_g, n, flex_deg, f0=8.0, f1=25.0):
    """Causal twist continuity for (near-)straight limbs, whose mid joint lies close to the root->end axis so that a
    rotation about that axis hardly moves the joints: keep the root orientation continuous with the previous frame
    (closed-form best rotation about n); full weight below f0 degrees of flexion, pure swivel solution above f1.
    The end effector is invariant. R_g: (T,3,3) time-ordered frames of ONE sequence."""
    Rn = R_g.detach().cpu().double().numpy(); nn = n.detach().cpu().double().numpy()
    w = np.clip((f1 - flex_deg.detach().cpu().double().numpy()) / (f1 - f0), 0, 1)
    out = Rn.copy()
    for t in range(1, len(Rn)):
        if w[t] <= 0:
            continue
        k = nn[t]; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        M = Rn[t] @ out[t - 1].T                      # maximise tr(rot(k, psi) @ M): closest to the previous frame
        psi = w[t] * np.arctan2(np.trace(K @ M), np.trace(M) - k @ M @ k)
        out[t] = (np.eye(3) + np.sin(psi) * K + (1 - np.cos(psi)) * K @ K) @ Rn[t]
    return torch.as_tensor(out, dtype=R_g.dtype, device=R_g.device)


def hinge_side_sign(o1, o2, a):
    """+-1 such that  sign * unit(a_perp x axis)  points to the mid-joint side of the base->end axis, where a is the
    (flexion-positive) hinge axis. Calibrated once on the rest bones o1, o2 at a clearly bent pose
    (world-frame equivalent: sign * unit(h_perp x n) with h the world hinge axis and n the base->end direction)."""
    u = lambda x: x / (x.norm(dim=-1, keepdim=True) + 1e-9)
    vpar = (a * o2).sum() * a
    tb = torch.atan2((o1 * torch.cross(a, o2, dim=-1)).sum(), (o1 * (o2 - vpar)).sum()) + 1.0
    wb = u(o1 + _rot_about(a.unsqueeze(0), tb.reshape(1))[0] @ o2)
    sb = o1 - (o1 * wb).sum() * wb
    hb = torch.cross(u(a - (a * wb).sum() * wb), wb, dim=-1)
    return torch.sign((sb * hb).sum())


def solve_limb_angles(predicted_angle, predicted_position, targets, J_rest, parents, kintree,
                 bm, trans, hinge_ref=None, wrist_ori=None, limits=True, temporal=False, twist_fn=None, fk=True):
    """Analytic Arm/Leg Solver (angle stage): recover each limb's LOCAL joint rotations from its swivel
    target geometry, then FK to positions.

    Under the anatomical DoF model the mid joint (knee/elbow) is a 1-DoF hinge about its flexion axis, and the
    limb-root rotation aligns the limb with the base->end axis and the swivel direction. The limb-root twist is
    kept within its joint range (limits) and, over a time-ordered sequence, continuous for nearly straight limbs
    (temporal). A known wrist orientation (wrist_ori) aligns the hands. twist_fn(R, n, r) -> R, if given, instead
    turns the limb-root rotations R (B,3,3) about their root->end axes n (B,3), r (B,) being the mid joint's distance
    from that axis; the end effector is unchanged. fk=False skips the final forward kinematics and returns
    (None, angles).

    predicted_angle: (B,66) axis-angle (limb joints overwritten here); predicted_position: (B,22,3) FK world.
    targets: list of (ja, jb, jc, mid_world, end_world, toe_world_or_None[, swivel_dir_world[, bend_sign]]),
    ja=root, jb=hinge, jc=end.
    Returns (positions (B,22,3), angles (B,66)).
    """
    eps = 1e-9
    dev = predicted_angle.device
    u = lambda x: x / (x.norm(dim=-1, keepdim=True) + eps)
    if hinge_ref is None:                                            # flexion-direction reference per hinge
        hinge_ref = {k: (torch.tensor(v, device=dev) / torch.tensor(v, device=dev).norm())
                     for k, v in HINGE_REF.items()}
    Jr = J_rest                                                     # rest-pose joint positions (bone geometry)
    _loc = utils_transform.aa2matrot(predicted_angle.reshape(-1, 3)).reshape(predicted_angle.shape[0], -1, 9)
    Rg = local2global_pose(_loc, kintree)                          # global rotations of the current pose
    # Collect overrides then rebuild `ang` functionally; in-place slice writes break autograd in training.
    overrides = {}
    for tgt in targets:
        ja, jb, jc, mid, end, toe = tgt[:6]
        kdir = tgt[6] if len(tgt) > 6 else None                    # swivel direction (world), if known
        bend = tgt[7] if len(tgt) > 7 else None                    # per-frame +1 flexion / -1 hyperextension, optional
        R_pa = Rg[:, parents[ja], :].reshape(-1, 3, 3)             # parent global rot (global->local for ja)
        p_ja = predicted_position[:, ja, :]                        # limb-root world position (kept from FK)
        o1 = (Jr[jb] - Jr[ja]).unsqueeze(0).expand_as(p_ja)       # rest bone root->hinge
        o2 = (Jr[jc] - Jr[jb]).unsqueeze(0).expand_as(p_ja)       # rest bone hinge->end
        # hinge = anatomical flexion axis (positive rotation = flexion)
        a = u(hinge_ref[jb].to(o1.dtype)).unsqueeze(0).expand_as(o1)
        # 1. hinge angle from the root->end distance: |o1 + rot(a,t) o2|^2 = |o1|^2 + |o2|^2 + 2(A + B cos t + C sin t)
        w = end - p_ja; d = w.norm(dim=-1)
        n_ax = w / (d.unsqueeze(-1) + eps)
        vpar = (a * o2).sum(-1, keepdim=True) * a
        A = (o1 * vpar).sum(-1); B = (o1 * (o2 - vpar)).sum(-1); C = (o1 * torch.cross(a, o2, dim=-1)).sum(-1)
        Kc = (d ** 2 - (o1 * o1).sum(-1) - (o2 * o2).sum(-1)) / 2 - A
        sgn = torch.ones_like(d) if bend is None else bend.to(d.dtype)   # flexion unless given
        theta = torch.atan2(C, B) + sgn * torch.arccos(torch.clamp(Kc / (torch.sqrt(B ** 2 + C ** 2) + eps), -1 + 1e-7, 1 - 1e-7))
        R_kn = _rot_about(a, theta)
        w0 = o1 + torch.bmm(R_kn, o2.unsqueeze(-1)).squeeze(-1)   # root->end in the root's rest frame
        e1 = u(w0)
        # 2. which side of the root->end axis the mid joint lies on (rest frame); hs covers the degenerate straight case.
        s0 = o1 - (o1 * e1).sum(-1, keepdim=True) * e1
        a_perp = u(a - (a * e1).sum(-1, keepdim=True) * e1)
        hs = u(torch.cross(a_perp, e1, dim=-1)) * hinge_side_sign(o1[0], o2[0], a[0]) * sgn.unsqueeze(-1)
        deg = (s0.norm(dim=-1, keepdim=True) < 1e-6 * o1.norm(dim=-1, keepdim=True)).to(s0.dtype)
        e2 = u((1 - deg) * u(s0) + deg * hs)
        # 3. target side: the swivel direction k (defined even for a straight limb), else from the mid target.
        k = kdir if kdir is not None else (mid - p_ja)
        k = u(k - (k * n_ax).sum(-1, keepdim=True) * n_ax)
        # 4. root global rotation: (e1, e2) in the rest frame -> (limb axis, swivel direction) in the world.
        R_ja_g = torch.bmm(torch.stack([n_ax, k, torch.cross(n_ax, k, dim=-1)], dim=-1),
                           torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], dim=-1).transpose(1, 2))
        # 5. joint range: the root's internal/external rotation about its upper bone stays within TWIST_LIMIT. A
        #    frame outside the range is rotated about the root->end axis, so the end effector (tracked wrist / ankle)
        #    stays exact and only the elbow/knee moves on its swivel circle. (limits=False when converting
        #    mocap poses.)
        if limits and ja in TWIST_LIMIT:
            R_ja_g = _limit_twist(R_ja_g, R_pa, n_ax, u(o1), math.radians(TWIST_LIMIT[ja]))
        # 6. inference (temporal=True, frames = one time-ordered sequence): twist continuity for (near-)straight limbs.
        if temporal and R_ja_g.shape[0] > 1:
            m_ = torch.bmm(R_ja_g, o1.unsqueeze(-1)).squeeze(-1)                       # root->mid (world)
            flex = torch.rad2deg(torch.arccos(torch.clamp((u(m_) * u(w - m_)).sum(-1), -1, 1)))
            R_ja_g = _temporal_twist(R_ja_g, n_ax, flex)
            if limits and ja in TWIST_LIMIT:
                R_ja_g = _limit_twist(R_ja_g, R_pa, n_ax, u(o1), math.radians(TWIST_LIMIT[ja]))
        elif twist_fn is not None:
            r_mid = ((mid - p_ja) - ((mid - p_ja) * n_ax).sum(-1, keepdim=True) * n_ax).norm(dim=-1)
            R_ja_g = twist_fn(R_ja_g, n_ax, r_mid)
            if limits and ja in TWIST_LIMIT:
                R_ja_g = _limit_twist(R_ja_g, R_pa, n_ax, u(o1), math.radians(TWIST_LIMIT[ja]))
        R_jb_g = torch.bmm(R_ja_g, R_kn)                          # hinge global rot = root ∘ hinge
        mid = p_ja + torch.bmm(R_ja_g, o1.unsqueeze(-1)).squeeze(-1)   # solved mid joint (= swivel target when bent)
        overrides[ja] = utils_transform.matrot2aa(torch.bmm(R_pa.transpose(1, 2), R_ja_g))  # ja LOCAL rot
        overrides[jb] = utils_transform.matrot2aa(R_kn)                # hinge: pure 1-DoF flexion
        if toe is not None:
            # Leg: also orient the foot so the rest toe bone points at the measured toe position.
            jt = jc + 3
            p_jc = mid + torch.bmm(R_jb_g, o2.unsqueeze(-1)).squeeze(-1)
            o3 = (Jr[jt] - Jr[jc]).unsqueeze(0).expand_as(p_jc)
            R_jc_g = torch.bmm(_rot_align(torch.bmm(R_jb_g, o3.unsqueeze(-1)).squeeze(-1), toe - p_jc), R_jb_g)
            overrides[jc] = utils_transform.matrot2aa(torch.bmm(R_jb_g.transpose(1, 2), R_jc_g))
        elif wrist_ori is not None:
            # Arm: match the known wrist orientation. Split the wrist-local rotation (relative to the forearm)
            # into a TWIST about the forearm axis and a SWING perpendicular to it. Anatomically the twist is
            # forearm pronation/supination, so fold it into the elbow; the swing (wrist flexion/abduction)
            # stays at the wrist. The twist is about the forearm axis through the wrist, so wrist position
            # is unchanged.
            side = 0 if jc == 20 else 1
            R_wl = torch.bmm(R_jb_g.transpose(1, 2), wrist_ori[:, side])   # target wrist rot, local to forearm
            R_tw, R_sw = _swing_twist_first(R_wl, u(o2))
            overrides[jb] = utils_transform.matrot2aa(torch.bmm(R_kn, R_tw))  # elbow = flexion ∘ pronation
            overrides[jc] = utils_transform.matrot2aa(R_sw)                    # wrist = flex-ext + abd-add
    # rebuild (B,66) functionally: kept joints from predicted_angle, solved limb joints overridden
    cols = [overrides[j] if j in overrides else predicted_angle[:, j*3:j*3+3] for j in range(22)]
    ang = torch.cat(cols, dim=-1)
    if not fk:
        return None, ang
    body = bm(**{'pose_body': ang[..., 3:66], 'root_orient': ang[..., :3], 'trans': trans})
    return body.Jtr[:, :22, :], ang


def analytic_limb_solver(predicted_angle, predicted_position, wrist_world, ankle_world, toe_world,
                       swivel_ang, J_rest, parents, kintree, bm, trans, hinge_ref=None, wrist_ori=None, temporal=False):
    """Analytic Arm/Leg Solver: from each limb's base joint (shoulder/hip, from Torso FK), its swivel angle,
    and the known end effector, reconstruct the mid joint (elbow/knee) then solve all limb joint angles.
    predicted_angle: (B,66) aa; predicted_position: (B,22,3) Torso-FK world positions (base joints).
    wrist_world/ankle_world/toe_world: (B,2,3) L/R end effectors (wrist known; ankle/toe predicted).
    swivel_ang: (B,4) [L-arm, R-arm, L-leg, R-leg].
    wrist_ori: optional (B,2,3,3) L/R known wrist global rots; when given, hand aligned exactly.
    """
    pd = torch.nn.PairwiseDistance(p=2, keepdim=True)
    _loc = utils_transform.aa2matrot(predicted_angle.reshape(-1, 3)).reshape(predicted_angle.shape[0], -1, 9)
    Rg = local2global_pose(_loc, kintree).reshape(predicted_angle.shape[0], 22, 3, 3)
    targets = []
    for li, (ja, jb, jc) in enumerate(LIMBS):
        # Swivel reference direction = the limb-root's global Y axis (defines swivel angle 0 for this limb).
        vert = Rg[:, ja, :, 1].unsqueeze(1)
        a = predicted_position[:, ja, :]                          # limb-root world position
        if jc in (20, 21):                                       # arm: end effector = tracker-known wrist
            c = wrist_world[:, 0 if jc == 20 else 1, :]
            toe = None
        else:                                                    # leg: end effector = predicted ankle (+ toe)
            c = ankle_world[:, 0 if jc == 7 else 1, :]
            toe = toe_world[:, 0 if jc == 7 else 1, :]
        d1 = pd(J_rest[ja].unsqueeze(0), J_rest[jb].unsqueeze(0)).expand(a.shape[0], 1)   # rest bone lengths
        d2 = pd(J_rest[jb].unsqueeze(0), J_rest[jc].unsqueeze(0)).expand(a.shape[0], 1)
        d3 = pd(a.unsqueeze(1), c.unsqueeze(1)).squeeze(1)       # current base->end distance
        sw = swivel_ang[:, li].unsqueeze(-1)
        # Place the hinge (mid) on its circle from the swivel angle, in the base frame (base passed as 0).
        mid = a + swivel_to_mid((a - a).unsqueeze(1), (c - a).unsqueeze(1), vert,
                             sw.unsqueeze(1), d1.unsqueeze(1), d2.unsqueeze(1), d3.unsqueeze(1)).squeeze(1)
        nn_ = (c - a) / ((c - a).norm(dim=-1, keepdim=True) + 1e-8)                  # same (u, v) frame as swivel_to_mid
        v0 = vert.squeeze(1)
        uu = -v0 + (v0 * nn_).sum(-1, keepdim=True) * nn_; uu = uu / (uu.norm(dim=-1, keepdim=True) + 1e-8)
        vv = torch.cross(uu, nn_, dim=-1)
        kdir = uu * torch.cos(sw) + vv * torch.sin(sw)                              # swivel direction (world)
        targets.append((ja, jb, jc, mid, c, toe, kdir))
    # Turn the (base, mid, end) world targets back into legal SMPL local rotations, then FK.
    return solve_limb_angles(predicted_angle, predicted_position, targets, J_rest, parents, kintree,
                        bm, trans, hinge_ref, wrist_ori=wrist_ori, temporal=temporal)
