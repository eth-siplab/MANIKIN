#!/usr/bin/env python
# --------------------------------------------
# Visualize SMPL motion with its swivel and joint angles, and its retargeting to the Unitree G1.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

"""Visualize one sequence written by tools/smpl2biomech.py: its swivel and joint angle curves, alone or as a video.

    python tools/visualize.py --input out.npz --out angles.png                                           # curves
    python tools/visualize.py --input out.npz --out video.mp4 --blender <blender>                        # video
    python tools/visualize.py --input out.npz --out video.mp4 --blender <blender> \\
        --g1 <retarget_g1 output npz> --g1_xml <mujoco_menagerie>/unitree_g1/g1.xml                     # + G1

The video shows the body (see-through, with an anatomical skeleton inside) and, with --g1, next to it the Unitree G1
of tools/retarget_g1.py, rendered by tools/view_blender.py with Blender (Cycles, on the GPU when one is available); the
curves run below, with a cursor at the current frame. Writing the video needs ffmpeg, and --g1 needs mujoco.
Curves: one panel per angle, right side solid, left side dashed; swivel_arm / swivel_leg are the swivel angles
(degrees) of the arms / legs, the left side negated so that both sides share one sign (the npz stores them mirrored),
left out where the swivel angle is undefined (limb nearly straight, or pointing along its reference axis).
"""
import os, sys, json, glob, argparse, subprocess, tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURVES = ('swivel_arm,swivel_leg,hip_flexion,hip_adduction,hip_rotation,knee_angle,ankle_angle,subtalar_angle,'
          'arm_flex,arm_add,arm_rot,elbow_flex,pro_sup,wrist_flex')
VIDEO_CURVES = 'swivel_arm,arm_flex,elbow_flex,swivel_leg,hip_flexion,knee_angle'           # arms / legs, 3 per row
SIDES = (('r', '-', '#3d8bf2', 'right'), ('l', '--', '#f5a03a', 'left'))
BG, FACE, FG, EDGE = '#16191f', '#1d2129', '#e6e8ec', '#3a3f4a'               # video theme
WIDTH, PANEL_H, HEAD_H, ROW_H = 1920, 1008, 72, 260                          # video layout (px)


def limb_axes(z, support_data):
    """Unit base->end axis (T,4,3) of each limb [L-arm, R-arm, L-leg, R-leg] (forward kinematics of the body joints)."""
    g = str(z['gender']) if 'gender' in z.files and str(z['gender']) in ('male', 'female') else 'neutral'
    m = np.load(os.path.join(support_data, f'body_models/smplh/{g}/model.npz'))
    b = np.zeros(m['shapedirs'].shape[-1])
    bb = np.asarray(z['betas']).reshape(-1)[:len(b)] if 'betas' in z.files else []; b[:len(bb)] = bb
    J = (m['J_regressor'] @ (m['v_template'] + m['shapedirs'] @ b))[:22]
    parents = m['kintree_table'][0][:22].astype(np.int64)
    aa = np.asarray(z['poses'], dtype=np.float64)[:, :66].reshape(-1, 22, 3)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)[..., None]; k = aa / np.maximum(th[..., 0], 1e-12)
    K = np.zeros(aa.shape + (3,))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0], K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 2], k[..., 1], k[..., 2], -k[..., 0], -k[..., 1], k[..., 0]
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    Rg, P = R.copy(), np.zeros(aa.shape)
    for j in range(1, 22):
        Rg[:, j] = Rg[:, parents[j]] @ R[:, j]
        P[:, j] = P[:, parents[j]] + np.einsum('tij,j->ti', Rg[:, parents[j]], J[j] - J[parents[j]])
    n = np.stack([P[:, e] - P[:, a] for a, e in ((16, 20), (17, 21), (1, 7), (2, 8))], 1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def angle_table(z, support_data):
    """{name: (T,) degrees} of the joint angles and the swivel angles, with NaN where a curve wraps around and where
    the swivel angle is undefined (limb within 10 deg of straight, or its axis within 10 deg of the reference)."""
    table = {str(n): z['joint_angles'][:, i].astype(np.float64) for i, n in enumerate(z['joint_angle_names'])}
    sw = np.degrees(np.asarray(z['swivel'], dtype=np.float64))
    ref = np.degrees(np.arccos(np.clip(np.abs(np.sum(limb_axes(z, support_data) * z['swivel_vref'], -1)), 0, 1)))
    undefined = (np.degrees(z['flexion']) < 10) | (ref < 10)
    for k, (name, sign) in enumerate((('swivel_arm_l', -1), ('swivel_arm_r', 1), ('swivel_leg_l', -1), ('swivel_leg_r', 1))):
        table[name] = np.where(undefined[:, k], np.nan, (sign * sw[:, k] + 180) % 360 - 180)
    for v in table.values():
        v[1:][np.abs(np.diff(v)) > 180] = np.nan                                  # no line across a wrap
    return table


def series(table, base):
    """[(values, linestyle, color, label)] of an angle given without the _r / _l suffix."""
    if base in table:
        return [(table[base], '-', SIDES[0][2], None)]
    out = [(table[f'{base}_{s}'], ls, c, lab) for s, ls, c, lab in SIDES if f'{base}_{s}' in table]
    if not out:
        valid = sorted({n[:-2] if n[-2:] in ('_r', '_l') else n for n in table})
        sys.exit(f'unknown angle {base!r}; available: {", ".join(valid)}')
    return out


def draw_curves(axs, t, table, bases, lw=1.3, ms=5):
    """One panel per angle; returns per panel the plotted (values, marker) pairs."""
    markers = []
    for ax, base in zip(axs, bases):
        m = []
        for v, ls, c, lab in series(table, base):
            ax.plot(t, v, ls, color=c, lw=lw, label=lab)
            m.append((v, ax.plot([], [], 'o', color=c, ms=ms)[0]))
        ax.set_title(base); ax.grid(alpha=0.3)
        markers.append(m)
    for ax in axs:                                                            # legend on the first two-sided panel
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=10, loc='upper right', ncol=2, handlelength=1.8); break
    return markers


def plot_png(table, t, bases, out):
    cols = min(4, len(bases)); rows = int(np.ceil(len(bases) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), squeeze=False, sharex=True)
    draw_curves(list(axs.flat), t, table, bases)
    for ax in axs.flat[len(bases):]:
        ax.axis('off')
    for ax in axs[-1]:
        ax.set_xlabel('time (s)')
    for ax in axs[:, 0]:
        ax.set_ylabel('deg')
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def g1_render_data(a, frames_in, path):
    """G1 mesh geoms, their world transforms at the shown frames (MuJoCo forward kinematics of the retargeted qpos),
    the floor under the feet and the root path, for tools/view_blender.py."""
    import mujoco
    r = np.load(a.g1)
    if len(r['qpos']) <= frames_in[-1]:
        sys.exit(f'{a.g1} has {len(r["qpos"])} frames, {a.input} more')
    m = mujoco.MjModel.from_xml_path(a.g1_xml); d = mujoco.MjData(m)
    geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    V, F, cls = [], [], []
    for g in geoms:
        k = m.geom_dataid[g]
        V.append(m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].copy())
        F.append(m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k] + m.mesh_facenum[k]].copy())
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        cls.append(int(rgba[:3].mean() < 0.45))                                  # dark parts
    pose = np.tile(np.eye(4, dtype=np.float32), (len(frames_in), len(geoms), 1, 1))
    for i, t in enumerate(frames_in):
        d.qpos[:] = r['qpos'][t]; mujoco.mj_kinematics(m, d)
        pose[i, :, :3, :3] = d.geom_xmat[geoms].reshape(-1, 3, 3); pose[i, :, :3, 3] = d.geom_xpos[geoms]
    feet = [m.body(b).id for b in ('left_ankle_roll_link', 'right_ankle_roll_link')]
    fg = [i for i, g in enumerate(geoms) if m.geom_bodyid[g] in feet]
    sole = np.array([min((V[g] @ pose[i, g, :3, :3].T + pose[i, g, :3, 3])[:, 2].min() for g in fg) for i in range(len(frames_in))])
    np.savez(path, nv=[len(v) for v in V], nf=[len(f) for f in F], V=np.concatenate(V), F=np.concatenate(F), cls=cls,
             pose=pose, floor=np.percentile(sole, 3), root=r['qpos'][frames_in, :3], leg_scale=float(r['leg_scale']))


def render(a, frame_dir, panel_w):
    """Render the panels with Blender; returns frames.json."""
    cmd = [a.blender, '-b', '--python', os.path.join(ROOT, 'tools', 'view_blender.py'), '--', '--input', a.input,
           '--support_data', a.support_data, '--fps', str(a.fps), '--size', str(panel_w), str(PANEL_H),
           '--samples', str(a.samples), '--render', frame_dir]
    cmd += ['--no_skeleton'] * a.no_skeleton + ['--orbit'] * a.orbit
    if a.g1:
        cmd += ['--g1_render', os.path.join(frame_dir, 'g1.npz')]
    log = os.path.join(frame_dir, 'blender.log')
    with open(log, 'w') as f:
        failed = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode != 0
    if failed:
        with open(log) as f:
            sys.exit('Blender failed:\n' + ''.join(f.readlines()[-20:]))
    with open(os.path.join(frame_dir, 'frames.json')) as f:
        return json.load(f)


def write_video(z, table, bases, a, frame_dir):
    fps_in = next((float(z[k]) for k in ('mocap_framerate', 'mocap_frame_rate') if k in z.files), 30.0)
    step = max(1, int(round(fps_in / a.fps)))
    frames_in = np.arange(0, len(z['poses']), step)
    if a.g1:
        g1_render_data(a, frames_in, os.path.join(frame_dir, 'g1.npz'))
    panel_w = WIDTH // (2 if a.g1 else 1)
    print('rendering with Blender ...', flush=True)
    info = render(a, frame_dir, panel_w)
    titles = {'body': 'SMPL' if a.no_skeleton else 'SMPL + Skeleton', 'robot': 'Unitree G1'}
    passes = info['passes']
    images = {p: sorted(glob.glob(os.path.join(frame_dir, f'{p}_*.png'))) for p in passes}
    n = len(images[passes[0]])
    t = np.arange(len(z['joint_angles'])) / fps_in
    rows = int(np.ceil(len(bases) / 3))
    W, H = panel_w * len(passes), HEAD_H + PANEL_H + ROW_H * rows
    plt.rcParams.update({'text.color': FG, 'axes.labelcolor': FG, 'xtick.color': FG, 'ytick.color': FG,
                         'axes.edgecolor': EDGE, 'axes.facecolor': FACE, 'grid.color': EDGE, 'legend.facecolor': FACE,
                         'legend.edgecolor': EDGE, 'font.size': 12, 'axes.titlesize': 15, 'axes.titleweight': 'bold'})
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    shown = [fig.figimage(plt.imread(images[p][0])[..., :3], panel_w * c, ROW_H * rows) for c, p in enumerate(passes)]
    for c, p in enumerate(passes):
        fig.text((c + 0.5) * panel_w / W, 1 - HEAD_H / 2 / H, titles[p], ha='center', va='center', fontsize=26,
                 fontweight='bold', color=FG)
        if c:
            fig.add_artist(Line2D([c * panel_w / W] * 2, [ROW_H * rows / H, 1 - HEAD_H / H], color=BG, lw=3))
    gs = fig.add_gridspec(rows, 3, left=0.045, right=0.99, top=(ROW_H * rows - 34) / H, bottom=58 / H,
                          wspace=0.25, hspace=0.45)
    axs = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(len(bases))]
    for ax in axs[1:]:
        ax.sharex(axs[0])
    for i, ax in enumerate(axs):
        if i // 3 == rows - 1:
            ax.set_xlabel('time (s)')
        else:
            ax.tick_params(labelbottom=False)
        if i % 3 == 0:
            ax.set_ylabel('deg')
    markers = draw_curves(axs, t, table, bases, lw=1.8, ms=7)
    cursors = [ax.axvline(0, color=FG, lw=1.2) for ax in axs]
    scroll = t[-1] > a.window
    if not scroll:
        axs[0].set_xlim(0, t[-1])
    print(f'writing {n} frames to {a.out} ...', flush=True)
    writer = FFMpegWriter(fps=info['fps'], codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '18'])
    with writer.saving(fig, a.out, dpi=100):
        for k in range(n):
            i = frames_in[k]; ti = t[i]
            for im, p in zip(shown, passes):
                im.set_data(plt.imread(images[p][k])[..., :3])
            for cur in cursors:
                cur.set_xdata([ti, ti])
            for m in markers:
                for v, mk in m:
                    mk.set_data([ti], [v[i]])
            if scroll:                                                        # window around the cursor
                lo = min(max(ti - a.window / 2, 0.0), t[-1] - a.window)
                axs[0].set_xlim(lo, lo + a.window)
            writer.grab_frame(facecolor=BG)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='output npz of tools/smpl2biomech.py')
    p.add_argument('--out', required=True, help='.png for the curves, .mp4 for the video')
    p.add_argument('--angles', default=None, help='comma-separated angle names without the _r / _l suffix '
                   f'(default: {CURVES} for .png, {VIDEO_CURVES} for .mp4)')
    p.add_argument('--blender', default='blender', help='Blender executable (video only)')
    p.add_argument('--g1', default=None, help='output npz of tools/retarget_g1.py for the same sequence (video only)')
    p.add_argument('--g1_xml', default=None, help='<mujoco_menagerie>/unitree_g1/g1.xml, with --g1')
    p.add_argument('--fps', type=float, default=30, help='video frame rate (input frames are subsampled to it)')
    p.add_argument('--window', type=float, default=10.0, help='seconds of curve shown around the cursor in the video')
    p.add_argument('--samples', type=int, default=32, help='Cycles samples per pixel')
    p.add_argument('--no_skeleton', action='store_true', help='opaque body without the skeleton in the video')
    p.add_argument('--orbit', action='store_true', help='turn the cameras with the body heading (paths that loop)')
    p.add_argument('--support_data', default=os.path.join(ROOT, 'support_data'))
    a = p.parse_args()

    z = np.load(a.input, allow_pickle=True)
    table = angle_table(z, a.support_data)
    video = os.path.splitext(a.out)[1].lower() in ('.mp4', '.mov', '.mkv')
    bases = [s.strip() for s in (a.angles or (VIDEO_CURVES if video else CURVES)).split(',') if s.strip()]
    for b in bases:
        series(table, b)
    if video:
        if a.g1 and not a.g1_xml:
            sys.exit('--g1 needs --g1_xml')
        a.input, a.support_data = os.path.abspath(a.input), os.path.abspath(a.support_data)
        with tempfile.TemporaryDirectory(prefix='manikin_vis_') as frame_dir:
            write_video(z, table, bases, a, frame_dir)
    else:
        fps = next((float(z[k]) for k in ('mocap_framerate', 'mocap_frame_rate') if k in z.files), 30.0)
        plot_png(table, np.arange(len(z['joint_angles'])) / fps, bases, a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
