# --------------------------------------------
# View SMPL-H motion in Blender: the body with an anatomical skeleton, and the retargeted Unitree G1.
# MANIKIN: Biomechanically Accurate Neural Inverse Kinematics for Human Motion Estimation (ECCV 2024)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
# https://github.com/eth-siplab/MANIKIN
# Contact: Jiaxi Jiang <jiaxi.jiang@inf.ethz.ch>
# Licensed under the MIT License (see LICENSE).
# --------------------------------------------

"""View SMPL-H motion (e.g. the output of tools/smpl2biomech.py, or any AMASS npz) in Blender, or render it to images.

    blender --python tools/view_blender.py -- --input out.npz [--save scene.blend]
    blender -b --python tools/view_blender.py -- --input out.npz --render <frame dir>          # without a GUI

The scene has two collections:
  body   the body as a see-through mesh with an anatomical skeleton inside (--no_skeleton: opaque body only)
  robot  the Unitree G1 of tools/retarget_g1.py (--g1_render, a file written by tools/visualize.py), on its own floor;
         hidden in the viewport, as it moves along its own scaled path
The body is computed here with numpy (SMPL-H linear blend skinning with the models in support_data) and played from a
PC2 point cache, so the timeline can be scrubbed and the scene saved (the .pc2 files are written next to the .blend).
The skeleton (tools/assets/skeleton.npz, see tools/assets/README.md) is fitted to the male SMPL-H template; each bone
moves rigidly with its joint (finger bones with the SMPL-H finger joints) and is scaled with the subject's segment
lengths, the patellae glide on the femur and the clavicles span sternum and acromion. The cameras look at the body
(and at the robot) 30 deg from the front and follow it; --orbit turns them with the body's heading.
--render writes every frame of each collection as <collection>_<frame>.png with Cycles (on the GPU when one is
available), plus frames.json with the frame rate and the input frames shown; tools/visualize.py composes them.
"""
import os, sys, json, argparse, struct, tempfile
import numpy as np
import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BG = (0.14, 0.155, 0.19)
VFOV = 2 * np.arctan(np.tan(np.radians(19)) * 1008 / 720 / 1.05)                # vertical field of view


# ============================================================ rotations
def rodrigues(aa):
    """(N,3) axis-angle -> (N,3,3)."""
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-12)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 2], k[:, 1], k[:, 2], -k[:, 0], -k[:, 1], k[:, 0]
    th = th[:, :, None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def log_rot(R):
    """(N,3,3) rotations -> (N,3) axis-angle."""
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1); th = np.arccos(c)
    w = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    s = np.sin(th)
    aa = w * np.where(s > 1e-6, th / (2 * np.maximum(s, 1e-12)), 0.5)[:, None]
    near_pi = th > np.pi - 1e-3                                                  # axis from the symmetric part
    if near_pi.any():
        B = (R[near_pi] + np.eye(3)) / 2
        ax = B[np.arange(len(B)), :, np.argmax(np.diagonal(B, axis1=1, axis2=2), 1)]
        aa[near_pi] = ax / np.linalg.norm(ax, axis=1, keepdims=True) * th[near_pi, None]
    return aa


def quat(R):
    """(N,3,3) rotations -> (N,4) quaternions (w,x,y,z), sign kept continuous along N."""
    aa = log_rot(R); th = np.linalg.norm(aa, axis=1, keepdims=True)
    q = np.concatenate([np.cos(th / 2), np.sin(th / 2) * aa / np.maximum(th, 1e-12)], 1)
    for i in range(1, len(q)):
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    return q


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def look_at(eye, target):
    """(T,3), (T,3) -> rotations (T,3,3) of a camera / light at eye looking at target (Blender: -z forward, y up)."""
    f = unit(target - eye); r = unit(np.cross(f, [0, 0, 1])); u = np.cross(r, f)
    return np.stack([r, u, -f], 2)


def smooth(x, k):
    """Moving average over k frames along axis 0."""
    pad = np.pad(x, ((k // 2, k - 1 - k // 2),) + ((0, 0),) * (x.ndim - 1), mode='edge')
    return np.stack([np.convolve(pad[:, i], np.ones(k) / k, mode='valid') for i in range(x.shape[1])], 1)


# ============================================================ human
def load_motion(path, fps_default):
    """poses (T,156), trans (T,3), betas (16,), gender and frame rate of the input."""
    z = np.load(path, allow_pickle=True)
    poses, trans = np.asarray(z['poses'], dtype=np.float64), np.asarray(z['trans'], dtype=np.float64)
    if poses.shape[1] < 156:                                                   # body only: hands at rest
        poses = np.concatenate([poses[:, :66], np.zeros((len(poses), 90))], 1)
    betas = np.zeros(16)
    b = np.asarray(z['betas']).reshape(-1)[:16] if 'betas' in z.files else np.zeros(0)
    betas[:len(b)] = b
    gender = str(z['gender']) if 'gender' in z.files else 'neutral'
    fps = next((float(z[k]) for k in ('mocap_framerate', 'mocap_frame_rate') if k in z.files), fps_default)
    return poses[:, :156], trans, betas, gender, fps


def smplh_model(model_path, betas):
    m = np.load(model_path, allow_pickle=True)
    parents = m['kintree_table'][0].astype(np.int64); parents[0] = -1
    v_shaped = m['v_template'] + m['shapedirs'][:, :, :16] @ betas
    return m, v_shaped, m['J_regressor'] @ v_shaped, parents


def forward_kinematics(poses, trans, J, parents):
    """Global rotations (T,52,3,3) and world positions (T,52,3) of the SMPL-H joints."""
    T = len(poses)
    R = rodrigues(poses.reshape(-1, 3)).reshape(T, 52, 3, 3)
    Rg, Jw = np.empty_like(R), np.empty((T, 52, 3))
    for j in range(52):
        if j == 0:
            Rg[:, 0], Jw[:, 0] = R[:, 0], J[0] + trans
        else:
            Rg[:, j] = Rg[:, parents[j]] @ R[:, j]
            Jw[:, j] = Jw[:, parents[j]] + np.einsum('tij,j->ti', Rg[:, parents[j]], J[j] - J[parents[j]])
    return Rg, Jw


def smplh_vertices(m, v_shaped, J, parents, poses, Rg, Jw, chunk=256):
    """SMPL-H linear blend skinning (no DMPLs): vertices (T,6890,3) float32."""
    posedirs = m['posedirs'].reshape(-1, m['posedirs'].shape[-1])             # (6890*3, 459)
    out = np.empty((len(poses), len(v_shaped), 3), dtype=np.float32)
    for s in range(0, len(poses), chunk):
        n = len(poses[s:s + chunk])
        R = rodrigues(poses[s:s + chunk].reshape(-1, 3)).reshape(n, 52, 3, 3)
        v = v_shaped[None] + (posedirs @ (R[:, 1:] - np.eye(3)).reshape(n, -1).T).T.reshape(n, -1, 3)
        A = np.concatenate([Rg[s:s + n], (Jw[s:s + n] - np.einsum('tjab,jb->tja', Rg[s:s + n], J))[..., None]], 3)
        A = np.einsum('vj,njab->nvab', m['weights'], A)                         # (n,6890,3,4)
        out[s:s + n] = np.einsum('nvab,nvb->nva', A[..., :3], v) + A[..., 3]
    return out


def write_pc2(path, verts):
    """PC2 point cache: header, then float32 xyz for every point of every frame."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<12siiffi', b'POINTCACHE2\0', 1, verts.shape[1], 0.0, 1.0, verts.shape[0]))
        f.write(np.ascontiguousarray(verts, dtype='<f4').tobytes())


# ============================================================ skeleton
def load_skeleton(path):
    z = np.load(path)
    V, F = z['V'].astype(np.float64), z['F'].astype(np.int64)
    ov, of = np.r_[0, np.cumsum(z['nv'])], np.r_[0, np.cumsum(z['nf'])]
    bones = [dict(name=str(n), parent=int(p), parent2=int(q), weight2=float(w), centroid=c.astype(np.float64),
                  V=V[ov[i]:ov[i + 1]], F=F[of[i]:of[i + 1]])
             for i, (n, p, q, w, c) in enumerate(zip(z['names'], z['parent'], z['parent2'], z['weight2'], z['centroid']))]
    return bones, {k: z[k].astype(np.float64) for k in ('J_rest', 'SC', 'AC', 'CLF')}


def skeleton_poses(bones, S, Rg, Jw, J_sub, parents):
    """Per-bone-group rotation (T,3,3), position (T,3) and scale (T,3). Joints 0-51: SMPL-H joints (body and fingers);
    52/53: left / right patella; 54/55: left / right clavicle. Bones are scaled by the subject / template length of
    their segment."""
    T, J_tpl, nj = len(Rg), S['J_rest'], len(parents)
    ratio = np.ones(nj)
    for j in range(nj):                                                         # parents come before children
        ch = [c for c in range(nj) if parents[c] == j]
        ratio[j] = np.mean([np.linalg.norm(J_sub[c] - J_sub[j]) / np.linalg.norm(J_tpl[c] - J_tpl[j]) for c in ch]) \
            if ch else ratio[parents[j]]
    R56, P56, S56 = np.zeros((T, nj + 4, 3, 3)), np.zeros((T, nj + 4, 3)), np.ones((T, nj + 4, 3))
    R56[:, :nj], P56[:, :nj], S56[:, :nj] = Rg, Jw, ratio[None, :, None]
    byname = {b['name']: b for b in bones}
    for k, (s, hip, knee) in enumerate((('l', 1, 4), ('r', 2, 5))):            # patella: half the knee rotation,
        rel = log_rot(np.einsum('tji,tjk->tik', Rg[:, hip], Rg[:, knee]))      # kept 2 mm off the femur surface
        R = Rg[:, hip] @ rodrigues(rel / 2)
        pa = byname[f'{s}_patella']['V'][::2] * ratio[knee]
        fe = byname[f'{s}_femur']['V'][::5] * ratio[hip]
        pw = np.einsum('tij,nj->tni', R, pa) + Jw[:, knee, None]
        fw = np.einsum('tij,nj->tni', Rg[:, hip], fe) + Jw[:, hip, None]
        P = Jw[:, knee].copy()
        for t in range(T):
            d = np.linalg.norm(pw[t, :, None] - fw[t, None], axis=-1)
            i, j = np.unravel_index(d.argmin(), d.shape)
            P[t] += (d[i, j] - 0.002) * (fw[t, j] - pw[t, i]) / max(d[i, j], 1e-9)
        R56[:, nj + k], P56[:, nj + k], S56[:, nj + k] = R, P, ratio[knee]
    for k, col in enumerate((13, 14)):                                          # clavicle: sternoclavicular point on
        sc = np.einsum('tij,j->ti', Rg[:, 9], (S['SC'][k] - J_tpl[9]) * ratio[9]) + Jw[:, 9]    # the thorax, acromion
        ac = np.einsum('tij,j->ti', Rg[:, col], (S['AC'][k] - J_tpl[col]) * ratio[col]) + Jw[:, col]   # on the scapula
        x = ac - sc; L = np.linalg.norm(x, axis=1, keepdims=True); x /= L
        y = Rg[:, col] @ S['CLF'][k][:, 1]; y -= np.sum(y * x, 1, keepdims=True) * x; y /= np.linalg.norm(y, axis=1, keepdims=True)
        R56[:, nj + 2 + k] = np.stack([x, y, np.cross(x, y)], 2); P56[:, nj + 2 + k] = sc
        S56[:, nj + 2 + k] = np.concatenate([L / np.linalg.norm(S['AC'][k] - S['SC'][k]), np.full((T, 2), ratio[col])], 1)
    groups = {}                                                                 # bones sharing one transform
    for b in bones:
        groups.setdefault((b['parent'], b['parent2'], b['name'] if b['parent2'] >= 0 else ''), []).append(b)
    out = []
    for (p, q, _), bs in groups.items():
        if q < 0:
            out.append((bs, R56[:, p], P56[:, p], S56[:, p]))
            continue
        b, w = bs[0], bs[0]['weight2']                                         # between two joints (costal
        R = Rg[:, p] @ rodrigues(w * log_rot(np.einsum('tji,tjk->tik', Rg[:, p], Rg[:, q])))   # cartilages, C6/C7)
        ca = np.einsum('tij,j->ti', Rg[:, p], (b['centroid'] - J_tpl[p]) * ratio[p]) + Jw[:, p]
        cb = np.einsum('tij,j->ti', Rg[:, q], (b['centroid'] - J_tpl[q]) * ratio[q]) + Jw[:, q]
        out.append((bs, R, (1 - w) * ca + w * cb, np.full((T, 3), (1 - w) * ratio[p] + w * ratio[q])))
    return out


# ============================================================ scene objects
def animate(obj, loc, rot, scale=None):
    """Keyframe the object transform at every frame."""
    obj.rotation_mode = 'QUATERNION'
    act = bpy.data.actions.new(obj.name)
    obj.animation_data_create().action = act
    frames = np.arange(1, len(loc) + 1, dtype=np.float64)
    channels = [('location', loc), ('rotation_quaternion', quat(rot))] + ([('scale', scale)] if scale is not None else [])
    for path, arr in channels:
        for i in range(arr.shape[1]):
            fc = act.fcurves.new(path, index=i)
            fc.keyframe_points.add(len(arr))
            fc.keyframe_points.foreach_set('co', np.stack([frames, arr[:, i]], 1).ravel())
            fc.update()


def collection(name):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def move_to(obj, col):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    col.objects.link(obj)


def material(name, rgb, rough, metal=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (*rgb, 1.0)
    bsdf.inputs['Roughness'].default_value, bsdf.inputs['Metallic'].default_value = rough, metal
    return mat


def see_through(mat):
    """Transparent where the surface faces the camera, nearly opaque at the silhouette."""
    nt = mat.node_tree
    lw, mr = nt.nodes.new('ShaderNodeLayerWeight'), nt.nodes.new('ShaderNodeMapRange')
    lw.inputs['Blend'].default_value = 0.5
    mr.inputs['To Min'].default_value, mr.inputs['To Max'].default_value = 0.10, 0.85
    nt.links.new(lw.outputs['Facing'], mr.inputs['Value'])
    nt.links.new(mr.outputs['Result'], nt.nodes['Principled BSDF'].inputs['Alpha'])


def mesh_object(name, V, F, mat, col, smooth_shading=True):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(np.asarray(V).tolist(), [], np.asarray(F).tolist())
    if smooth_shading:
        mesh.shade_smooth()
    obj = bpy.data.objects.new(name, mesh)
    col.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def cached_mesh(name, verts, faces, pc2, mat, col):
    """Mesh whose vertices (T,N,3) are played from a PC2 point cache."""
    write_pc2(pc2, verts)
    obj = mesh_object(name, verts[0], faces, mat, col)
    mod = obj.modifiers.new('motion', 'MESH_CACHE')
    mod.cache_format, mod.filepath = 'PC2', pc2
    mod.forward_axis, mod.up_axis = 'POS_Y', 'POS_Z'                          # cache is in the world frame (z up)
    mod.frame_start = 1
    return obj


def add_skeleton(groups, mat, col):
    for k, (bs, R, P, s) in enumerate(groups):
        V = np.concatenate([b['V'] for b in bs])
        F = np.concatenate([b['F'] + o for b, o in zip(bs, np.r_[0, np.cumsum([len(b['V']) for b in bs])[:-1]])])
        animate(mesh_object(f'bones_{k:02d}', V, F, mat, col), P, R, s)


def add_robot(g1, col):
    """G1 mesh geoms, posed per frame by the transforms tools/visualize.py computed with MuJoCo."""
    metal, black = material('g1_metal', (0.74, 0.75, 0.78), 0.38, 0.45), material('g1_black', (0.035, 0.037, 0.045), 0.5)
    ov, of = np.r_[0, np.cumsum(g1['nv'])], np.r_[0, np.cumsum(g1['nf'])]
    for gi in range(len(g1['nv'])):
        obj = mesh_object(f'g1_{gi:02d}', g1['V'][ov[gi]:ov[gi + 1]], g1['F'][of[gi]:of[gi + 1]],
                          black if g1['cls'][gi] else metal, col, smooth_shading=False)
        animate(obj, g1['pose'][:, gi, :3, 3], g1['pose'][:, gi, :3, :3])


def add_floor(center, size, z, square, col):
    """Checkerboard floor at height z."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=(center[0], center[1], z))
    floor = bpy.context.active_object
    move_to(floor, col)
    mat = material(f'{col.name}_checker', (0.33, 0.35, 0.40), 0.9)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    coord, checker = nodes.new('ShaderNodeTexCoord'), nodes.new('ShaderNodeTexChecker')
    checker.inputs['Scale'].default_value = 1.0 / square
    checker.inputs['Color1'].default_value = (0.42, 0.45, 0.50, 1.0)
    checker.inputs['Color2'].default_value = (0.24, 0.26, 0.31, 1.0)
    links.new(coord.outputs['Object'], checker.inputs['Vector'])
    links.new(checker.outputs['Color'], nodes['Principled BSDF'].inputs['Base Color'])
    floor.data.materials.append(mat)


def add_camera(name, eye, target, scale, col):
    """Camera keyed at eye -> target, with an area light above its right shoulder (sized with `scale`)."""
    cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    cam.data.sensor_fit, cam.data.angle = 'VERTICAL', VFOV
    bpy.context.scene.collection.objects.link(cam)
    R = look_at(eye, target)
    animate(cam, eye, R)
    tgt = eye - R[:, :, 2] * 3.3 * scale                                        # about where the camera looks
    lo = tgt + scale * (2.2 * R[:, :, 0] + np.array([0, 0, 3.2]) + 1.2 * R[:, :, 2])
    light = bpy.data.objects.new(name + '_light', bpy.data.lights.new(name + '_light', 'AREA'))
    light.data.energy, light.data.size = 700.0, 5.0
    col.objects.link(light)
    animate(light, lo, look_at(lo, tgt))
    return cam


def camera_path(Rg0, root, floor, fps, orbit):
    """Eye and target (T,3) 30 deg to the left of the body's front, following the smoothed root (T,3); the look-at
    height follows the pelvis (0.35-0.95 m above the floor) and the camera moves in as it drops."""
    fa = (Rg0[:, :, 2] + Rg0[:, :, 1])[:, :2]                                   # pelvis forward + up: forward when crawling
    if orbit:                                                                   # heading smoothed over 1 s
        ang = np.unwrap(np.arctan2(fa[:, 1], fa[:, 0]))
        g = np.exp(-0.5 * (np.arange(-3 * fps, 3 * fps + 1) / fps) ** 2); g /= g.sum()
        ang = np.convolve(np.pad(ang, len(g) // 2, mode='edge'), g, 'valid') + np.arctan2(0.55, 1.0)
        view = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1)
    else:
        f = np.r_[unit(fa.mean(0)), 0.0]
        view = np.broadcast_to(unit(f + 0.55 * np.cross([0, 0, 1], f)), (len(root), 3))
    c = smooth(root, max(1, int(round(fps))) | 1)
    hz = np.clip(c[:, 2] - floor, 0.35, 0.95)
    dist = 3.3 * (0.55 + 0.45 * hz / 0.95)
    target = np.c_[c[:, :2], floor + hz]
    eye = target + np.c_[np.zeros((len(c), 2)), np.full(len(c), 0.45)] + dist[:, None] * view
    return eye, target, hz, dist, view


def setup_cycles(scene, samples):
    """Cycles on the first available GPU backend, else on the CPU."""
    scene.render.engine = 'CYCLES'
    scene.cycles.samples, scene.cycles.use_denoising = samples, True
    scene.render.use_persistent_data = True
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for backend in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = 'GPU'
            if backend == 'OPTIX':
                scene.cycles.denoiser = 'OPTIX'
            return backend
    scene.cycles.device = 'CPU'
    return 'CPU'


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    p = argparse.ArgumentParser(prog='blender --python tools/view_blender.py --')
    p.add_argument('--input', required=True, help='npz with poses / trans (/ betas / gender / mocap_framerate)')
    p.add_argument('--g1_render', default=None, help='G1 geometry and poses written by tools/visualize.py')
    p.add_argument('--support_data', default=os.path.join(ROOT, 'support_data'))
    p.add_argument('--skeleton', default=os.path.join(ROOT, 'tools', 'assets', 'skeleton.npz'))
    p.add_argument('--no_skeleton', action='store_true', help='opaque body without the skeleton')
    p.add_argument('--orbit', action='store_true', help='turn the cameras with the body heading')
    p.add_argument('--fps', type=float, default=30, help='playback frame rate (frames are subsampled to it)')
    p.add_argument('--fps_in', type=float, default=30, help='frame rate of npz files without mocap_framerate')
    p.add_argument('--size', type=int, nargs=2, default=[640, 1008], help='image width and height')
    p.add_argument('--save', default=None, help='save the scene to this .blend')
    p.add_argument('--render', default=None, help='render every frame of each collection into this directory')
    p.add_argument('--samples', type=int, default=32, help='Cycles samples per pixel for --render')
    a = p.parse_args(argv)

    for o in list(bpy.data.objects):                                          # start from an empty scene
        bpy.data.objects.remove(o)
    if a.save:
        cache_dir = os.path.dirname(os.path.abspath(a.save))
    elif a.render:
        cache_dir = os.path.abspath(a.render)
    else:
        cache_dir = tempfile.mkdtemp(prefix='smplh_view_')
    os.makedirs(cache_dir, exist_ok=True)
    poses, trans, betas, gender, fps_in = load_motion(a.input, a.fps_in)
    step = max(1, int(round(fps_in / a.fps))); fps = fps_in / step
    poses, trans = poses[::step], trans[::step]                                # input frames shown
    g = gender if gender in ('male', 'female', 'neutral') else 'neutral'
    m, v_shaped, J, parents = smplh_model(os.path.join(a.support_data, f'body_models/smplh/{g}/model.npz'), betas)
    Rg, Jw = forward_kinematics(poses, trans, J, parents)
    verts = smplh_vertices(m, v_shaped, J, parents, poses, Rg, Jw)
    T = len(poses)
    print(f'{T} frames at {fps:.1f} fps')

    col = {n: collection(n) for n in ('body', 'floor', 'robot', 'robot_floor')}
    skin = material('skin', (0.80, 0.76, 0.72), 0.75)
    cached_mesh('body', verts, m['f'], os.path.join(cache_dir, 'body.pc2'), skin, col['body'])
    if not a.no_skeleton:
        see_through(skin)
        bones, S = load_skeleton(a.skeleton)
        add_skeleton(skeleton_poses(bones, S, Rg, Jw, J, parents), material('bone', (0.88, 0.84, 0.72), 0.5), col['body'])

    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, T
    scene.render.fps = int(round(fps))
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = a.size[0], a.size[1], 100
    floor = float(np.percentile(verts[:, :, 2].min(1), 3))                    # floor under the lowest frames
    extent = float(np.ptp(Jw[:, 0, :2], 0).max())
    add_floor(Jw[:, 0, :2].mean(0), extent + 60.0, floor, 0.5, col['floor'])
    eye, target, hz, dist, view = camera_path(Rg[:, 0], Jw[:, 0], floor, fps, a.orbit)
    cam_h = add_camera('camera', eye, target, 1.0, col['floor'])
    passes = ['body']
    cam_r = None
    if a.g1_render:
        g1 = dict(np.load(a.g1_render))
        assert len(g1['pose']) == T, 'G1 poses and the input do not have the same frames'
        s, floor_r, root_r = float(g1['leg_scale']), float(g1['floor']), g1['root']
        add_robot(g1, col['robot'])
        add_floor(root_r[:, :2].mean(0), float(np.ptp(root_r[:, :2], 0).max()) + 60.0, floor_r, 0.5 * s, col['robot_floor'])
        c = smooth(root_r, max(1, int(round(fps))) | 1)
        target_r = np.c_[c[:, :2], floor_r + hz * s]
        eye_r = target_r + np.c_[np.zeros((T, 2)), np.full(T, 0.45 * s)] + (s * dist)[:, None] * view
        cam_r = add_camera('camera_robot', eye_r, target_r, s, col['robot_floor'])
        for c_ in (col['robot'], col['robot_floor']):
            c_.hide_viewport = True
        passes.append('robot')
    scene.camera = cam_h
    if scene.world is None:
        scene.world = bpy.data.worlds.new('world')
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value, bg.inputs['Strength'].default_value = (*BG, 1.0), 0.55
    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save))
        print('saved', os.path.abspath(a.save))
    if a.render:
        print('rendering with Cycles on', setup_cycles(scene, a.samples))
        scene.render.image_settings.file_format = 'PNG'
        with open(os.path.join(cache_dir, 'frames.json'), 'w') as f:
            json.dump({'n_frames': T, 'fps': fps, 'fps_in': fps_in, 'step': step, 'passes': passes}, f)
        visible = {'body': ('body', 'floor'), 'robot': ('robot', 'robot_floor')}
        for ps in passes:
            for n, c_ in col.items():
                c_.hide_render = n not in visible[ps]
            scene.camera = cam_r if ps == 'robot' else cam_h
            scene.render.filepath = os.path.join(cache_dir, f'{ps}_')
            bpy.ops.render.render(animation=True)


if __name__ == '__main__':
    main()
