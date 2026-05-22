"""CPU mesh-boolean IoU.

Computes IoU(A, B) = volume(A ∩ B) / volume(A ∪ B) using manifold3d directly.
Replaces the prior GPU ray-cast sampler.

Mesh manifoldification is done via pymeshlab's `generate_alpha_wrap`, which
emits a guaranteed-watertight surface around any input (non-manifold, multiple
disconnected solids, micro-gaps, dense tessellation, etc.). This is much more
robust than the previous trimesh-based heal + manifold3d cleanup chain.
"""
import os
import numpy as np
import trimesh
import manifold3d


_OP_INTERSECT = manifold3d.OpType.Intersect
_OP_ADD = manifold3d.OpType.Add

# Alpha-wrap parameters (percentages of the mesh bounding-box diagonal).
# alpha: largest feature size that gets smoothed over (smaller = tighter).
# offset: how far outside the input surface the wrap can sit.
ALPHA_WRAP_ALPHA_PCT = 1.0
ALPHA_WRAP_OFFSET_PCT = 0.1

# Decimation target — alpha-wrap on multi-hundred-thousand-face meshes
# dominates per-STL runtime, so we cheaply decimate first.
_DECIMATE_FACE_THRESHOLD = 50_000


def normalize_mesh(mesh, box_min: float = -1.0, box_max: float = 1.0):
    """Center and uniformly scale `mesh` so its largest extent fits [box_min, box_max]."""
    bb_min = mesh.bounds[0]
    bb_max = mesh.bounds[1]
    center = 0.5 * (bb_min + bb_max)
    max_dim = float((bb_max - bb_min).max())
    if max_dim > 0:
        scale = (box_max - box_min) / max_dim
        mesh.apply_translation(-center)
        mesh.apply_scale(scale)
    return mesh


def _alpha_wrap(mesh, target_faces: int = _DECIMATE_FACE_THRESHOLD):
    """Return a guaranteed-watertight trimesh produced by pymeshlab's
    alpha-wrap filter. Decimates the input to `target_faces` first when it
    has more (cheap pass that drastically speeds up alpha-wrap on dense
    CAD STLs without changing the wrap's geometric output meaningfully).
    Falls back to the original mesh on any failure."""
    try:
        import pymeshlab
    except ImportError:
        return mesh
    try:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        ))
        if len(mesh.faces) > target_faces:
            try:
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=target_faces,
                    preservenormal=True,
                    preserveboundary=True,
                )
            except Exception:
                pass
        ms.generate_alpha_wrap(
            alpha=pymeshlab.PercentageValue(ALPHA_WRAP_ALPHA_PCT),
            offset=pymeshlab.PercentageValue(ALPHA_WRAP_OFFSET_PCT),
        )
        out = ms.current_mesh()
        return trimesh.Trimesh(vertices=out.vertex_matrix(),
                                faces=out.face_matrix(),
                                process=True)
    except Exception as e:
        # Soft failure: caller can still try to build a manifold from the original.
        return mesh


def _to_manifold(mesh):
    """Build a manifold3d.Manifold from a trimesh mesh. Returns None on
    failure or empty result (volume == 0)."""
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    try:
        mm = manifold3d.Mesh(vert_properties=verts, tri_verts=faces)
        m = manifold3d.Manifold(mm)
    except Exception:
        return None
    if m.volume() == 0.0:
        return None
    return m


def _manifold_boolean(a, b, op):
    """Run a manifold boolean and return the result Manifold, or None on failure."""
    try:
        return manifold3d.Manifold.batch_boolean([a, b], op)
    except AttributeError:
        # Older manifold3d API.
        if op is _OP_INTERSECT:
            return a & b
        if op is _OP_ADD:
            return a + b
    except Exception:
        return None
    return None


_MANIFOLD_CACHE: dict = {}
_CACHE_MAX = 8


def _cached_manifold(path: str, normalize: bool):
    """Manifoldify the mesh at `path`, caching by `(path, mtime, normalize)`.

    Tries the direct conversion first (fast, no geometric distortion).
    Only falls back to alpha-wrap when the mesh is non-manifold / non-volume
    and manifold3d returns an empty manifold. Greedy search calls IoU
    hundreds of times against the same ground-truth file, so caching the
    result avoids re-running the (slow) alpha-wrap per call.
    """
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    key = (path, mtime, normalize)
    if key in _MANIFOLD_CACHE:
        return _MANIFOLD_CACHE[key]
    try:
        mesh = trimesh.load(path, force='mesh')
    except Exception:
        _MANIFOLD_CACHE[key] = None
        return None
    if normalize:
        mesh = normalize_mesh(mesh)

    # Multi-body meshes (Compound STLs from CadQuery `.add()`) pass
    # `is_watertight` and `is_volume` but contain disjoint shells whose
    # pairwise overlapping regions break manifold3d's volume math
    # (gives vol(A∩B) > vol(A∪B)). Alpha-wrap merges them.
    try:
        multi_body = mesh.body_count > 1
    except Exception:
        multi_body = False

    # Try the direct manifold3d conversion first — even on non-watertight
    # inputs it often produces the right volume when vertex-merging closes
    # the topology. Compare against trimesh's signed-integral volume to
    # detect cases where it silently went wrong (the alpha-wrap pathway
    # used to produce ~50% under-volumes on slabs with holes).
    m = None
    if not multi_body:
        m_direct = _to_manifold(mesh)
        if m_direct is not None:
            try:
                raw_vol = abs(mesh.volume)
                if raw_vol > 0:
                    ratio = m_direct.volume() / raw_vol
                    if 0.9 <= ratio <= 1.1:
                        m = m_direct
            except Exception:
                pass

    # Fall back to alpha-wrap if direct didn't yield a volume-consistent
    # manifold, or for multi-body meshes.
    if m is None:
        m = _to_manifold(_alpha_wrap(mesh))
        if m is None:
            m = _to_manifold(mesh)

    if len(_MANIFOLD_CACHE) >= _CACHE_MAX:
        _MANIFOLD_CACHE.pop(next(iter(_MANIFOLD_CACHE)))
    _MANIFOLD_CACHE[key] = m
    return m


def mesh_iou_cpu_boolean(mesh_path_a: str,
                         mesh_path_b: str,
                         normalize: bool = False,
                         verbose: bool = False) -> float:
    """IoU between two meshes via CPU booleans.

    Each input mesh is run through pymeshlab's alpha-wrap so it becomes a
    guaranteed-watertight manifold before manifold3d's boolean engine touches
    it. An LRU-ish cache keyed on (path, mtime, normalize) so the same
    ground-truth file isn't re-wrapped for every candidate during greedy search.
    `normalize=True` rescales mesh A to a [-1, 1] cube. Returns 0.0 on any
    failure or empty union.
    """
    ma = _cached_manifold(mesh_path_a, normalize)
    mb = _cached_manifold(mesh_path_b, normalize=False)
    if ma is None or mb is None:
        if verbose:
            print("[IOU] manifold construction failed")
        return 0.0

    inter = _manifold_boolean(ma, mb, _OP_INTERSECT)
    uni = _manifold_boolean(ma, mb, _OP_ADD)
    if inter is None or uni is None:
        if verbose:
            print("[IOU] boolean failed")
        return 0.0

    v_inter = abs(inter.volume())
    v_uni = abs(uni.volume())
    if v_uni <= 0:
        return 0.0
    iou = v_inter / v_uni
    if verbose:
        print(f"[IOU] inter={v_inter:.6f} uni={v_uni:.6f} IoU={iou:.6f}")
    return iou
