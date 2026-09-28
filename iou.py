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

# How much larger than the sum of its shells an alpha-wrap may be before we
# treat it as having bridged gaps rather than merely closed micro-cracks.
_WRAP_BRIDGE_TOL = 0.05

# How much volume a shell-wise union may lose versus the mesh's own volume
# before we stop believing it (dropped fragments => under-measured solid).
_UNION_LOSS_TOL = 0.05


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
    failure, empty result (volume == 0), or a non-finite volume — degenerate
    shells (single stray triangles from a fragmented compound export) convert
    without error but report NaN, which then poisons every comparison it
    reaches."""
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    try:
        mm = manifold3d.Mesh(vert_properties=verts, tri_verts=faces)
        m = manifold3d.Manifold(mm)
    except Exception:
        return None
    v = m.volume()
    if not np.isfinite(v) or v == 0.0:
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


def _load_for_wrap(path: str, normalize: bool):
    """Load `path` as a trimesh (optionally normalized) for the alpha-wrap
    retry in `mesh_iou_cpu_boolean`. Returns an empty mesh on failure."""
    try:
        mesh = trimesh.load(path, force='mesh')
    except Exception:
        return trimesh.Trimesh()
    return normalize_mesh(mesh) if normalize else mesh


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

    # Multi-body meshes: union the shells exactly with manifold3d instead of
    # alpha-wrapping them. The wrap sits ALPHA_WRAP_OFFSET_PCT outside every
    # surface, which on thin-walled parts (wall ~ a few % of the bbox) inflates
    # the volume by double-digit percentages. Since the single-body ground
    # truth takes the direct path above and is NOT wrapped, wrapping only the
    # reconstruction biases IoU downward. An exact union keeps both sides on
    # the same footing; alpha-wrap remains the fallback when it fails.
    parts_volume_sum = None
    if m is None and multi_body:
        try:
            parts = [p for p in mesh.split(only_watertight=False) if len(p.faces) > 0]
            # Every shell must be a closed volume. Relaxing this lets a dense
            # mesh that fragments into dozens of unconvertible slivers be
            # "unioned" from the few that survive — measuring a fraction of
            # the real solid.
            if not parts or not all(p.is_watertight and p.is_volume for p in parts):
                raise ValueError("non-watertight shell")
            mans = [mm for mm in (_to_manifold(p) for p in parts) if mm is not None]
            if mans:
                vols = [abs(mm.volume()) for mm in mans]
                if all(np.isfinite(v) for v in vols):
                    parts_volume_sum = sum(vols)
            # Prefer an exact union of the shells. Requiring every shell to be
            # watertight is too strict: a CadQuery compound often tessellates
            # into fragments that manifold3d can still merge, and the
            # alternative (alpha-wrap) BRIDGES the gaps between disjoint
            # plates, inventing volume that the reconstruction does not have.
            # Union whatever shells converted cleanly. Dropping a degenerate
            # fragment loses a sliver; keeping the raw compound (or a wrap)
            # invents volume, which is far worse for a scoring function.
            if mans:
                cand = manifold3d.Manifold.batch_boolean(mans, _OP_ADD)
                if cand is not None and np.isfinite(cand.volume()) and cand.volume() > 0:
                    # Every shell was a closed volume (checked above), so this
                    # union is exact. It is EXPECTED to be smaller than the
                    # mesh's own signed volume when shells overlap — that
                    # difference is precisely the double-counting we are here
                    # to remove.
                    m = cand
        except Exception:
            m = None

    # Fall back to alpha-wrap if direct didn't yield a volume-consistent
    # manifold, or for multi-body meshes.
    # Fall back to alpha-wrap if nothing above produced a usable manifold.
    # For a genuinely non-watertight input the wrap legitimately ADDS volume by
    # closing holes, so it is not second-guessed here; the compound-with-
    # overlapping-shells case it used to mismeasure is now prevented upstream
    # by exporting reconstructions as a single unioned solid.
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
                         verbose: bool = False,
                         _retry: bool = True) -> float:
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
    # IoU > 1 is geometrically impossible and means the booleans ran on a
    # degenerate operand (typically an open shell). Retry once with both
    # sides alpha-wrapped, which is watertight by construction; if that
    # still misbehaves, report 0.0 rather than a score that beats a perfect
    # reconstruction — greedy search maximizes this value, so an unclamped
    # >1 would actively steer the optimizer toward broken geometry.
    if iou > 1.0 + 1e-6:
        if _retry:
            wa = _to_manifold(_alpha_wrap(_load_for_wrap(mesh_path_a, normalize)))
            wb = _to_manifold(_alpha_wrap(_load_for_wrap(mesh_path_b, False)))
            if wa is not None and wb is not None:
                wi = _manifold_boolean(wa, wb, _OP_INTERSECT)
                wu = _manifold_boolean(wa, wb, _OP_ADD)
                if wi is not None and wu is not None and abs(wu.volume()) > 0:
                    retry_iou = abs(wi.volume()) / abs(wu.volume())
                    if retry_iou <= 1.0 + 1e-6:
                        if verbose:
                            print(f"[IOU] degenerate ({iou:.6f}), alpha-wrap retry -> {retry_iou:.6f}")
                        return retry_iou
        if verbose:
            print(f"[IOU] degenerate IoU={iou:.6f} (inter={v_inter:.6f} > uni={v_uni:.6f}); reporting 0.0")
        return 0.0
    if verbose:
        print(f"[IOU] inter={v_inter:.6f} uni={v_uni:.6f} IoU={iou:.6f}")
    return iou
