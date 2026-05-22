"""Per-edge fillet/chamfer refinement.

Walks the OCP edge graph of a reconstructed solid, tests `fillet` and
`chamfer` at a small probe size on each modifiable edge, finds the best
size by an incremental sweep, then applies all selected operations to the
initial solid in one batched pass (grouped by size for kernel efficiency).

The IoU oracle is `iou.mesh_iou_cpu_boolean` — no GPU needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cadquery as cq
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_CurveType

from iou import mesh_iou_cpu_boolean


# Tunables — chosen to match the reference script.
PROBE_SIZE = 0.02
SWEEP_MIN = 0.02
SWEEP_MAX = 0.16
SWEEP_STEP = 0.02
MIN_EDGE_LENGTH = 0.05
TANGENT_TOL = 1e-3


@dataclass
class EdgeOp:
    """An applied per-edge refinement."""
    edge_index: int       # index into the sorted, filtered edge list
    op_kind: str          # 'fillet' or 'chamfer'
    size: float


# ---------------------------------------------------------------------------
# OCP edge utilities
# ---------------------------------------------------------------------------

def _faces_sharing_edge(solid_wrapped, edge):
    """Return the OCP faces that share `edge` (typically 2 for a watertight solid)."""
    faces = []
    exp_face = TopExp_Explorer(solid_wrapped, TopAbs_FACE)
    while exp_face.More():
        face = TopoDS.Face_s(exp_face.Current())
        exp_edge = TopExp_Explorer(face, TopAbs_EDGE)
        while exp_edge.More():
            fe = TopoDS.Edge_s(exp_edge.Current())
            if fe.IsEqual(edge):
                faces.append(face)
                break
            exp_edge.Next()
        exp_face.Next()
    return faces


def _face_normal(face):
    """Representative face normal as a cq.Vector."""
    surf = BRepAdaptor_Surface(face)
    if surf.GetType() == 0:  # planar
        n = surf.Plane().Axis().Direction()
    else:
        u_mid = 0.5 * (surf.FirstUParameter() + surf.LastUParameter())
        v_mid = 0.5 * (surf.FirstVParameter() + surf.LastVParameter())
        n = surf.Normal(u_mid, v_mid)
    return cq.Vector(n.X(), n.Y(), n.Z())


def find_filletable_edges(solid_wrapped,
                          min_length: float = 0.001,
                          tangent_tol: float = TANGENT_TOL) -> List:
    """Return OCP edges that are linear or circular, exceed `min_length`, and
    sit between non-tangent faces (so fillet / chamfer is well-defined)."""
    out = []
    exp = TopExp_Explorer(solid_wrapped, TopAbs_EDGE)
    while exp.More():
        edge = TopoDS.Edge_s(exp.Current())
        try:
            curve = BRepAdaptor_Curve(edge)
            curve_type = curve.GetType()
            length = cq.Edge(edge).Length()
            if length < min_length:
                exp.Next()
                continue
            if curve_type not in (GeomAbs_CurveType.GeomAbs_Line,
                                  GeomAbs_CurveType.GeomAbs_Circle):
                exp.Next()
                continue
            faces = _faces_sharing_edge(solid_wrapped, edge)
            if len(faces) == 2:
                n1, n2 = _face_normal(faces[0]), _face_normal(faces[1])
                if abs(1.0 - n1.dot(n2)) < tangent_tol:
                    exp.Next()
                    continue
            out.append(edge)
        except Exception as e:
            print(f"  [refine] skipping edge: {e}")
        exp.Next()
    return out


def _sorted_long_edges(solid_wrapped, min_length: float = MIN_EDGE_LENGTH):
    """Return edges from `find_filletable_edges` filtered by `min_length`,
    sorted longest-first. This is the canonical edge ordering used both
    when probing and when re-applying operations from an `EdgeOp` log."""
    raw = find_filletable_edges(solid_wrapped)
    annotated = []
    for e in raw:
        try:
            length = cq.Edge(e).Length()
        except Exception:
            continue
        if length >= min_length:
            annotated.append((e, length))
    annotated.sort(key=lambda t: t[1], reverse=True)
    return [t[0] for t in annotated]


# ---------------------------------------------------------------------------
# IoU oracle
# ---------------------------------------------------------------------------

def _iou_of(solid, gt_stl: str, temp_stl: str, normalize: bool) -> float:
    """Export `solid` to `temp_stl` and return its CPU IoU vs the ground truth."""
    try:
        solid.exportStl(temp_stl)
    except Exception as e:
        print(f"  [refine] export failed: {e}")
        return 0.0
    if not os.path.exists(temp_stl) or os.path.getsize(temp_stl) == 0:
        return 0.0
    try:
        return mesh_iou_cpu_boolean(gt_stl, temp_stl, normalize=normalize)
    except Exception as e:
        print(f"  [refine] IoU failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Per-edge optimizer
# ---------------------------------------------------------------------------

def _try_fillet(base_solid_wrapped, edge, size: float):
    """Apply fillet(size, [edge]) to a fresh cq.Solid wrapping `base_solid_wrapped`."""
    return cq.Solid(base_solid_wrapped).fillet(size, [cq.Edge(edge)])


def _try_chamfer(base_solid_wrapped, edge, size: float):
    return cq.Solid(base_solid_wrapped).chamfer(
        edgeList=[cq.Edge(edge)], length=size, length2=size)


def _sweep_sizes(base_solid_wrapped, edge, gt_stl: str, temp_folder: str,
                 op_kind: str, normalize: bool,
                 min_size: float = SWEEP_MIN,
                 max_size: float = SWEEP_MAX,
                 step: float = SWEEP_STEP):
    """Sweep size from `max_size` down to `min_size` (largest first; bigger
    radii fail more often, so finding any working size early caps work).
    Returns `(best_size, best_iou)` or `(None, -inf)` on total failure."""
    best_size = None
    best_iou = float("-inf")
    sizes = []
    s = max_size
    while s >= min_size - 1e-9:
        sizes.append(round(s, 3))
        s -= step
    for size in sizes:
        temp = f"{temp_folder}/sweep_{op_kind}_{size:.3f}.stl"
        try:
            if op_kind == "fillet":
                cand = _try_fillet(base_solid_wrapped, edge, size)
            else:
                cand = _try_chamfer(base_solid_wrapped, edge, size)
        except Exception:
            continue
        iou = _iou_of(cand, gt_stl, temp, normalize)
        if iou > best_iou:
            best_iou = iou
            best_size = size
    return best_size, best_iou


def _coalesce_workplane(wp: cq.Workplane) -> cq.Solid:
    """Return a single `cq.Solid` from a Workplane stack.

    The pipeline composes sub-solids with `.add()` (non-merging append), so a
    Workplane may carry multiple `cq.Solid`s. `fillet` / `chamfer` need a
    single connected solid, so we union them first when there's more than one.
    """
    vals = wp.vals()
    if not vals:
        raise ValueError("Workplane has no values")
    if len(vals) == 1:
        v = vals[0]
        if isinstance(v, cq.Solid):
            return v
        # If it's a Compound (e.g. from .add()), pull its inner solids out.
        try:
            inner = list(v.Solids())
        except Exception:
            inner = []
        if len(inner) == 1:
            return inner[0]
        if len(inner) > 1:
            merged = inner[0]
            for s in inner[1:]:
                merged = merged.fuse(s)
            return cq.Solid(merged.wrapped)
        return cq.Solid(v.wrapped)
    # Multiple stack values: fuse them all.
    merged = vals[0] if isinstance(vals[0], cq.Solid) else cq.Solid(vals[0].wrapped)
    for v in vals[1:]:
        s = v if isinstance(v, cq.Solid) else cq.Solid(v.wrapped)
        merged = merged.fuse(s)
    return cq.Solid(merged.wrapped)


def optimize_edges(initial_solid_wp: cq.Workplane,
                   gt_stl: str,
                   work_dir: str,
                   normalize: bool = True,
                   operation_type: str = "both",
                   verbose: bool = True
                   ) -> Tuple[cq.Solid, List[EdgeOp], float]:
    """Greedy per-edge fillet/chamfer refinement.

    For every modifiable edge of `initial_solid_wp` (longest-first), probe
    fillet + chamfer at `PROBE_SIZE` independently against the initial
    solid. If either improves IoU, sweep its size and keep the best. After
    all edges are probed, batch-apply the selected fillets (grouped by size,
    largest first), then the selected chamfers.

    Returns the final `cq.Solid`, the `EdgeOp` log, and the post-refinement
    IoU. The log uses indices into `_sorted_long_edges(initial_solid)`, so
    the same ops can later be re-applied to a freshly executed base script.
    """
    os.makedirs(work_dir, exist_ok=True)
    temp_folder = f"{work_dir}/temp"
    os.makedirs(temp_folder, exist_ok=True)

    try:
        initial_solid = _coalesce_workplane(initial_solid_wp)
    except Exception as e:
        print(f"  [refine] could not coalesce workplane: {e}")
        return initial_solid_wp.val(), [], 0.0
    initial_wrapped = initial_solid.wrapped

    base_iou = _iou_of(initial_solid, gt_stl, f"{temp_folder}/initial.stl", normalize)
    if verbose:
        print(f"  [refine] base IoU: {base_iou:.6f}")

    edges = _sorted_long_edges(initial_wrapped)
    if verbose:
        print(f"  [refine] {len(edges)} candidate edges (length >= {MIN_EDGE_LENGTH})")
    if not edges:
        return initial_solid, [], base_iou

    fillet_ops: List[Tuple[int, float]] = []   # (edge_index, size)
    chamfer_ops: List[Tuple[int, float]] = []  # (edge_index, size)

    for i, edge in enumerate(edges):
        edge_len = cq.Edge(edge).Length()
        if verbose:
            print(f"  [refine] edge {i+1}/{len(edges)}  length={edge_len:.5f}")
        fillet_iou = float("-inf")
        chamfer_iou = float("-inf")

        if operation_type in ("fillet", "both"):
            try:
                cand = _try_fillet(initial_wrapped, edge, PROBE_SIZE)
                fillet_iou = _iou_of(cand, gt_stl,
                                     f"{temp_folder}/probe_fillet_{i}.stl",
                                     normalize)
            except Exception as e:
                if verbose:
                    print(f"    fillet probe failed: {e}")

        if operation_type in ("chamfer", "both"):
            try:
                cand = _try_chamfer(initial_wrapped, edge, PROBE_SIZE)
                chamfer_iou = _iou_of(cand, gt_stl,
                                      f"{temp_folder}/probe_chamfer_{i}.stl",
                                      normalize)
            except Exception as e:
                if verbose:
                    print(f"    chamfer probe failed: {e}")

        if fillet_iou <= base_iou and chamfer_iou <= base_iou:
            if verbose:
                print(f"    no improvement (fillet={fillet_iou:.6f}, chamfer={chamfer_iou:.6f})")
            continue

        op_kind = "fillet" if fillet_iou >= chamfer_iou else "chamfer"
        best_size, best_iou = _sweep_sizes(initial_wrapped, edge, gt_stl,
                                           temp_folder, op_kind, normalize)
        if best_size is None or best_iou <= base_iou:
            if verbose:
                print(f"    sweep ({op_kind}) found no improvement")
            continue

        if verbose:
            print(f"    ✓ {op_kind}({best_size:.3f}) IoU {best_iou:.6f}  (+{best_iou-base_iou:.4f})")
        if op_kind == "fillet":
            fillet_ops.append((i, best_size))
        else:
            chamfer_ops.append((i, best_size))

    if not fillet_ops and not chamfer_ops:
        return initial_solid, [], base_iou

    # Batch-apply on a fresh copy of the initial solid.
    final_solid = cq.Solid(initial_wrapped)

    applied_log: List[EdgeOp] = []

    def _apply_group(current_solid, ops: List[Tuple[int, float]], op_kind: str):
        if not ops:
            return current_solid
        # Group by size, apply largest-first for kernel stability.
        by_size: dict = {}
        for idx, size in ops:
            by_size.setdefault(size, []).append(idx)
        for size in sorted(by_size.keys(), reverse=True):
            indices = by_size[size]
            edge_objs = [cq.Edge(edges[idx]) for idx in indices]
            try:
                if op_kind == "fillet":
                    current_solid = current_solid.fillet(size, edge_objs)
                else:
                    current_solid = current_solid.chamfer(
                        edgeList=edge_objs, length=size, length2=size)
                for idx in indices:
                    applied_log.append(EdgeOp(idx, op_kind, size))
                if verbose:
                    print(f"  [refine] applied {op_kind}({size:.3f}) to {len(indices)} edges")
            except Exception as e:
                if verbose:
                    print(f"  [refine] batch {op_kind}({size:.3f}) failed: {e}; skipping group")
        return current_solid

    final_solid = _apply_group(final_solid, fillet_ops, "fillet")
    final_solid = _apply_group(final_solid, chamfer_ops, "chamfer")

    if not applied_log:
        return initial_solid, [], base_iou

    final_stl = f"{work_dir}/final_refined.stl"
    final_iou = _iou_of(final_solid, gt_stl, final_stl, normalize)

    if final_iou <= base_iou:
        if verbose:
            print(f"  [refine] batched result {final_iou:.6f} did not beat base {base_iou:.6f}; rolling back")
        return initial_solid, [], base_iou

    if verbose:
        print(f"  [refine] final IoU {final_iou:.6f}  (+{final_iou-base_iou:.4f}) from {len(applied_log)} ops")
    return final_solid, applied_log, final_iou


# ---------------------------------------------------------------------------
# Script (re-)generation
# ---------------------------------------------------------------------------

_REFINE_HEADER = """# === CADFit per-edge fillet/chamfer refinement (auto-generated) ===
from cadquery import Edge as _CQEdge, Solid as _CQSolid
from OCP.TopAbs import TopAbs_EDGE as _TA_EDGE, TopAbs_FACE as _TA_FACE
from OCP.TopExp import TopExp_Explorer as _TopExp_Explorer
from OCP.TopoDS import TopoDS as _TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Curve as _BRA_Curve, BRepAdaptor_Surface as _BRA_Surface
from OCP.GeomAbs import GeomAbs_CurveType as _GACT
import cadquery as _cq

def _refine_faces_sharing(solid_w, edge):
    out = []
    ef = _TopExp_Explorer(solid_w, _TA_FACE)
    while ef.More():
        f = _TopoDS.Face_s(ef.Current())
        ee = _TopExp_Explorer(f, _TA_EDGE)
        while ee.More():
            fe = _TopoDS.Edge_s(ee.Current())
            if fe.IsEqual(edge):
                out.append(f); break
            ee.Next()
        ef.Next()
    return out

def _refine_face_normal(face):
    s = _BRA_Surface(face)
    if s.GetType() == 0:
        n = s.Plane().Axis().Direction()
    else:
        um = 0.5 * (s.FirstUParameter() + s.LastUParameter())
        vm = 0.5 * (s.FirstVParameter() + s.LastVParameter())
        n = s.Normal(um, vm)
    return _cq.Vector(n.X(), n.Y(), n.Z())

def _refine_sorted_long_edges(solid_w, min_length=0.05, tangent_tol=1e-3):
    out = []
    e = _TopExp_Explorer(solid_w, _TA_EDGE)
    while e.More():
        edge = _TopoDS.Edge_s(e.Current())
        try:
            c = _BRA_Curve(edge)
            if c.GetType() not in (_GACT.GeomAbs_Line, _GACT.GeomAbs_Circle):
                e.Next(); continue
            length = _CQEdge(edge).Length()
            if length < min_length:
                e.Next(); continue
            faces = _refine_faces_sharing(solid_w, edge)
            if len(faces) == 2:
                n1 = _refine_face_normal(faces[0])
                n2 = _refine_face_normal(faces[1])
                if abs(1.0 - n1.dot(n2)) < tangent_tol:
                    e.Next(); continue
            out.append((edge, length))
        except Exception:
            pass
        e.Next()
    out.sort(key=lambda t: t[1], reverse=True)
    return [t[0] for t in out]
"""


def render_refined_script(base_script_path: str, ops_log: List[EdgeOp],
                          output_path: str, output_stl_path: str) -> bool:
    """Generate a self-contained refined CadQuery script.

    Strategy: copy the base script verbatim up to its export call, append a
    refinement block that re-detects edges with the same sort, applies the
    operations from `ops_log` by edge index (grouped by size), and finally
    exports to `output_stl_path`.
    """
    try:
        with open(base_script_path) as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  [refine] base-script load failed: {e}")
        return False

    body, result_var = [], "result"
    for line in lines:
        if "exporters.export(" in line or ".exportStl(" in line:
            import re
            m = re.search(r"export\(\s*([A-Za-z_][A-Za-z0-9_]*)", line)
            if m:
                result_var = m.group(1)
            break
        if line.strip().startswith("print("):
            break
        body.append(line)

    fillet_groups: dict = {}
    chamfer_groups: dict = {}
    for op in ops_log:
        bucket = fillet_groups if op.op_kind == "fillet" else chamfer_groups
        bucket.setdefault(op.size, []).append(op.edge_index)

    out_lines: List[str] = []
    out_lines.extend(body)
    out_lines.append("\n")
    out_lines.append(_REFINE_HEADER)
    out_lines.append("\n")
    out_lines.append(f"{result_var} = _CQSolid({result_var}.val().wrapped)\n")
    out_lines.append(f"_refine_edges = _refine_sorted_long_edges({result_var}.wrapped)\n")

    for size in sorted(fillet_groups.keys(), reverse=True):
        idxs = fillet_groups[size]
        out_lines.append(
            f"try:\n"
            f"    _es = [_CQEdge(_refine_edges[i]) for i in {idxs} if i < len(_refine_edges)]\n"
            f"    if _es: {result_var} = {result_var}.fillet({size:.4f}, _es)\n"
            f"except Exception as _e:\n"
            f"    print('refined fillet failed:', _e)\n"
        )
    for size in sorted(chamfer_groups.keys(), reverse=True):
        idxs = chamfer_groups[size]
        out_lines.append(
            f"try:\n"
            f"    _es = [_CQEdge(_refine_edges[i]) for i in {idxs} if i < len(_refine_edges)]\n"
            f"    if _es: {result_var} = {result_var}.chamfer(edgeList=_es, length={size:.4f}, length2={size:.4f})\n"
            f"except Exception as _e:\n"
            f"    print('refined chamfer failed:', _e)\n"
        )

    out_lines.append(f"_cq.exporters.export({result_var}, '{output_stl_path}')\n")
    out_lines.append(f"print('Refined export →', '{output_stl_path}')\n")

    try:
        with open(output_path, "w") as f:
            f.writelines(out_lines)
        return True
    except Exception as e:
        print(f"  [refine] script write failed: {e}")
        return False


def _ops_log_to_json(ops_log: List[EdgeOp]) -> list:
    return [{"edge_index": op.edge_index, "op_kind": op.op_kind, "size": op.size}
            for op in ops_log]


def run_optimization(base_script: str, gt_stl: str, work_dir: str,
                     output_stl: str, output_log: str,
                     normalize: bool = True,
                     operation_type: str = "both"):
    """Crash-isolated entry point: load the solid from `base_script`,
    run `optimize_edges`, save the refined STL and ops log JSON.

    Designed to be invoked via subprocess so OCP segfaults inside `fillet` /
    `chamfer` don't bring the main pipeline down.
    """
    import json
    os.makedirs(work_dir, exist_ok=True)
    solid_wp = load_solid_from_script(base_script)
    if solid_wp is None:
        raise SystemExit("could not load solid from script")
    refined, ops_log, final_iou = optimize_edges(
        initial_solid_wp=solid_wp,
        gt_stl=gt_stl,
        work_dir=work_dir,
        normalize=normalize,
        operation_type=operation_type,
        verbose=True,
    )
    if not ops_log:
        # Nothing applied; signal no-op via empty STL + empty log.
        with open(output_log, "w") as f:
            json.dump({"ops": [], "in_memory_iou": final_iou}, f)
        return
    refined.exportStl(output_stl)
    with open(output_log, "w") as f:
        json.dump({"ops": _ops_log_to_json(ops_log),
                   "in_memory_iou": final_iou}, f, indent=2)


def load_solid_from_script(script_path: str) -> Optional[cq.Workplane]:
    """Exec a CadQuery script and return its `final_result` / `result` workplane.

    Removes any `exportStl` / `cq.exporters.export` / `print()` lines first
    so executing the script in-process doesn't do disk I/O."""
    try:
        with open(script_path) as f:
            src_lines = f.readlines()
    except Exception as e:
        print(f"  [refine] script load failed: {e}")
        return None
    keep = []
    for line in src_lines:
        if ".exportStl(" in line or "exporters.export(" in line:
            continue
        if line.strip().startswith("print("):
            continue
        keep.append(line)
    namespace: dict = {}
    try:
        exec("".join(keep), namespace)
    except Exception as e:
        print(f"  [refine] script exec failed: {e}")
        return None
    for name in ("final_result", "result"):
        if name in namespace and namespace[name] is not None:
            return namespace[name]
    for value in namespace.values():
        if isinstance(value, cq.Workplane):
            return value
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Crash-isolated per-edge refinement worker.")
    ap.add_argument("--base-script", required=True)
    ap.add_argument("--gt-stl", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--output-stl", required=True)
    ap.add_argument("--output-log", required=True)
    ap.add_argument("--operation-type", default="both", choices=["fillet", "chamfer", "both"])
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args()
    run_optimization(
        base_script=args.base_script,
        gt_stl=args.gt_stl,
        work_dir=args.work_dir,
        output_stl=args.output_stl,
        output_log=args.output_log,
        normalize=not args.no_normalize,
        operation_type=args.operation_type,
    )
