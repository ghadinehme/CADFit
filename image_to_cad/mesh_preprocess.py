"""Post-process a raw Hunyuan3D mesh into something CADFit can consume.

Apply Taubin smoothing (suppresses high-frequency tessellation artifacts
without shrinking the shape the way pure Laplacian smoothing would), then
quadric-collapse-decimate to a fixed face budget. The CADFit pipeline does
its own decimation pass too, but lowering face count up front keeps the
slicing / chamfer steps fast.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


DEFAULT_TARGET_FACES = 50_000
DEFAULT_TAUBIN_STEPS = 20
DEFAULT_TAUBIN_LAMBDA = 0.5
DEFAULT_TAUBIN_MU = -0.53


def preprocess_mesh(in_path: str,
                     out_path: str,
                     target_faces: int = DEFAULT_TARGET_FACES,
                     taubin_steps: int = DEFAULT_TAUBIN_STEPS,
                     taubin_lambda: float = DEFAULT_TAUBIN_LAMBDA,
                     taubin_mu: float = DEFAULT_TAUBIN_MU,
                     fill_holes: bool = True,
                     min_component_faces: int = 200) -> str:
    """Smooth + decimate the mesh at `in_path` and write the result to `out_path`.

    Uses pymeshlab when available (cleaner topology + Taubin smoother), and
    falls back to trimesh's filters otherwise.
    Returns `out_path`.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    try:
        import pymeshlab  # noqa: F401
        return _preprocess_pymeshlab(in_path, out_path,
                                       target_faces=target_faces,
                                       taubin_steps=taubin_steps,
                                       taubin_lambda=taubin_lambda,
                                       taubin_mu=taubin_mu,
                                       fill_holes=fill_holes,
                                       min_component_faces=min_component_faces)
    except ImportError:
        return _preprocess_trimesh(in_path, out_path,
                                    target_faces=target_faces,
                                    taubin_steps=taubin_steps,
                                    taubin_lambda=taubin_lambda,
                                    taubin_mu=taubin_mu)


def _preprocess_pymeshlab(in_path, out_path, *, target_faces, taubin_steps,
                           taubin_lambda, taubin_mu, fill_holes, min_component_faces):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(in_path)

    # Topology cleanup.
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_null_faces()
    if min_component_faces > 0:
        try:
            ms.meshing_remove_connected_component_by_face_number(
                mincomponentsize=min_component_faces)
        except Exception:
            pass
    if fill_holes:
        try:
            ms.meshing_close_holes(maxholesize=200)
        except Exception:
            pass

    # Taubin smoothing.
    if taubin_steps > 0:
        ms.apply_coord_taubin_smoothing(
            stepsmoothnum=taubin_steps,
            lambda_=taubin_lambda,
            mu=taubin_mu,
        )

    # Decimate to target face count.
    if target_faces > 0 and ms.current_mesh().face_number() > target_faces:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preservenormal=True,
            preservetopology=True,
            optimalplacement=True,
            qualitythr=1.0,
        )

    ms.meshing_remove_unreferenced_vertices()
    ms.save_current_mesh(out_path)
    return out_path


def _preprocess_trimesh(in_path, out_path, *, target_faces, taubin_steps,
                         taubin_lambda, taubin_mu):
    import trimesh
    from trimesh.smoothing import filter_taubin
    mesh = trimesh.load(in_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    if taubin_steps > 0:
        mesh = filter_taubin(mesh, lamb=taubin_lambda, nu=taubin_mu,
                             iterations=taubin_steps)
    if target_faces > 0 and len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)
    mesh.export(out_path)
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Taubin smooth + decimate a mesh.")
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    ap.add_argument("--target-faces", type=int, default=DEFAULT_TARGET_FACES)
    ap.add_argument("--taubin-steps", type=int, default=DEFAULT_TAUBIN_STEPS)
    ap.add_argument("--taubin-lambda", type=float, default=DEFAULT_TAUBIN_LAMBDA)
    ap.add_argument("--taubin-mu", type=float, default=DEFAULT_TAUBIN_MU)
    args = ap.parse_args()
    out = preprocess_mesh(args.in_path, args.out_path,
                          target_faces=args.target_faces,
                          taubin_steps=args.taubin_steps,
                          taubin_lambda=args.taubin_lambda,
                          taubin_mu=args.taubin_mu)
    print(f"wrote {out}")
