#!/usr/bin/env python3
"""
Filter loops in one or more loops_split-style JSONs using the trained MLP
classifier (DINO ViT-L MV + 17-d geom features).

For each loop in each JSON file:
  - render 4 PyVista views (iso_mesh, iso_loop, face_mesh, face_loop)
  - embed each view with DINOv2 ViT-L/14
  - compute 17-d geometric features from the loop polyline + plane
  - concat (4096+17 = 4113-d), standardize with saved (mu, sd)
  - score with the mlp_med MLP
Keep loops whose probability >= threshold. Overwrite each JSON in place,
saving the unfiltered original to <name>.unfiltered.json first.

Usage:
    python filter_loops.py --stl mesh.stl --json loops_split.json \\
        --model model_mlp_med.pt --norm normalization.npz \\
        --threshold 0.425
"""
import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def _setup_pyvista():
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv
    pv.OFF_SCREEN = True
    pv.global_theme.notebook = False
    if not os.environ.get("DISPLAY"):
        # Try to start an Xvfb; if it fails, error out clearly.
        try:
            pv.start_xvfb()
        except Exception as e:
            print(f"WARN: pv.start_xvfb() failed ({e}); relying on existing DISPLAY", flush=True)
    return pv


# ---------- mesh + render -------------------------------------------------

def _normalize_mesh(mesh, np):
    pts = np.asarray(mesh.points, dtype=np.float64)
    bb_min = pts.min(0); bb_max = pts.max(0)
    center = 0.5 * (bb_min + bb_max)
    max_dim = float((bb_max - bb_min).max())
    if max_dim > 0:
        scale = 2.0 / max_dim
        mesh.points = ((pts - center) * scale).astype(np.float32)
    return mesh


def _mesh_actor(plotter, mesh):
    plotter.add_mesh(
        mesh, color="#3a7bd5", opacity=0.18,
        show_edges=True, edge_color="#1f4e8a", line_width=0.6,
        smooth_shading=False,
    )


def _face_on_camera(plotter, origin, normal, xdir, np):
    o = np.asarray(origin, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    nrm = float(np.linalg.norm(n)) or 1.0
    n = n / nrm
    cam = o + n * 4.0
    view_up = np.asarray(xdir, dtype=np.float64)
    vu = float(np.linalg.norm(view_up)) or 1.0
    view_up = view_up / vu
    if abs(float(np.dot(view_up, n))) > 0.95:
        view_up = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    plotter.camera.position = tuple(cam.tolist())
    plotter.camera.focal_point = tuple(o.tolist())
    plotter.camera.up = tuple(view_up.tolist())
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 1.4


def _render_one(pv, np, mesh, loop_pts, plane, view, size=224, line_width=5, point_size=6):
    """Returns (mesh_only_img, mesh+loop_img) as HxWx3 uint8 numpy arrays."""
    # mesh-only
    p = pv.Plotter(off_screen=True, window_size=(size, size))
    p.set_background("white")
    _mesh_actor(p, mesh)
    if view == "iso":
        p.view_isometric(); p.reset_camera()
    else:
        _face_on_camera(p, *plane, np=np)
    img_mesh = p.screenshot(transparent_background=False, return_img=True)
    cam = p.camera_position
    p.close()
    # mesh + loop
    n = len(loop_pts)
    lines = np.empty(n + 1, dtype=np.int64)
    lines[0] = n
    lines[1:] = np.arange(n)
    poly = pv.PolyData(loop_pts)
    poly.lines = lines
    p2 = pv.Plotter(off_screen=True, window_size=(size, size))
    p2.set_background("white")
    _mesh_actor(p2, mesh)
    p2.add_mesh(poly, color="red", line_width=line_width)
    p2.add_points(loop_pts, color="red", point_size=point_size, render_points_as_spheres=True)
    p2.camera_position = cam
    img_loop = p2.screenshot(transparent_background=False, return_img=True)
    p2.close()
    return img_mesh, img_loop


# ---------- geometric features (17-d) ------------------------------------

def _geom_feats(entry, lp_idx, np):
    try:
        loop3d = entry["loops_3d"][lp_idx]
    except (IndexError, KeyError):
        return np.zeros(17, dtype=np.float32)
    pts = np.asarray(loop3d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return np.zeros(17, dtype=np.float32)
    n = len(pts)
    areas = entry.get("loop_areas") or [0.0] * (lp_idx + 1)
    area = float(areas[lp_idx]) if lp_idx < len(areas) else 0.0
    closed = 1 if (np.linalg.norm(pts[0] - pts[-1]) < 1e-6) else 0
    seg = np.diff(pts, axis=0)
    perim = float(np.linalg.norm(seg, axis=1).sum())
    bb_min = pts.min(0); bb_max = pts.max(0)
    bbox = (bb_max - bb_min).astype(np.float64)
    bbox_diag = float(np.linalg.norm(bbox))
    compact = perim / max(math.sqrt(max(area, 1e-12)), 1e-12)
    origin = np.asarray(entry.get("origin", [0, 0, 0]), dtype=np.float64)
    normal = np.asarray(entry.get("normal", [0, 0, 1]), dtype=np.float64)
    nl = float(np.linalg.norm(normal)) or 1.0
    normal = normal / nl
    axis_align = float(np.abs(normal).max())
    return np.array([
        area, math.log1p(max(area, 0.0)), n, closed, perim, compact,
        bbox[0], bbox[1], bbox[2], bbox_diag,
        origin[0], origin[1], origin[2],
        normal[0], normal[1], normal[2],
        axis_align,
    ], dtype=np.float32)


# ---------- DINO embedding ------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _preprocess(img_np, size=224):
    """img_np: HxWx3 uint8. Returns 3x224x224 float32 (ImageNet-normalized)."""
    from PIL import Image
    if img_np.shape[0] != size or img_np.shape[1] != size:
        img = Image.fromarray(img_np).convert("RGB").resize((size, size), Image.BILINEAR)
        img_np = np.asarray(img)
    if img_np.shape[-1] == 4:
        img_np = img_np[..., :3]
    arr = (img_np.astype(np.float32) / 255.0).transpose(2, 0, 1)
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def _embed_images(images, dino, device, bs=64):
    import torch
    out = []
    for i in range(0, len(images), bs):
        batch = images[i:i+bs]
        t = torch.from_numpy(np.stack(batch, axis=0)).to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                             dtype=torch.float16, enabled=(device == "cuda")):
            emb = dino(t)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------- classifier ----------------------------------------------------

def _build_mlp(arch):
    import torch.nn as nn
    h1, h2 = arch["hidden"]
    p = arch["dropout"]
    return nn.Sequential(
        nn.Linear(arch["D_in"], h1), nn.GELU(), nn.Dropout(p),
        nn.Linear(h1, h2), nn.GELU(), nn.Dropout(p),
        nn.Linear(h2, 1),
    )


# ---------- filtering driver ---------------------------------------------

def _flatten_loops(loops):
    """Return list of (sketch_idx, loop_idx, entry, loop_pts) and entry counts."""
    flat = []
    for si, entry in enumerate(loops):
        loops_3d = entry.get("loops_3d") or []
        for li, l3d in enumerate(loops_3d):
            if l3d and len(l3d) >= 2:
                flat.append((si, li, entry, np.asarray(l3d, dtype=np.float32)))
    return flat


def _rebuild_loops(loops, keep_mask, flat_idx):
    """Rebuild loops list, retaining only loops whose (si, li) is kept.
    Entries with zero remaining loops are dropped."""
    by_sketch = {}
    for k, (si, li, _, _) in enumerate(flat_idx):
        if keep_mask[k]:
            by_sketch.setdefault(si, []).append(li)
    out = []
    for si, entry in enumerate(loops):
        keep_lps = sorted(by_sketch.get(si, []))
        if not keep_lps:
            continue
        new = dict(entry)
        new["loops_3d"] = [entry["loops_3d"][i] for i in keep_lps]
        if "loop_areas" in entry and entry["loop_areas"] is not None:
            la = entry["loop_areas"]
            new["loop_areas"] = [la[i] if i < len(la) else 0.0 for i in keep_lps]
        out.append(new)
    return out


def filter_json(stl_path, json_path, dino, mlp, mu, sd, threshold, pv, device, np_,
                size=224, batch_size=64, keep_min=0, save_backup=True):
    import torch
    with open(json_path) as f:
        loops = json.load(f)
    flat = _flatten_loops(loops)
    n_total = len(flat)
    if n_total == 0:
        print(f"  {json_path}: 0 loops — nothing to filter", flush=True)
        return

    # Render
    mesh = pv.read(stl_path)
    _normalize_mesh(mesh, np_)
    all_imgs = []         # list of preprocessed 3x224x224 images, 4 per loop in order
    geoms = np_.zeros((n_total, 17), dtype=np_.float32)
    print(f"  {json_path}: rendering {n_total} loops...", flush=True)
    for k, (si, li, entry, loop_pts) in enumerate(flat):
        plane = (entry.get("origin", [0, 0, 0]),
                 entry.get("normal", [0, 0, 1]),
                 entry.get("xdir", [1, 0, 0]))
        try:
            iso_m, iso_l = _render_one(pv, np_, mesh, loop_pts, plane, "iso", size=size)
            fac_m, fac_l = _render_one(pv, np_, mesh, loop_pts, plane, "face", size=size)
        except Exception as e:
            # render failure -> blank images so the row still flows through
            blank = np_.full((size, size, 3), 255, dtype=np_.uint8)
            iso_m = iso_l = fac_m = fac_l = blank
        all_imgs.extend([_preprocess(iso_m), _preprocess(iso_l),
                         _preprocess(fac_m), _preprocess(fac_l)])
        geoms[k] = _geom_feats(entry, li, np_)

    # Embed (4 images per loop, embedded together for memory efficiency)
    print(f"  embedding {len(all_imgs)} images via DINO...", flush=True)
    embs = _embed_images(all_imgs, dino, device, bs=batch_size)
    D = embs.shape[1]
    feats_dino = embs.reshape(n_total, 4 * D).astype(np_.float32)
    feats = np_.concatenate([feats_dino, geoms], axis=1)

    # Standardize
    feats = (feats - mu) / sd

    # Score
    with torch.no_grad():
        x = torch.as_tensor(feats, dtype=torch.float32, device=device)
        probs = torch.sigmoid(mlp(x).squeeze(-1)).cpu().numpy()

    # Build keep mask
    keep_mask = probs >= threshold
    if keep_min > 0 and keep_mask.sum() < keep_min:
        order = np_.argsort(-probs)
        keep_idx = order[:keep_min]
        keep_mask = np_.zeros(n_total, dtype=bool)
        keep_mask[keep_idx] = True

    kept = int(keep_mask.sum())
    dropped = n_total - kept
    print(f"  scored {n_total} loops -> kept {kept} ({100*kept/n_total:.1f}%) "
          f"dropped {dropped}  (threshold={threshold:.3f})", flush=True)

    new_loops = _rebuild_loops(loops, keep_mask, flat)

    if save_backup:
        backup = json_path + ".unfiltered.json"
        if not os.path.exists(backup):
            shutil.copy2(json_path, backup)
    with open(json_path, "w") as f:
        json.dump(new_loops, f, indent=2)
    print(f"  wrote filtered {json_path}  ({len(loops)} -> {len(new_loops)} sketch entries)",
          flush=True)


# Default HuggingFace Hub location for the model weights.
HF_REPO = "ghadinehme/CADFit"
HF_MODEL_FILE = "model_mlp_med.pt"
HF_NORM_FILE = "normalization.npz"


def _hf_download(filename: str, override_path: str = None) -> str:
    """Return a local path to `filename`. If `override_path` is given and the
    file exists there, use it. Otherwise download from the HF Hub repo
    `ghadinehme/CADFit` (cached in ~/.cache/huggingface)."""
    if override_path and os.path.exists(override_path):
        return override_path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required for --filter-sketches when no "
            "--filter-model / --filter-norm path is given. "
            "`pip install huggingface-hub`."
        ) from e
    return hf_hub_download(repo_id=HF_REPO, filename=filename)


def load_filter(model_path: str = None, norm_path: str = None, device: str = None):
    """Load DINOv2 + MLP + normalization stats. Downloads from HuggingFace
    Hub if local paths are not provided. Returns
    (dino, mlp, mu, sd, threshold, device)."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_local = _hf_download(HF_MODEL_FILE, model_path)
    norm_local = _hf_download(HF_NORM_FILE, norm_path)

    ckpt = torch.load(model_local, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    threshold = float(ckpt.get("default_threshold", 0.425))
    dino_name = ckpt.get("dino_model", "dinov2_vitl14")
    print(f"loaded checkpoint  arch={arch}  dino={dino_name}  threshold={threshold}", flush=True)

    dino = torch.hub.load("facebookresearch/dinov2", dino_name, trust_repo=True)
    dino.eval().to(device)
    mlp = _build_mlp(arch).to(device)
    mlp.load_state_dict(ckpt["state_dict"])
    mlp.eval()

    norm = np.load(norm_local)
    mu = norm["mu"].astype(np.float32)
    sd = norm["sd"].astype(np.float32)
    return dino, mlp, mu, sd, threshold, device


def filter_jsons_inplace(stl_path: str, json_paths, threshold: float = None,
                          model_path: str = None, norm_path: str = None,
                          keep_min: int = 1, batch_size: int = 64,
                          save_backup: bool = True):
    """Programmatic entry point used by the pipeline. Filters each JSON in
    place; if `threshold` is None, falls back to the checkpoint's default."""
    pv = _setup_pyvista()
    dino, mlp, mu, sd, ckpt_thresh, device = load_filter(model_path, norm_path)
    thr = ckpt_thresh if threshold is None else float(threshold)
    for path in json_paths:
        if not os.path.exists(path):
            print(f"SKIP (missing): {path}", flush=True)
            continue
        filter_json(stl_path, path, dino, mlp, mu, sd, thr, pv, device, np,
                    batch_size=batch_size, keep_min=keep_min,
                    save_backup=save_backup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--json", nargs="+", required=True,
                    help="one or more loops_split-style JSON files to filter in place")
    ap.add_argument("--model", default=None,
                    help=f"path to {HF_MODEL_FILE}. If omitted, downloads from "
                         f"HuggingFace Hub repo `{HF_REPO}`.")
    ap.add_argument("--norm", default=None,
                    help=f"path to {HF_NORM_FILE}. If omitted, downloads from "
                         f"HuggingFace Hub repo `{HF_REPO}`.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="default = checkpoint's default_threshold (0.425)")
    ap.add_argument("--keep-min", type=int, default=1,
                    help="always keep at least this many loops (top-prob) to avoid empties")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    filter_jsons_inplace(args.stl, args.json,
                          threshold=args.threshold,
                          model_path=args.model, norm_path=args.norm,
                          keep_min=args.keep_min, batch_size=args.batch_size,
                          save_backup=not args.no_backup)


if __name__ == "__main__":
    main()
