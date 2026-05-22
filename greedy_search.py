#!/usr/bin/env python3
"""
CAD Sketch Optimization using Pre-computed Analysis Results

This script uses pre-computed results from:
- analyze_loop_translation.py (for extrusion heights)
- analyze_revolve_axis.py (for revolve operations)

Instead of searching for optimal parameters, it loads them from JSON files
and directly constructs the best extrusion/revolve operations, then runs
the greedy IoU-based pruning phase.
"""
import os
import sys
import json
import argparse
import numpy as np
import trimesh
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import signal
import multiprocessing as mp
from functools import partial
from fit_primitives import build_action_instances_from_entry
from iou import normalize_mesh
from cadquery_ops import (
    greedy_iou_based_pruning,
    execute_cadquery_script,
    generate_cadquery_script,
    compute_iou,
    generate_cadquery_revolve_script,
)


def load_translation_analysis(analysis_folder: str, sketch_idx: int) -> Optional[Dict]:
    """
    Load translation analysis results for a sketch.
    
    Args:
        analysis_folder: Folder containing analysis JSON files
        sketch_idx: Sketch index
        
    Returns:
        Dictionary with interval data or None if file doesn't exist or has no valid intervals
    """
    json_path = os.path.join(analysis_folder, f"sketch_{sketch_idx}_intervals.json")
    
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if there are valid intervals
    if not data.get('merged_intervals') and (data.get('all_intervals_min') is None or data.get('all_intervals_max') is None):
        return None
    
    return data


def load_revolve_analysis(analysis_folder: str, sketch_idx: int) -> List[Dict]:
    """
    Load revolve analysis results for a sketch (simplified JSONs).
    
    Args:
        analysis_folder: Folder containing analysis JSON files
        sketch_idx: Sketch index
        
    Returns:
        List of revolve configurations with axis_direction and mean_point
    """
    revolve_configs = []
    
    # Search for all simplified JSON files for this sketch
    import glob
    pattern = os.path.join(analysis_folder, f"sketch_{sketch_idx}_axis_*_simplified.json")
    
    for json_path in glob.glob(pattern):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if data.get('all_cd_under_threshold') and 'mean_point' in data and 'axis_direction' in data:
                revolve_configs.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_path}: {e}")
    
    return revolve_configs


def create_extrusion_from_translation(entry: Dict, translation_data: Dict, sketch_idx: int) -> List[Dict]:
    """
    Create extrusion operations from translation analysis data.
    
    Args:
        entry: Sketch entry from loops_split.json
        translation_data: Translation analysis results
        sketch_idx: Sketch index
        
    Returns:
        List of extrusion dictionaries with sketch actions and heights
    """
    extrusions = []
    
    # Get plane information
    origin = np.array(entry.get('origin', [0, 0, 0]))
    normal = np.array(entry.get('normal', [0, 0, 1]))
    normal = normal / np.linalg.norm(normal)
    
    # Use merged_intervals if available, otherwise fall back to old format
    intervals = translation_data.get('merged_intervals', [])
    
    if not intervals:
        # Fall back: build intervals from old fields
        interval_zero = translation_data.get('interval_containing_zero')
        all_min = translation_data.get('all_intervals_min')
        all_max = translation_data.get('all_intervals_max')
        if interval_zero is not None:
            intervals.append({'t_start': interval_zero['t_min'], 't_end': interval_zero['t_max']})
        if all_min is not None and all_max is not None:
            intervals.append({'t_start': all_min, 't_end': all_max})
    
    # If more than 5 intervals, keep the 5 largest by width
    if len(intervals) > 5:
        intervals = sorted(intervals, key=lambda iv: iv['t_end'] - iv['t_start'], reverse=True)[:5]
    
    import copy
    for iv_idx, interval in enumerate(intervals):
        t_min = interval['t_start']
        t_max = interval['t_end']
        
        # Move origin by t_min * normal
        new_origin = origin + t_min * normal
        height = t_max - t_min
        
        # Create modified entry with translated origin (deep copy to preserve loops_3d, etc.)
        modified_entry = copy.deepcopy(entry)
        modified_entry['origin'] = new_origin.tolist()
        
        # Build sketch action instances with modified origin
        insts = build_action_instances_from_entry(modified_entry, entry_idx=sketch_idx)
        
        if insts:
            extrusions.append({
                'type': 'extrude',
                'sketch_idx': sketch_idx,
                'actions': insts,
                'height': height,
                'origin': new_origin.tolist(),
                'normal': normal.tolist(),
                'xdir': entry.get('xdir', [1, 0, 0]),
                'source': f'interval_{iv_idx}',
                't_min': t_min,
                't_max': t_max
            })
    
    return extrusions


def _split_loop_along_axis_2d(pts2: np.ndarray, mp2: np.ndarray, ad2: np.ndarray):
    """Cut a closed 2D polyline by the axis line through mp2 with direction ad2.
    Keep the half whose signed perpendicular distance to the axis is positive.
    Returns a closed 2D polyline (Nx2) or None if the half is empty/degenerate.
    """
    perp = np.array([-ad2[1], ad2[0]])
    signed = (pts2 - mp2[None, :]).dot(perp)
    N = len(pts2)
    out = []
    for i in range(N):
        p, s = pts2[i], signed[i]
        sn = signed[(i + 1) % N]
        q = pts2[(i + 1) % N]
        if s >= 0:
            out.append(p)
        # Edge crosses the axis: insert the crossing point.
        # Consecutive crossings produce the on-axis closing segment automatically.
        if (s > 0 and sn < 0) or (s < 0 and sn > 0):
            t = s / (s - sn)
            out.append(p + t * (q - p))
    if len(out) < 3:
        return None
    out = np.asarray(out, dtype=float)
    if not np.allclose(out[0], out[-1]):
        out = np.vstack([out, out[0:1]])
    return out


def _refine_symmetry_axis_2d(pts2: np.ndarray, mp2: np.ndarray, ad2: np.ndarray,
                             angle_range_deg: float = 3.0, n_angle: int = 13,
                             offset_frac: float = 0.03, n_offset: int = 13):
    """Locally refine a 2D symmetry axis by minimizing reflection chamfer.
    Searches around (angle, perpendicular offset) of the initial axis."""
    import math
    from scipy.spatial import cKDTree

    # Subsample for speed
    M = len(pts2)
    if M > 500:
        idx = np.linspace(0, M - 1, 500).astype(int)
        sample = pts2[idx]
    else:
        sample = pts2

    tree = cKDTree(sample)
    bbox_diag = float(np.linalg.norm(pts2.max(axis=0) - pts2.min(axis=0)))
    init_angle = math.atan2(ad2[1], ad2[0])
    angles = np.linspace(init_angle - math.radians(angle_range_deg),
                         init_angle + math.radians(angle_range_deg), n_angle)
    offsets = np.linspace(-offset_frac * bbox_diag, offset_frac * bbox_diag, n_offset)

    best_cost = float('inf')
    best_mp, best_ad = mp2, ad2
    for ang in angles:
        ad_try = np.array([math.cos(ang), math.sin(ang)])
        perp_try = np.array([-ad_try[1], ad_try[0]])
        for off in offsets:
            mp_try = mp2 + off * perp_try
            d = (sample - mp_try).dot(perp_try)
            reflected = sample - 2.0 * d[:, None] * perp_try[None, :]
            dists, _ = tree.query(reflected)
            cost = float(np.mean(dists ** 2))
            if cost < best_cost:
                best_cost = cost
                best_mp = mp_try
                best_ad = ad_try
    return best_mp, best_ad, best_cost


def _build_split_entry_for_revolve(entry: Dict, mean_point_world: np.ndarray,
                                   axis_direction_world: np.ndarray):
    """Return a deep-copied entry with loops_3d replaced by the half-loop
    on one side of the symmetry axis. Returns None if the axis cannot be
    projected to 2D (e.g., axis is along the plane normal) or any loop fails to split.
    Also returns the refined axis (mean_point, axis_direction) in WORLD coords."""
    import copy
    from fit_primitives import plane_basis, project_points_to_plane

    origin = np.array(entry.get('origin', [0, 0, 0]), dtype=float)
    normal = entry.get('normal', [0, 0, 1])
    xdir = entry.get('xdir', [1, 0, 0])
    loops_3d = entry.get('loops_3d') or []
    if not loops_3d:
        return None, None, None

    u, v, n = plane_basis(normal, xdir)

    # Project axis direction to 2D
    ad_world = np.array(axis_direction_world, dtype=float)
    ad2 = np.array([ad_world.dot(u), ad_world.dot(v)])
    ad2_norm = float(np.linalg.norm(ad2))
    if ad2_norm < 1e-6:
        # Axis is along the plane normal — no in-plane axis to split on.
        return None, None, None
    ad2 = ad2 / ad2_norm

    # Project mean point to 2D
    mp_world = np.array(mean_point_world, dtype=float)
    rel = mp_world - origin
    mp2 = np.array([rel.dot(u), rel.dot(v)])

    # Project all loops to 2D
    all_pts2 = []
    pts2_per_loop = []
    for loop_3d in loops_3d:
        pts3 = np.array(loop_3d, dtype=float)
        if pts3.shape[0] > 2 and np.allclose(pts3[0], pts3[-1]):
            pts3 = pts3[:-1]
        pts2 = project_points_to_plane(pts3, origin.tolist(), normal, xdir)
        pts2_per_loop.append(pts2)
        all_pts2.append(pts2)
    all_pts2 = np.vstack(all_pts2)

    # Refine the 2D axis using all loop points
    try:
        mp2, ad2, _ = _refine_symmetry_axis_2d(all_pts2, mp2, ad2)
    except Exception as e:
        print(f"  Warning: axis refinement failed ({e}); using analyzer axis")

    # Split each loop. A loop entirely on the *wrong* side of the axis is
    # already the mirror of another loop on the right side — silently skip it
    # instead of bailing on the whole entry. Only bail if NO loop survived.
    new_loops_3d = []
    for pts2 in pts2_per_loop:
        half2 = _split_loop_along_axis_2d(pts2, mp2, ad2)
        if half2 is None or len(half2) < 3:
            continue
        # Unproject to 3D
        half3 = origin[None, :] + half2[:, 0:1] * u[None, :] + half2[:, 1:2] * v[None, :]
        new_loops_3d.append(half3.tolist())
    if not new_loops_3d:
        return None, None, None

    # Reconstruct refined axis in world coords
    refined_mp_world = origin + mp2[0] * u + mp2[1] * v
    refined_ad_world = ad2[0] * u + ad2[1] * v
    refined_ad_world = refined_ad_world / (np.linalg.norm(refined_ad_world) + 1e-12)

    new_entry = copy.deepcopy(entry)
    new_entry['loops_3d'] = new_loops_3d
    new_entry.pop('loops', None)
    return new_entry, refined_mp_world, refined_ad_world


def create_revolve_from_analysis(entry: Dict, revolve_configs: List[Dict], sketch_idx: int) -> List[Dict]:
    """
    Create revolve operations from revolve analysis data.

    For each candidate symmetry axis, split the sketch along the (refined) axis
    and revolve the resulting half-profile by 360°. Skips the candidate if the
    profile cannot be split — a 180° revolve of the full (symmetric) sketch
    relies on OCC to handle self-intersection across the axis and produces a
    degenerate body in practice.
    """
    revolves = []

    # Get plane information for the script
    origin = entry.get('origin', [0, 0, 0])
    normal = entry.get('normal', [0, 0, 1])
    xdir = entry.get('xdir', [1, 0, 0])

    ydir = np.cross(np.array(normal), np.array(xdir))
    rotation_matrix = np.array([xdir, ydir, normal]).T
    inverse_matrix = np.linalg.inv(rotation_matrix)

    for config in revolve_configs:
        mp_world = np.array(config['mean_point'])
        ad_world = np.array(config['axis_direction'])
        ad_world = ad_world / (np.linalg.norm(ad_world) + 1e-12)

        split_entry, refined_mp_world, refined_ad_world = _build_split_entry_for_revolve(
            entry, mp_world, ad_world)

        if split_entry is None:
            print(f"  Sketch {sketch_idx} axis {config.get('axis_idx', '?')}: profile split failed, skipping revolve candidate")
            continue
        insts = build_action_instances_from_entry(split_entry, entry_idx=sketch_idx)
        angle = 360.0
        mp_for_axis = refined_mp_world
        ad_for_axis = refined_ad_world
        split_used = True

        if not insts:
            continue

        # Axis in plane-local coords (existing convention used by the script generator)
        mp_local = inverse_matrix @ mp_for_axis
        ad_local = inverse_matrix @ ad_for_axis
        ad_local = ad_local / (np.linalg.norm(ad_local) + 1e-12)
        axis_start = mp_local
        axis_end = mp_local + ad_local

        revolves.append({
            'type': 'revolve',
            'sketch_idx': sketch_idx,
            'actions': insts,
            'axis_start': axis_start.tolist(),
            'axis_end': axis_end.tolist(),
            'angle': angle,
            'origin': origin,
            'normal': normal,
            'xdir': xdir,
            'axis_idx': config.get('axis_idx', 'unknown'),
            'rotation_type': config.get('rotation_type', 'unknown'),
            'split_profile': split_used,
        })

    return revolves


def _process_extrusion_worker(args):
    """Worker function for parallel extrusion processing."""
    sketch_idx, entry, translation_analysis_folder = args
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Extrusion processing timed out")
    
    # Per-sketch timeout. Dense slice loops (3000+ pts × many loops) make
    # `segment_greedy_fit` (~O(n²)) take 1-2 min on the worst sketches —
    # we still want to skip truly stuck workers but not drop legitimate ones.
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)

    try:
        translation_data = load_translation_analysis(translation_analysis_folder, sketch_idx)

        if translation_data is None:
            signal.alarm(0)  # Cancel the alarm
            return sketch_idx, []

        extrusions = create_extrusion_from_translation(entry, translation_data, sketch_idx)
        signal.alarm(0)  # Cancel the alarm
        return sketch_idx, extrusions
    except TimeoutError:
        signal.alarm(0)  # Cancel the alarm
        print(f"  ⏱️  Sketch {sketch_idx} extrusion processing timed out after 300 seconds")
        return sketch_idx, []
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        print(f"  Error processing sketch {sketch_idx} extrusion: {e}")
        return sketch_idx, []


def _process_revolve_worker(args):
    """Worker function for parallel revolve processing."""
    sketch_idx, entry, revolve_analysis_folder = args
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Revolve processing timed out")
    
    # Per-sketch timeout; mirror the extrusion worker's headroom.
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)

    try:
        revolve_configs = load_revolve_analysis(revolve_analysis_folder, sketch_idx)

        if not revolve_configs:
            signal.alarm(0)  # Cancel the alarm
            return sketch_idx, []

        revolves = create_revolve_from_analysis(entry, revolve_configs, sketch_idx)
        signal.alarm(0)  # Cancel the alarm
        return sketch_idx, revolves
    except TimeoutError:
        signal.alarm(0)  # Cancel the alarm
        print(f"  ⏱️  Sketch {sketch_idx} revolve processing timed out after 300 seconds")
        return sketch_idx, []
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        print(f"  Error processing sketch {sketch_idx} revolve: {e}")
        return sketch_idx, []

def precompute_operations_from_analysis(
    loops_split_path: str,
    loops_split_revolve_path: str,
    translation_analysis_folder: str,
    revolve_analysis_folder: str,
    output_folder: str = "search_precomputed",
    num_workers: int = 8
) -> Dict[str, Any]:
    """
    Load pre-computed analysis results and create operation list.
    
    Args:
        loops_split_path: Path to loops_split.json (for extrusion)
        loops_split_revolve_path: Path to loops_split_revolve.json (for revolve)
        translation_analysis_folder: Folder with translation analysis JSONs
        revolve_analysis_folder: Folder with revolve analysis JSONs
        output_folder: Output folder for results
        num_workers: Number of parallel workers
        
    Returns:
        Dictionary with precomputed operations
    """
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Loading loops from {loops_split_path}")
    with open(loops_split_path, 'r') as f:
        loops_data = json.load(f)
    
    print(f"Loading revolve loops from {loops_split_revolve_path}")
    with open(loops_split_revolve_path, 'r') as f:
        revolve_data = json.load(f)
    
    all_operations = []
    
    # Process extrusions in parallel
    print(f"\nProcessing extrusions from translation analysis (using {num_workers} workers)...")
    extrusion_count = 0
    
    # Prepare worker arguments
    extrusion_args = [(sketch_idx, entry, translation_analysis_folder) 
                      for sketch_idx, entry in enumerate(loops_data)]
    
    with mp.Pool(num_workers) as pool:
        results = pool.map(_process_extrusion_worker, extrusion_args)
    
    # Collect results
    for sketch_idx, extrusions in results:
        if extrusions:
            all_operations.extend(extrusions)
            extrusion_count += len(extrusions)
            print(f"  Sketch {sketch_idx}: {len(extrusions)} extrusion(s)")
    
    print(f"Total extrusions: {extrusion_count}")
    
    # Process revolves in parallel
    print(f"\nProcessing revolves from revolve analysis (using {num_workers} workers)...")
    revolve_count = 0
    
    # Prepare worker arguments
    revolve_args = [(sketch_idx, entry, revolve_analysis_folder) 
                    for sketch_idx, entry in enumerate(revolve_data)]
    
    with mp.Pool(num_workers) as pool:
        results = pool.map(_process_revolve_worker, revolve_args)
    
    # Collect results
    for sketch_idx, revolves in results:
        if revolves:
            all_operations.extend(revolves)
            revolve_count += len(revolves)
            print(f"  Sketch {sketch_idx}: {len(revolves)} revolve(s)")
    
    print(f"Total revolves: {revolve_count}")
    print(f"Total operations: {len(all_operations)}")
    
    # Save precomputed operations
    precomp_data = {
        'operations': all_operations,
        'total_extrusions': extrusion_count,
        'total_revolves': revolve_count,
        'total_operations': len(all_operations)
    }
    
    precomp_path = os.path.join(output_folder, 'precomputed_operations.json')
    with open(precomp_path, 'w') as f:
        json.dump(precomp_data, f, indent=2)
    
    print(f"\nSaved precomputed operations to {precomp_path}")
    
    return precomp_data


def _evaluate_operation_worker(args):
    """Worker function for parallel operation evaluation with timeout."""
    operation, gt_mesh_path, output_folder, alpha, normalize, points, inside_a = args
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Operation evaluation timed out")
    
    # Set 4 second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    try:
        result = evaluate_single_operation(
            operation=operation,
            gt_mesh_path=gt_mesh_path,
            output_folder=output_folder,
            alpha=alpha,
            normalize=normalize,
            points=points,
            inside_a=inside_a
        )
        signal.alarm(0)  # Cancel the alarm
        
        if result is not None and result.get('iou', 0.0) >= 0.0005:
            return result
        elif result is not None:
            print(f"  Skipping operation with IoU < 0.0005 (IoU = {result.get('iou', 0.0):.6f})")
            return None
        return None
    except TimeoutError:
        signal.alarm(0)  # Cancel the alarm
        print(f"  ⏱️  Operation timed out after 4 seconds, skipping...")
        return None
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        print(f"  Error evaluating operation: {e}")
        return None


def evaluate_single_operation(operation: Dict, gt_mesh_path: str, output_folder: str, 
                              alpha: float = 0.01, normalize: bool = True,
                              points=None, inside_a=None) -> Optional[Dict]:
    """
    Evaluate a single operation (extrude or revolve) and compute IoU.
    
    Args:
        operation: Operation dictionary
        gt_mesh_path: Path to ground truth mesh
        output_folder: Output folder
        alpha: Alpha for IoU computation
        normalize: Whether to normalize meshes
        points: Precomputed sample points
        inside_a: Precomputed ground truth inside mask
        
    Returns:
        Dictionary with operation results and IoU, or None if failed
    """
    sketch_idx = operation['sketch_idx']
    op_type = operation['type']
    
    # Create unique filename
    if op_type == 'extrude':
        source = operation.get('source', 'unknown')
        filename = f"sketch_{sketch_idx}_{source}"
    else:  # revolve
        axis_idx = operation.get('axis_idx', 'unknown')
        filename = f"sketch_{sketch_idx}_revolve_{axis_idx}"
    
    stl_path = os.path.join(output_folder, f"{filename}.stl")
    script_path = os.path.join(output_folder, f"{filename}.py")
    
    # Generate CadQuery script
    if op_type == 'extrude':
        script_content = generate_cadquery_script(
            sketch_actions_list=[operation['actions']],
            height=operation['height'],
            output_stl_path=stl_path
        )
    else:  # revolve
        script_content = generate_cadquery_revolve_script(
            sketch_actions=operation['actions'],
            axis_start=tuple(operation['axis_start']),
            axis_end=tuple(operation['axis_end']),
            angle_degrees=operation['angle'],
            output_stl_path=stl_path,
            origin=operation['origin'],
            normal=operation['normal'],
            xdir=operation['xdir']
        )
    
    # Save and execute script
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    success = execute_cadquery_script(script_content, stl_path, script_path)
    
    if not success or not os.path.exists(stl_path):
        print(f"  Failed to generate {filename}")
        return None
    
    # Compute IoU
    try:
        iou = compute_iou(gt_mesh_path, stl_path, normalize=normalize)
        
        result = operation.copy()
        result['iou'] = iou
        result['stl_path'] = stl_path
        result['script_path'] = script_path
        
        print(f"  {filename}: IoU = {iou:.6f}")
        
        return result
        
    except Exception as e:
        print(f"  Error computing IoU for {filename}: {e}")
        return None


def run_greedy_search(
    loops_split_path: str,
    loops_split_revolve_path: str,
    ground_truth_path: str,
    translation_analysis_folder: str,
    revolve_analysis_folder: str,
    output_folder: str = "search_precomputed",
    alpha: float = 0.01,
    normalize: bool = True,
    num_workers: int = 8
):
    """
    Run CAD reconstruction using pre-computed analysis results.
    
    Args:
        loops_split_path: Path to loops_split.json
        loops_split_revolve_path: Path to loops_split_revolve.json
        ground_truth_path: Path to ground truth STL
        translation_analysis_folder: Folder with translation analysis
        revolve_analysis_folder: Folder with revolve analysis
        output_folder: Output folder
        alpha: Alpha for IoU computation
        normalize: Whether to normalize meshes
        num_workers: Number of parallel workers
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: Load pre-computed operations
    print("="*80)
    print("STEP 1: Loading pre-computed operations")
    print("="*80)
    
    precomp_data = precompute_operations_from_analysis(
        loops_split_path=loops_split_path,
        loops_split_revolve_path=loops_split_revolve_path,
        translation_analysis_folder=translation_analysis_folder,
        revolve_analysis_folder=revolve_analysis_folder,
        output_folder=output_folder,
        num_workers=num_workers
    )
    
    operations = precomp_data['operations']
    
    if not operations:
        print("No valid operations found! Writing IoU=0 result.")
        final_iou_path = os.path.join(output_folder, 'final_iou.json')
        with open(final_iou_path, 'w') as f:
            json.dump({"iou": 0.0, "note": "no valid operations"}, f, indent=2)
        return
    
    # Step 2: Evaluate all operations
    print("\n" + "="*80)
    print("STEP 2: Evaluating all operations")
    print("="*80)
    
    # Load ground truth mesh for IoU computation
    print(f"Loading ground truth: {ground_truth_path}")
    gt_mesh = trimesh.load(ground_truth_path, force='mesh')
    
    if normalize:
        gt_mesh = normalize_mesh(gt_mesh)
    
    # CPU boolean IoU does not need sample-point precomputation
    inside_gt, points = None, None
    
    # Evaluate each operation in parallel
    print(f"\nEvaluating operations in parallel (using {num_workers} workers)...")
    
    # Prepare worker arguments
    eval_args = [(operation, ground_truth_path, output_folder, alpha, normalize, points, inside_gt) 
                 for operation in operations]
    
    # Was previously capped at 8 to avoid GPU contention during IoU sampling.
    # IoU is fully CPU now, so let it scale with the requested worker count.
    with mp.Pool(max(1, num_workers)) as pool:
        results = pool.map(_evaluate_operation_worker, eval_args)
    
    # Filter out None results
    evaluated_operations = [r for r in results if r is not None]
    
    print(f"\nSuccessfully evaluated {len(evaluated_operations)}/{len(operations)} operations (filtered IoU >= 0.0005)")
    
    # Save evaluated operations
    eval_path = os.path.join(output_folder, 'evaluated_operations.json')
    with open(eval_path, 'w') as f:
        json.dump(evaluated_operations, f, indent=2)
    print(f"Saved evaluated operations to {eval_path}")
    
    if not evaluated_operations:
        print("\nNo valid operations after evaluation! Writing IoU=0 result.")
        final_iou_path = os.path.join(output_folder, 'final_iou.json')
        with open(final_iou_path, 'w') as f:
            json.dump({"iou": 0.0, "note": "no valid operations"}, f, indent=2)
        return
    
    # Early stop: if any single operation already has IoU > 0.99, use it directly
    best_single = max(evaluated_operations, key=lambda r: r.get('iou', 0.0))
    if best_single.get('iou', 0.0) > 0.99:
        print(f"\n✓ Single operation already has IoU {best_single['iou']:.6f} > 0.99, skipping greedy search")
        import shutil, subprocess
        final_stl_path = os.path.join(output_folder, 'best_greedy_parallel.stl')
        if best_single.get('stl_path') and os.path.exists(best_single['stl_path']):
            shutil.copy2(best_single['stl_path'], final_stl_path)
        if best_single.get('script_path') and os.path.exists(best_single['script_path']):
            shutil.copy2(best_single['script_path'], os.path.join(output_folder, 'best_greedy_parallel.py'))
        print(f"\n{'='*80}")
        print("DONE!")
        print(f"Final IoU: {best_single['iou']:.6f}")
        print(f"{'='*80}")
        return
    
    # Step 3: Run greedy IoU-based pruning
    print("\n" + "="*80)
    print("STEP 3: Running greedy IoU-based pruning")
    print("="*80)
    
    # Convert to format expected by greedy_iou_based_pruning
    # The function expects a list of dicts, one per sketch, with fields:
    # insts, pos_h, pos_iou, neg_h, neg_iou, revolve_angle, revolve_iou, etc.
    
    # First, group by sketch_idx
    # IMPORTANT: Store actions separately for each operation since each has different translated origins
    sketches_by_idx = {}
    for result in evaluated_operations:
        sketch_idx = result['sketch_idx']
        if result['iou'] < 0.0005:
            continue
        
        if sketch_idx not in sketches_by_idx:
            sketches_by_idx[sketch_idx] = {
                'insts': result['actions'],  # Will be overwritten with best operation's actions
                'faces_idx': sketch_idx,
                'pos_h': None,
                'pos_iou': 0.0,
                'pos_actions': None,  # Store actions specific to positive extrusion
                'neg_h': None,
                'neg_iou': 0.0,
                'neg_actions': None,  # Store actions specific to negative extrusion
                'revolve_angle': None,
                'revolve_iou': 0.0,
                'revolve_axis_start': None,
                'revolve_axis_end': None,
                'revolve_actions': None,  # Store actions specific to revolve
                'is_symmetric': False,
                'use_revolve': False,
                'zero_only': True
            }
        
        if result['type'] == 'extrude':
            # Determine if positive or negative based on height
            height = result['height']
            iou_value = result['iou']
            
            if height > 0:
                # Keep the best positive extrusion (allow IoU >= 0)
                if sketches_by_idx[sketch_idx]['pos_h'] is None or sketches_by_idx[sketch_idx]['pos_iou'] < iou_value:
                    sketches_by_idx[sketch_idx]['pos_h'] = height
                    sketches_by_idx[sketch_idx]['pos_iou'] = iou_value
                    sketches_by_idx[sketch_idx]['pos_actions'] = result['actions']  # Store this operation's actions
                    sketches_by_idx[sketch_idx]['zero_only'] = False
            else:
                # Keep the best negative extrusion (allow IoU >= 0)
                if sketches_by_idx[sketch_idx]['neg_h'] is None or sketches_by_idx[sketch_idx]['neg_iou'] < iou_value:
                    sketches_by_idx[sketch_idx]['neg_h'] = abs(height)
                    sketches_by_idx[sketch_idx]['neg_iou'] = iou_value
                    sketches_by_idx[sketch_idx]['neg_actions'] = result['actions']  # Store this operation's actions
                    sketches_by_idx[sketch_idx]['zero_only'] = False
        
        else:  # revolve
            iou_value = result['iou']
            # Keep the best revolve operation (allow IoU >= 0)
            if sketches_by_idx[sketch_idx]['revolve_angle'] is None or sketches_by_idx[sketch_idx]['revolve_iou'] < iou_value:
                sketches_by_idx[sketch_idx]['revolve_angle'] = result['angle']
                sketches_by_idx[sketch_idx]['revolve_iou'] = iou_value
                sketches_by_idx[sketch_idx]['revolve_axis_start'] = result['axis_start']
                sketches_by_idx[sketch_idx]['revolve_axis_end'] = result['axis_end']
                sketches_by_idx[sketch_idx]['revolve_actions'] = result['actions']  # Store this operation's actions
                sketches_by_idx[sketch_idx]['is_symmetric'] = True
                sketches_by_idx[sketch_idx]['zero_only'] = False
                
                # Use revolve if it's better than extrusions
                best_extrude_iou = max(
                    sketches_by_idx[sketch_idx]['pos_iou'],
                    sketches_by_idx[sketch_idx]['neg_iou']
                )
                if iou_value > best_extrude_iou:
                    sketches_by_idx[sketch_idx]['use_revolve'] = True
    
    # Now determine which actions to use based on best operation
    for sketch_idx, sketch_data in sketches_by_idx.items():
        if sketch_data['use_revolve'] and sketch_data['revolve_actions'] is not None:
            sketch_data['insts'] = sketch_data['revolve_actions']
        elif sketch_data['pos_iou'] >= sketch_data['neg_iou'] and sketch_data['pos_actions'] is not None:
            sketch_data['insts'] = sketch_data['pos_actions']
        elif sketch_data['neg_actions'] is not None:
            sketch_data['insts'] = sketch_data['neg_actions']
        # If all are None, keep the default empty actions
    
    # Convert to list (sorted by sketch_idx)
    precomp_for_greedy = []
    max_sketch_idx = max(sketches_by_idx.keys()) if sketches_by_idx else 0
    
    for idx in range(max_sketch_idx + 1):
        if idx in sketches_by_idx:
            precomp_for_greedy.append(sketches_by_idx[idx])
        else:
            # Add dummy entry for missing sketches
            precomp_for_greedy.append({
                'insts': [],
                'faces_idx': idx,
                'pos_h': None,
                'pos_iou': 0.0,
                'neg_h': None,
                'neg_iou': 0.0,
                'revolve_angle': None,
                'revolve_iou': 0.0,
                'revolve_axis_start': None,
                'revolve_axis_end': None,
                'is_symmetric': False,
                'use_revolve': False,
                'zero_only': True
            })
    
    # Run greedy pruning
    final_iou, final_mesh, final_actions = greedy_iou_based_pruning(
        precomp=precomp_for_greedy,
        gt_mesh=gt_mesh,
        output_folder=output_folder,
        alpha=alpha,
        num_workers=num_workers,
    )
    
    # `greedy_tmp.stl` was built via manifold3d's robust union on each
    # candidate's STL and is clean + watertight — use that as the canonical
    # final STL. The saved `best_greedy_parallel.py` script is for human
    # inspection only (no exporter line); we don't execute it.
    best_greedy_script = os.path.join(output_folder, 'best_greedy_parallel.py')
    final_stl_path = os.path.join(output_folder, 'best_greedy_parallel.stl')
    import shutil as _shutil
    greedy_tmp = os.path.join(output_folder, 'greedy_tmp.stl')
    if os.path.exists(greedy_tmp):
        try:
            _shutil.copy2(greedy_tmp, final_stl_path)
            print(f"✓ Replaced {os.path.basename(final_stl_path)} with manifold3d-built greedy_tmp.stl")
        except Exception as e:
            print(f"⚠️  Could not copy greedy_tmp.stl over final STL: {e}")
    
    # The IoU returned by greedy_iou_based_pruning is for `greedy_tmp.stl`,
    # built via manifold3d unions on alpha-wrapped meshes. The rebuilt
    # `best_greedy_parallel.stl` uses CadQuery (OCC) `.union()`, which can
    # produce different geometry — so re-measure on the file that will
    # actually flow downstream.
    from iou import mesh_iou_cpu_boolean as _iou_cpu
    if os.path.exists(final_stl_path):
        try:
            rebuilt_iou = _iou_cpu(ground_truth_path, final_stl_path, normalize=True, verbose=False)
            print(f"\nRebuilt best_greedy_parallel.stl IoU vs GT: {rebuilt_iou:.6f}  (vs greedy_tmp {final_iou:.6f})")
            final_iou = rebuilt_iou
        except Exception as e:
            print(f"⚠️  Could not re-measure rebuilt IoU: {e}")

    # If any individual candidate beats the rebuilt mesh, swap.
    best_single = max(evaluated_operations, key=lambda r: r.get('iou', 0.0)) if evaluated_operations else {'iou': 0.0}
    if best_single.get('iou', 0.0) > final_iou:
        print(f"\n⚠️  Greedy IoU ({final_iou:.6f}) < best individual IoU ({best_single['iou']:.6f}), using individual instead")
        import shutil
        if best_single.get('stl_path') and os.path.exists(best_single['stl_path']):
            shutil.copy2(best_single['stl_path'], final_stl_path)
        if best_single.get('script_path') and os.path.exists(best_single['script_path']):
            shutil.copy2(best_single['script_path'], best_greedy_script)
        final_iou = best_single['iou']
    
    print("\n" + "="*80)
    print("DONE!")
    print(f"Final IoU: {final_iou:.6f}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CAD reconstruction using pre-computed analysis')
    parser.add_argument('loops_split', help='Path to loops_split.json')
    parser.add_argument('loops_split_revolve', help='Path to loops_split_revolve.json')
    parser.add_argument('ground_truth', help='Path to ground truth STL')
    parser.add_argument('translation_analysis', help='Folder with translation analysis JSONs')
    parser.add_argument('revolve_analysis', help='Folder with revolve analysis JSONs')
    parser.add_argument('--output-folder', default='search_precomputed', help='Output folder')
    parser.add_argument('--alpha', type=float, default=0.01, help='Alpha for IoU computation')
    parser.add_argument('--normalize', action='store_true', help='Normalize meshes')
    parser.add_argument('--num-workers', type=int, default=20, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    run_greedy_search(
        loops_split_path=args.loops_split,
        loops_split_revolve_path=args.loops_split_revolve,
        ground_truth_path=args.ground_truth,
        translation_analysis_folder=args.translation_analysis,
        revolve_analysis_folder=args.revolve_analysis,
        output_folder=args.output_folder,
        alpha=args.alpha,
        normalize=args.normalize,
        num_workers=args.num_workers
    )
