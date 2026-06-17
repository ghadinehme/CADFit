#!/usr/bin/env python3
"""
Analyze loop translation along normal direction by computing Chamfer Distance to surface points.

This script:
1. Loads and normalizes an STL file
2. Samples points on the surface
3. Loads loops from loops_split.json
4. Translates each loop along its normal direction (0 to 2 units, step 0.01)
5. Computes CD between translated loop points and surface points
6. Plots CD vs translation factor for each loop
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import argparse
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iou import normalize_mesh
from scipy.spatial import cKDTree
from tqdm import tqdm
import os
import multiprocessing as mp
from functools import partial

cd_threshold = 0.05

def chamfer_distance(points_a, points_b):
    """
    Compute one-sided Chamfer Distance from points_a to points_b.
    Fallback CPU version using scipy.
    
    Args:
        points_a: Nx3 array of source points
        points_b: Mx3 array of target points
        
    Returns:
        Mean distance from each point in A to nearest point in B
    """
    # Build KD-tree for fast nearest-neighbor search
    tree_b = cKDTree(points_b)
    
    # Distance from each point in A to closest point in B
    dists, _ = tree_b.query(points_a, k=1)
    
    # Return mean distance
    return np.mean(dists)


def sample_surface_points(mesh, num_samples=100000):
    """
    Sample points uniformly on the surface of a mesh.
    
    Args:
        mesh: Trimesh object
        num_samples: Number of points to sample
        
    Returns:
        Nx3 array of surface points
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_samples)
    return points


def translate_loop_along_normal(loop_3d, normal, translation_factor):
    """
    Translate a 3D loop along its normal direction.
    
    Args:
        loop_3d: Nx3 array of 3D points
        normal: 3D normal vector
        translation_factor: Distance to translate along normal
        
    Returns:
        Translated Nx3 array of points
    """
    loop_3d = np.array(loop_3d)
    normal = np.array(normal)
    
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    # Translate all points
    translated = loop_3d + translation_factor * normal
    
    return translated


def compute_cd_for_translations(loop_3d, normal, surface_points, translation_range=(0, 2), step=0.01):
    """Compute one-sided Chamfer Distance between a translated loop and the
    target surface for a range of translation factors along `normal`.

    Builds the surface KD-tree once per call (the chamfer_distance helper
    rebuilds it each time — that 100k-point tree was being constructed
    ~400× per sketch, which dominated runtime on dense ABC-hard meshes).
    """
    t_min, t_max = translation_range
    translation_factors = np.arange(t_min, t_max + step, step)
    tree = cKDTree(surface_points)
    loop_3d = np.asarray(loop_3d)
    normal = np.asarray(normal, dtype=float)
    nrm = np.linalg.norm(normal)
    if nrm > 0:
        normal = normal / nrm
    cd_values = np.empty(translation_factors.shape[0])
    for i, t in enumerate(translation_factors):
        translated = loop_3d + t * normal
        dists, _ = tree.query(translated, k=1)
        cd_values[i] = dists.mean()
    return translation_factors, cd_values


def merge_overlapping_intervals(intervals):
    """
    Merge overlapping or adjacent intervals.
    
    Args:
        intervals: List of interval dictionaries with t_start, t_end, etc.
        
    Returns:
        List of merged intervals
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x['t_start'])
    
    merged = [sorted_intervals[0].copy()]
    
    for current in sorted_intervals[1:]:
        last_merged = merged[-1]
        
        # Check if intervals overlap or are adjacent
        if current['t_start'] <= last_merged['t_end']:
            # Merge: extend the end and update statistics
            last_merged['t_end'] = max(last_merged['t_end'], current['t_end'])
            last_merged['end_idx'] = max(last_merged['end_idx'], current['end_idx'])
            
            # Update CD statistics by taking the union
            last_merged['mean_cd'] = (last_merged['mean_cd'] + current['mean_cd']) / 2.0
            last_merged['min_cd'] = min(last_merged['min_cd'], current['min_cd'])
            last_merged['max_cd'] = max(last_merged['max_cd'], current['max_cd'])
        else:
            # No overlap, add as new interval
            merged.append(current.copy())
    
    return merged


def plot_cd_vs_translation_all_loops(loops_data, sketch_idx, output_folder, save_plots=True):
    """
    Plot Chamfer Distance vs translation factor for all loops in a sketch.
    
    Args:
        loops_data: List of tuples (loop_idx, translation_factors, cd_values)
        sketch_idx: Sketch index
        output_folder: Output folder for plots
        save_plots: Whether to save visualization plots
        
    Returns:
        Dictionary with interval data for each loop and mean curve
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Store interval data
    intervals_data = {
        'sketch_idx': sketch_idx,
        'loop_intervals': [],
        'mean_intervals': [],
        'interval_containing_zero': None,
        'all_intervals_min': None,
        'all_intervals_max': None
    }
    
    # Colors for different loops
    colors = ['blue', 'orange', 'purple', 'green', 'cyan', 'magenta', 'brown','red']
    
    if save_plots:
        plt.figure(figsize=(12, 7))
    
    # Collect all cd_values for mean calculation
    all_cd_arrays = []
    
    for idx, (loop_idx, translation_factors, cd_values) in enumerate(loops_data):
        color = colors[idx % len(colors)]
        if save_plots:
            plt.plot(translation_factors, cd_values, linewidth=2, color=color, label=f'Loop {loop_idx}', alpha=0.6)
        all_cd_arrays.append(cd_values)
        
        # Compute absolute derivative
        abs_derivative = np.abs(np.gradient(cd_values, translation_factors))
        
        # Find where derivative crosses the threshold (0.2)
        # We want to check every interval between consecutive crossings
        low_deriv_mask = abs_derivative < 0.2
        
        # Store intervals for this loop
        loop_intervals = []
        
        # Find all transition points where derivative crosses threshold
        # Add boundary conditions if derivative is low at boundaries
        transitions = []
        
        # Check left boundary
        if abs_derivative[0] < 0.2:
            transitions.append(0)
        
        # Find all crossing points
        diff = np.diff(np.concatenate(([False], low_deriv_mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]  # Low derivative starts
        ends = np.where(diff == -1)[0]    # Low derivative ends
        
        # Add all crossing points
        for start, end in zip(starts, ends):
            if start > 0:  # Don't duplicate if already added as boundary
                transitions.append(start)
            if end < len(abs_derivative):  # Don't add if at the end boundary
                transitions.append(end)
        
        # Check right boundary
        if abs_derivative[-1] < 0.2 and len(abs_derivative) not in transitions:
            transitions.append(len(abs_derivative))
        
        # Sort transitions
        transitions = sorted(set(transitions))
        
        # Check each interval between consecutive transitions
        for i in range(len(transitions) - 1):
            interval_start = transitions[i]
            interval_end = transitions[i + 1]
            
            # Check if CD values in this interval are all < cd_threshold
            interval_cd_values = cd_values[interval_start:interval_end]
            if np.mean(interval_cd_values) < cd_threshold:
                # Expand interval by one step on each side, clipped to valid range
                expanded_start = max(0, interval_start - 1)
                expanded_end = min(len(translation_factors) - 1, interval_end)
                
                # Save interval data
                loop_intervals.append({
                    't_start': float(translation_factors[expanded_start]),
                    't_end': float(translation_factors[expanded_end]),
                    'start_idx': int(expanded_start),
                    'end_idx': int(expanded_end),
                    'mean_cd': float(np.mean(cd_values[expanded_start:expanded_end+1])),
                    'min_cd': float(np.min(cd_values[expanded_start:expanded_end+1])),
                    'max_cd': float(np.max(cd_values[expanded_start:expanded_end+1]))
                })
                
                if save_plots:
                    plt.axvspan(translation_factors[expanded_start], translation_factors[expanded_end], 
                               alpha=0.15, color=color, zorder=0)
        
        # Merge overlapping intervals for this loop
        merged_loop_intervals = merge_overlapping_intervals(loop_intervals)
        
        # Remove intervals smaller than 0.02 after merging
        #                            if (interval['t_end'] - interval['t_start']) >= 0.02]
        
        intervals_data['loop_intervals'].append({
            'loop_idx': int(loop_idx),
            'intervals': merged_loop_intervals
        })
        
        # Only mark individual loop minima if there's only one loop
        if len(loops_data) == 1 and save_plots:
            # Mark minimum CD for each loop
            min_idx = np.argmin(cd_values)
            min_t = translation_factors[min_idx]
            min_cd = cd_values[min_idx]

            # Find second minimum (local minimum that's not the global minimum)
            # Use scipy's find_peaks on inverted signal to find minima
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(-cd_values, prominence=0.001)  # Find local minima
            
            # Filter out the global minimum and find next best
            valid_peaks = [p for p in peaks if p != min_idx]
            if valid_peaks:
                # Sort by CD value and take the second best (smallest)
                valid_peaks_sorted = sorted(valid_peaks, key=lambda p: cd_values[p])
                second_min_idx = valid_peaks_sorted[0]
                second_min_t = translation_factors[second_min_idx]
                second_min_cd = cd_values[second_min_idx]
                print(f"  Loop {loop_idx}: Min CD={min_cd:.6f} at t={min_t:.2f}, 2nd min CD={second_min_cd:.6f} at t={second_min_t:.2f}")
            else:
                print(f"  Loop {loop_idx}: Min CD={min_cd:.6f} at t={min_t:.2f}")
    
    # Plot mean CD curve if there are multiple loops
    if len(loops_data) > 1:
        mean_cd = np.mean(all_cd_arrays, axis=0)
        if save_plots:
            plt.plot(translation_factors, mean_cd, linewidth=3, color='black', 
                     linestyle='--', label='Mean CD', alpha=0.8, zorder=100)
        
        # Compute absolute derivative for mean curve
        mean_abs_derivative = np.abs(np.gradient(mean_cd, translation_factors))
        # Find where derivative crosses the threshold (0.2)
        mean_low_deriv_mask = mean_abs_derivative < 0.2

        # Find all transition points where derivative crosses threshold
        transitions = []
        
        # Check left boundary
        if mean_abs_derivative[0] < 0.2:
            transitions.append(0)
        
        # Find all crossing points
        diff = np.diff(np.concatenate(([False], mean_low_deriv_mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]  # Low derivative starts
        ends = np.where(diff == -1)[0]    # Low derivative ends
        
        # Add all crossing points
        for start, end in zip(starts, ends):
            if start > 0:
                transitions.append(start)
            if end < len(mean_abs_derivative):
                transitions.append(end)
        
        # Check right boundary
        if mean_abs_derivative[-1] < 0.2 and len(mean_abs_derivative) not in transitions:
            transitions.append(len(mean_abs_derivative))
        
        # Sort transitions
        transitions = sorted(set(transitions))
        
        # Check each interval between consecutive transitions
        for i in range(len(transitions) - 1):
            interval_start = transitions[i]
            interval_end = transitions[i + 1]
            
            # Check if mean CD values in this interval are all < cd_threshold
            interval_mean_cd = mean_cd[interval_start:interval_end]
            if np.all(interval_mean_cd < cd_threshold):
                # Expand interval by one step on each side, clipped to valid range
                expanded_start = max(0, interval_start - 1)
                expanded_end = min(len(translation_factors) - 1, interval_end)
                
                # Save interval data for mean curve
                intervals_data['mean_intervals'].append({
                    't_start': float(translation_factors[expanded_start]),
                    't_end': float(translation_factors[expanded_end]),
                    'start_idx': int(expanded_start),
                    'end_idx': int(expanded_end),
                    'mean_cd': float(np.mean(mean_cd[expanded_start:expanded_end+1])),
                    'min_cd': float(np.min(mean_cd[expanded_start:expanded_end+1])),
                    'max_cd': float(np.max(mean_cd[expanded_start:expanded_end+1]))
                })
                
                if save_plots:
                    plt.axvspan(translation_factors[expanded_start], translation_factors[expanded_end], 
                               alpha=0.2, color='gray', zorder=0)
        
        # Merge overlapping intervals for mean curve
        merged_mean_intervals = merge_overlapping_intervals(intervals_data['mean_intervals'])
        
        # Remove intervals smaller than 0.02 after merging
        intervals_data['mean_intervals'] = [interval for interval in merged_mean_intervals 
                                            if (interval['t_end'] - interval['t_start']) >= 0.02]
        
        # Mark minimum on mean curve
        mean_min_idx = np.argmin(mean_cd)
        mean_min_t = translation_factors[mean_min_idx]
        mean_min_cd = mean_cd[mean_min_idx]

        # Find second minimum on mean curve
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(-mean_cd, prominence=0.001)
        valid_peaks = [p for p in peaks if p != mean_min_idx]
        if valid_peaks:
            valid_peaks_sorted = sorted(valid_peaks, key=lambda p: mean_cd[p])
            second_min_idx = valid_peaks_sorted[0]
            second_min_t = translation_factors[second_min_idx]
            second_min_cd = mean_cd[second_min_idx]
            print(f"  MEAN: Min CD={mean_min_cd:.6f} at t={mean_min_t:.2f}, 2nd min CD={second_min_cd:.6f} at t={second_min_t:.2f}")
        else:
            print(f"  MEAN: Min CD={mean_min_cd:.6f} at t={mean_min_t:.2f}")
    
    if save_plots:
        plt.xlabel('Translation Factor', fontsize=12)
        plt.ylabel('Chamfer Distance', fontsize=12)
        plt.title(f'Sketch {sketch_idx}: CD vs Translation along Normal ({len(loops_data)} loops)', fontsize=14)
        plt.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        output_path = f'{output_folder}/sketch_{sketch_idx}_cd_translation.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close('all')
        
        print(f"  Saved CD plot: {output_path}")
    
    # Plot absolute derivative
    if save_plots:
        plt.figure(figsize=(12, 7))
    
    if save_plots:
        for idx, (loop_idx, translation_factors, cd_values) in enumerate(loops_data):
            color = colors[idx % len(colors)]
            abs_derivative = np.abs(np.gradient(cd_values, translation_factors))
            plt.plot(translation_factors, abs_derivative, linewidth=2, color=color, 
                     label=f'Loop {loop_idx}', alpha=0.6)
            
            # Highlight derivative threshold
            plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Threshold (0.2)' if idx == 0 else '')
        
        # Plot mean derivative if multiple loops
        if len(loops_data) > 1:
            mean_cd = np.mean(all_cd_arrays, axis=0)
            mean_abs_derivative = np.abs(np.gradient(mean_cd, translation_factors))
            plt.plot(translation_factors, mean_abs_derivative, linewidth=3, color='black', 
                     linestyle='--', label='Mean |dCD/dt|', alpha=0.8, zorder=100)
        
        plt.xlabel('Translation Factor', fontsize=12)
        plt.ylabel('|dCD/dt|', fontsize=12)
        plt.title(f'Sketch {sketch_idx}: Absolute Derivative of CD ({len(loops_data)} loops)', fontsize=14)
        plt.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        derivative_path = f'{output_folder}/sketch_{sketch_idx}_derivative.png'
        plt.savefig(derivative_path, dpi=150, bbox_inches='tight')
        plt.close('all')
        
        print(f"  Saved derivative plot: {derivative_path}")
    
    # Compute additional interval statistics
    # When multiple loops exist, use only mean_intervals (more reliable);
    if len(loops_data) > 1 and intervals_data['mean_intervals']:
        all_intervals = list(intervals_data['mean_intervals'])
    else:
        all_intervals = []
        for loop_data in intervals_data['loop_intervals']:
            all_intervals.extend(loop_data['intervals'])
    
    if all_intervals:
        # Merge overlapping intervals
        merged_all_intervals = merge_overlapping_intervals(all_intervals)
        intervals_data['merged_intervals'] = merged_all_intervals
        
        # Find min and max across all merged intervals
        all_t_starts = [interval['t_start'] for interval in merged_all_intervals]
        all_t_ends = [interval['t_end'] for interval in merged_all_intervals]
        
        intervals_data['all_intervals_min'] = float(min(all_t_starts))
        intervals_data['all_intervals_max'] = float(max(all_t_ends))
        
        # Find interval containing zero
        for interval in merged_all_intervals:
            if interval['t_start'] <= 0 <= interval['t_end']:
                intervals_data['interval_containing_zero'] = {
                    't_min': interval['t_start'],
                    't_max': interval['t_end'],
                    'mean_cd': interval['mean_cd'],
                    'min_cd': interval['min_cd'],
                    'max_cd': interval['max_cd']
                }
                break
    
    # Save intervals data to JSON
    intervals_path = f'{output_folder}/sketch_{sketch_idx}_intervals.json'
    with open(intervals_path, 'w') as f:
        json.dump(intervals_data, f, indent=2)
    print(f"  Saved intervals: {intervals_path}")
    
    return intervals_data


def plot_loop_on_mesh_3d(loops_3d, normal, mesh, surface_points, sketch_idx, output_folder, save_plots=True):
    """
    Plot all loops from a sketch in 3D on top of normalized mesh and surface point cloud.
    
    Args:
        loops_3d: List of loops, where each loop is Nx3 array of points
        normal: 3D normal vector
        mesh: Trimesh object (normalized)
        surface_points: Mx3 array of surface sample points
        sketch_idx: Sketch index
        output_folder: Output folder for plots
        save_plots: Whether to save visualization plots
    """
    if not save_plots:
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Remove grid and background
    ax.grid(False)
    ax.set_axis_off()
    
    # Plot surface points (subsample for visualization)
    subsample_idx = np.random.choice(len(surface_points), min(10000, len(surface_points)), replace=False)
    surface_subsample = surface_points[subsample_idx]
    ax.scatter(surface_subsample[:, 0], surface_subsample[:, 1], surface_subsample[:, 2], 
               c='lightblue', s=1, alpha=0.3, label='Surface Points')
    
    # Plot mesh wireframe
    vertices = mesh.vertices
    faces = mesh.faces
    for face in faces[::10]:  # Subsample faces for clarity
        triangle = vertices[face]
        # Close the triangle
        triangle = np.vstack([triangle, triangle[0]])
        ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 'gray', linewidth=0.3, alpha=0.2)
    
    # Colors for different loops
    colors = ['blue', 'blue', 'orange', 'purple', 'green', 'cyan', 'magenta', 'yellow']
    
    # Plot all loops
    for loop_idx, loop_3d in enumerate(loops_3d):
        loop_array = np.array(loop_3d)
        color = colors[loop_idx % len(colors)]
        
        # Plot loop points
        ax.scatter(loop_array[:, 0], loop_array[:, 1], loop_array[:, 2], 
                   c=color, s=50, alpha=1.0, label=f'Loop {loop_idx}', edgecolors='darkblue', linewidths=2)
        
        # Plot loop as connected line
        loop_closed = np.vstack([loop_array, loop_array[0]])
        ax.plot(loop_closed[:, 0], loop_closed[:, 1], loop_closed[:, 2], 
                color=color, linewidth=2, alpha=0.8)
    
    # Plot normal vector from sketch centroid (mean of all loop points)
    all_loop_points = np.vstack([np.array(loop) for loop in loops_3d])
    centroid = np.mean(all_loop_points, axis=0)
    normal_normalized = np.array(normal) / np.linalg.norm(normal)
    normal_length = 0.3
    ax.quiver(centroid[0], centroid[1], centroid[2],
              normal_normalized[0], normal_normalized[1], normal_normalized[2],
              length=normal_length, color='darkgreen', arrow_length_ratio=0.3, linewidth=3, label='Normal')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'Sketch {sketch_idx}: 3D Visualization ({len(loops_3d)} loops)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    
    # Set equal aspect ratio based on mesh bounds (not loop)
    mesh_bounds = mesh.bounds
    max_range = np.array([mesh_bounds[1, 0] - mesh_bounds[0, 0],
                          mesh_bounds[1, 1] - mesh_bounds[0, 1],
                          mesh_bounds[1, 2] - mesh_bounds[0, 2]]).max() / 2.0
    mid_x = (mesh_bounds[1, 0] + mesh_bounds[0, 0]) * 0.5
    mid_y = (mesh_bounds[1, 1] + mesh_bounds[0, 1]) * 0.5
    mid_z = (mesh_bounds[1, 2] + mesh_bounds[0, 2]) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    output_path = f'{output_folder}/sketch_{sketch_idx}_3d_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    
    print(f"  Saved 3D visualization: {output_path}")


def process_single_sketch(args):
    """
    Worker function to process a single sketch.
    
    Args:
        args: Tuple of (sketch_idx, entry, surface_points, mesh_data, output_folder, translation_range, step, save_plots)
        
    Returns:
        Dictionary with sketch results
    """
    sketch_idx, entry, surface_points, mesh_data, output_folder, translation_range, step, save_plots = args
    
    loops_3d = entry.get('loops_3d', entry.get('loops', []))
    normal = entry.get('normal', [0, 0, 1])
    
    if not loops_3d:
        return {
            'sketch_idx': sketch_idx,
            'loops_cd_data': [],
            'results': [],
            'success': False
        }
    
    print(f"\nProcessing Sketch {sketch_idx} ({len(loops_3d)} loops)...")
    
    # Reconstruct mesh from data
    mesh = trimesh.Trimesh(vertices=mesh_data['vertices'], faces=mesh_data['faces'])
    
    # Plot 3D visualization once for all loops in this sketch
    if save_plots:
        plot_loop_on_mesh_3d(loops_3d, normal, mesh, surface_points, sketch_idx, output_folder, save_plots)
    
    # Collect CD data for all loops in this sketch
    loops_cd_data = []
    results = []
    
    for loop_idx, loop_3d in enumerate(loops_3d):
        if len(loop_3d) < 3:
            print(f"  Loop {loop_idx}: Too few points ({len(loop_3d)}), skipping")
            continue
        
        print(f"  Loop {loop_idx}: {len(loop_3d)} points", end=" - ")
        
        # Compute CD for translation range
        translation_factors, cd_values = compute_cd_for_translations(
            loop_3d, normal, surface_points,
            translation_range=translation_range,
            step=step,
        )

        # Store for combined plot
        loops_cd_data.append((loop_idx, translation_factors, cd_values))

        # Store results for summary
        min_idx = np.argmin(cd_values)
        results.append({
            'sketch_idx': sketch_idx,
            'loop_idx': loop_idx,
            'num_points': len(loop_3d),
            'min_cd': float(cd_values[min_idx]),
            'optimal_translation': float(translation_factors[min_idx]),
            'cd_at_zero': float(cd_values[0])
        })
    
    # Plot CD vs translation for all loops in this sketch (before returning)
    intervals_data = None
    if loops_cd_data:
        intervals_data = plot_cd_vs_translation_all_loops(loops_cd_data, sketch_idx, output_folder, save_plots)
    
    # Clean up memory
    del mesh
    import gc
    gc.collect()
    
    return {
        'sketch_idx': sketch_idx,
        'loops_cd_data': loops_cd_data,
        'results': results,
        'intervals': intervals_data,
        'success': True
    }


def analyze_all_loops(loops_json, stl_path, output_folder="loop_translation_analysis", normalize=False,
                      num_surface_samples=100000, translation_range=(0, 2), step=0.01,
                      num_workers=4, save_plots=True):
    """Analyze every loop in `loops_json` by sweeping it along its normal and
    computing one-sided chamfer distance to the target mesh surface."""
    print(f"Loading STL file: {stl_path}")
    mesh = trimesh.load(stl_path, force='mesh')

    if normalize:
        print("Normalizing mesh...")
        mesh = normalize_mesh(mesh)

    print(f"Sampling {num_surface_samples} points on surface...")
    surface_points = sample_surface_points(mesh, num_surface_samples)
    print(f"  Sampled {len(surface_points)} surface points")
    
    print(f"Loading loops from: {loops_json}")
    with open(loops_json, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} sketches")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare mesh data for serialization
    mesh_data = {
        'vertices': mesh.vertices,
        'faces': mesh.faces
    }
    
    # Store results for summary
    all_results = []
    
    if num_workers > 1:
        print(f"\nProcessing sketches in parallel with {num_workers} workers...")

        # Prepare arguments for parallel processing
        worker_args = []
        for sketch_idx, entry in enumerate(data):
            loops_3d = entry.get('loops_3d', entry.get('loops', []))
            if not loops_3d:
                continue
            worker_args.append((sketch_idx, entry, surface_points, mesh_data, output_folder,
                              translation_range, step, save_plots))
        
        # Process sketches in parallel using imap for chunked processing
        # This processes num_workers sketches at a time, starting a new one when one completes
        # Plots are generated inside the worker to ensure completion before new sketch starts
        all_intervals = []
        with mp.Pool(num_workers) as pool:
            sketch_results = []
            for i, result in enumerate(pool.imap(process_single_sketch, worker_args)):
                sketch_results.append(result)
                if result['success']:
                    all_results.extend(result['results'])
                    if result.get('intervals'):
                        all_intervals.append(result['intervals'])
                print(f"Completed {i+1}/{len(worker_args)} sketches")
        
        # Save combined intervals data
        combined_intervals_path = f'{output_folder}/all_intervals.json'
        with open(combined_intervals_path, 'w') as f:
            json.dump(all_intervals, f, indent=2)
        print(f"\nSaved combined intervals to: {combined_intervals_path}")
    
    else:
        # Sequential processing (original code)
        print(f"\nProcessing sketches sequentially...")
        
        for sketch_idx, entry in enumerate(data):
            loops_3d = entry.get('loops_3d', entry.get('loops', []))
            normal = entry.get('normal', [0, 0, 1])
            
            if not loops_3d:
                print(f"Sketch {sketch_idx}: No loops found, skipping")
                continue
            
            print(f"\nProcessing Sketch {sketch_idx} ({len(loops_3d)} loops)...")
            
            # Plot 3D visualization once for all loops in this sketch
            if save_plots:
                plot_loop_on_mesh_3d(loops_3d, normal, mesh, surface_points, sketch_idx, output_folder)
            
            # Collect CD data for all loops in this sketch
            loops_cd_data = []
            
            for loop_idx, loop_3d in enumerate(loops_3d):
                if len(loop_3d) < 3:
                    print(f"  Loop {loop_idx}: Too few points ({len(loop_3d)}), skipping")
                    continue
                
                print(f"  Loop {loop_idx}: {len(loop_3d)} points", end=" - ")
                
                # Compute CD for translation range
                translation_factors, cd_values = compute_cd_for_translations(
                    loop_3d, normal, surface_points,
                    translation_range=translation_range,
                    step=step,
                )
                
                # Store for combined plot
                loops_cd_data.append((loop_idx, translation_factors, cd_values))
                
                # Store results for summary
                min_idx = np.argmin(cd_values)
                all_results.append({
                    'sketch_idx': sketch_idx,
                    'loop_idx': loop_idx,
                    'num_points': len(loop_3d),
                    'min_cd': float(cd_values[min_idx]),
                    'optimal_translation': float(translation_factors[min_idx]),
                    'cd_at_zero': float(cd_values[0])
                })
            
            # Plot CD vs translation for all loops in this sketch
            if loops_cd_data:
                plot_cd_vs_translation_all_loops(loops_cd_data, sketch_idx, output_folder, save_plots)
    
    # Save summary results
    summary_path = f'{output_folder}/translation_analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, indent=2, fp=f)
    print(f"\nSaved summary results to: {summary_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if all_results:
        min_cds = [r['min_cd'] for r in all_results]
        optimal_ts = [r['optimal_translation'] for r in all_results]
        cd_at_zeros = [r['cd_at_zero'] for r in all_results]
        
        print(f"Total loops analyzed: {len(all_results)}")
        print(f"\nMinimum CD statistics:")
        print(f"  Mean: {np.mean(min_cds):.6f}")
        print(f"  Median: {np.median(min_cds):.6f}")
        print(f"  Min: {np.min(min_cds):.6f}")
        print(f"  Max: {np.max(min_cds):.6f}")
        
        print(f"\nOptimal translation factor statistics:")
        print(f"  Mean: {np.mean(optimal_ts):.3f}")
        print(f"  Median: {np.median(optimal_ts):.3f}")
        print(f"  Min: {np.min(optimal_ts):.3f}")
        print(f"  Max: {np.max(optimal_ts):.3f}")
        
        print(f"\nCD at zero translation:")
        print(f"  Mean: {np.mean(cd_at_zeros):.6f}")
        print(f"  Median: {np.median(cd_at_zeros):.6f}")
        
        # Plot overall distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(optimal_ts, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Optimal Translation Factor', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Optimal Translation Factors', fontsize=14)
        
        axes[1].hist(min_cds, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Minimum CD', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Distribution of Minimum CD Values', fontsize=14)
        
        plt.tight_layout()
        dist_plot_path = f'{output_folder}/overall_distributions.png'
        plt.savefig(dist_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved overall distribution plot: {dist_plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze loop translation along normal by computing Chamfer Distance')
    parser.add_argument('loops_json', help='Path to loops_split.json file')
    parser.add_argument('stl_path', help='Path to STL file')
    parser.add_argument('--output-folder', default='loop_translation_analysis', help='Output folder for results')
    parser.add_argument('--num-surface-samples', type=int, default=100000, help='Number of surface points to sample')
    parser.add_argument('--t-min', type=float, default=-2.0, help='Minimum translation factor')
    parser.add_argument('--t-max', type=float, default=2.0, help='Maximum translation factor')
    parser.add_argument('--step', type=float, default=0.01, help='Translation step size')
    parser.add_argument('--num-workers', type=int, default=25, help='Number of parallel workers')
    parser.add_argument('--normalize', action='store_true', help='Normalize mesh before analysis')
    parser.add_argument('--no-plots', action='store_true', help='Disable saving visualization plots')

    args = parser.parse_args()

    analyze_all_loops(
        args.loops_json,
        args.stl_path,
        output_folder=args.output_folder,
        num_surface_samples=args.num_surface_samples,
        translation_range=(args.t_min, args.t_max),
        step=args.step,
        num_workers=args.num_workers,
        normalize=args.normalize,
        save_plots=not args.no_plots,
    )
