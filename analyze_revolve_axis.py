#!/usr/bin/env python3
"""
Analyze revolve axis optimization by computing Chamfer Distance for rotated sketches.

This script:
1. Loads and normalizes an STL file
2. Samples points on the surface
3. Loads loops from loops_split.json
4. Finds symmetry axes for each sketch (same method as cadquery_ops)
5. Rotates each sketch around its symmetry axis (0° to 360°, step size configurable)
6. Computes CD between rotated loop points and surface points
7. Plots CD vs rotation angle for each sketch
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
import multiprocessing as mp
from functools import partial


def chamfer_distance(points_a, points_b, average=True):
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
    if average:
        return np.mean(dists)
    else:
        return dists


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


def project_3d_to_2d(points_3d, origin, normal, xdir):
    """
    Project 3D points onto a 2D plane coordinate system.
    
    Args:
        points_3d: Nx3 array of 3D points
        origin: Origin of the plane
        normal: Normal vector of the plane
        xdir: X-direction vector of the plane
        
    Returns:
        Nx2 array of 2D points
    """
    points_3d = np.array(points_3d)
    origin = np.array(origin)
    normal = np.array(normal)
    xdir = np.array(xdir)
    
    # Normalize vectors
    normal = normal / np.linalg.norm(normal)
    xdir = xdir / np.linalg.norm(xdir)
    
    # Compute y-direction (perpendicular to both normal and xdir)
    ydir = np.cross(normal, xdir)
    ydir = ydir / np.linalg.norm(ydir)
    
    # Project points onto plane
    points_2d = []
    for point in points_3d:
        vec = point - origin
        x = np.dot(vec, xdir)
        y = np.dot(vec, ydir)
        points_2d.append([x, y])
    
    return np.array(points_2d)


def check_symmetry_axis(points_2d, axis_angle_deg, tolerance=0.03):
    """
    Check if 2D points are symmetric about an axis through their centroid.
    
    Args:
        points_2d: Nx2 array of 2D points
        axis_angle_deg: Angle of axis in degrees (0 = x-axis)
        tolerance: Distance tolerance for symmetry (as fraction of point cloud size)
        
    Returns:
        (is_symmetric, axis_unit_vector, symmetry_ratio)
    """
    if len(points_2d) < 4:
        return False, None, 0.0
    
    # Compute centroid
    centroid = np.mean(points_2d, axis=0)
    centered = points_2d - centroid

    # Check if points form a circle - skip symmetry check for circles
    distances_from_centroid = np.linalg.norm(centered, axis=1)

    if np.allclose(distances_from_centroid, distances_from_centroid[0], rtol=0.01):
        # Circle detected - symmetric about any axis through centroid
        return True, None, 1.0
    
    # Axis direction
    axis_rad = np.radians(axis_angle_deg)
    axis_dir = np.array([np.cos(axis_rad), np.sin(axis_rad)])
    
    # Perpendicular to axis
    perp_dir = np.array([-np.sin(axis_rad), np.cos(axis_rad)])
    
    # Project points onto perpendicular direction (distance from axis)
    perp_distances = centered @ perp_dir
    
    # Flip points across axis
    flipped_centered = centered - 2 * np.outer(perp_distances, perp_dir)
    flipped = flipped_centered + centroid
    
    # Check if flipped points match original points
    min_distances = chamfer_distance(flipped, points_2d, average=False)
    
    # Compute characteristic size
    bbox_size = np.max(np.ptp(points_2d, axis=0))
    threshold = tolerance * bbox_size
    
    # Check if most points have a close match
    symmetric_points = np.sum(min_distances < threshold)
    symmetry_ratio = symmetric_points / len(points_2d)
    
    is_symmetric = symmetry_ratio > 0.95
    
    return is_symmetric, axis_dir, symmetry_ratio


def find_symmetry_axes(entry):
    """
    Find axes of symmetry for a sketch by testing multiple axes.
    Returns ALL symmetric axes found.
    
    Args:
        entry: Sketch entry with loops_3d or loops
        
    Returns:
        List of dicts with 'angle', 'ratio', 'axis_3d_start', 'axis_3d_end' for each symmetric axis,
        or empty list if no symmetry found
    """
    # Get 3D loops
    loops_3d = entry.get('loops_3d', entry.get('loops', []))
    
    if not loops_3d:
        return []
    
    # Get plane information
    origin = entry.get('origin', [0, 0, 0])
    normal = entry.get('normal', [0, 0, 1])
    xdir = entry.get('xdir', [1, 0, 0])
    
    # Project 3D loops to 2D
    all_points_2d = []
    for loop_3d in loops_3d:
        loop_2d = project_3d_to_2d(loop_3d, origin, normal, xdir)
        all_points_2d.extend(loop_2d)
    
    if len(all_points_2d) < 4:
        return []
    
    all_points_2d = np.array(all_points_2d)
    
    # Remove duplicates
    unique_points = np.unique(all_points_2d, axis=0)
    
    # Compute principal axes using PCA
    centered_points = unique_points - np.mean(unique_points, axis=0)
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalue (descending)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    principal_axes = eigenvectors[:, sorted_indices]
    
    # Convert principal axes to angles (in degrees)
    pca_angles = []
    for i in range(2):
        angle = np.degrees(np.arctan2(principal_axes[1, i], principal_axes[0, i]))
        angle = angle % 180
        pca_angles.append(angle)
    
    # Test fixed axes plus PCA axes
    test_angles = [0, 45, 90, 135] + pca_angles
    symmetric_axes = []
    symmetric_axes_2d = []
    
    for angle in test_angles:
        is_sym, axis_dir, ratio = check_symmetry_axis(unique_points, angle)
        if is_sym:
            symmetric_axes_2d.append({'angle': angle, 'ratio': ratio, 'axis_dir': axis_dir})
    
    if not symmetric_axes_2d:
        return []
    
    # Convert all symmetric axes to 3D
    origin_arr = np.array(origin)
    normal_arr = np.array(normal)
    xdir_arr = np.array(xdir)
    
    # Normalize vectors
    normal_arr = normal_arr / np.linalg.norm(normal_arr)
    xdir_arr = xdir_arr / np.linalg.norm(xdir_arr)
    
    # Compute y-direction (perpendicular to both normal and xdir)
    ydir_arr = np.cross(normal_arr, xdir_arr)
    ydir_arr = ydir_arr / np.linalg.norm(ydir_arr)
    
    # Centroid in 2D
    centroid_2d = np.mean(unique_points, axis=0)
    
    # Centroid in 3D (fixed: swap xdir and ydir components)
    centroid_3d = origin_arr + centroid_2d[0] * ydir_arr + centroid_2d[1] * xdir_arr
    
    # Convert each symmetric axis to 3D
    for sym_axis in symmetric_axes_2d:
        angle = sym_axis['angle']
        axis_dir_2d = sym_axis['axis_dir']
        
        if axis_dir_2d is None:
            # Circle - use any axis (e.g., x-axis)
            axis_dir_2d = np.array([1, 0])
        
        # Convert 2D axis direction to 3D
        # axis_dir_2d is in (xdir, ydir) coordinates
        axis_dir_3d = axis_dir_2d[0] * ydir_arr + axis_dir_2d[1] * xdir_arr
        axis_dir_3d = axis_dir_3d / np.linalg.norm(axis_dir_3d)
        
        # Axis passes through centroid
        axis_start = centroid_3d
        axis_end = centroid_3d + axis_dir_3d
        
        symmetric_axes.append({
            'angle': angle,
            'ratio': sym_axis['ratio'],
            'axis_3d_start': axis_start.tolist(),
            'axis_3d_end': axis_end.tolist(),
            'axis_direction': axis_dir_3d.tolist()
        })
    
    return symmetric_axes


def rotate_points_around_axis(points, axis_start, axis_direction, angle_degrees):
    """
    Rotate 3D points around an arbitrary axis using Rodrigues' rotation formula.
    
    Args:
        points: Nx3 array of 3D points
        axis_start: 3D point on the axis
        axis_direction: 3D unit vector defining axis direction
        angle_degrees: Rotation angle in degrees
        
    Returns:
        Rotated Nx3 array of points
    """
    points = np.array(points)
    axis_start = np.mean(points, axis=0)
    axis_direction = np.array(axis_direction)
    
    # Normalize axis direction
    axis_direction = axis_direction / np.linalg.norm(axis_direction)
    
    # Convert angle to radians
    theta = np.radians(angle_degrees)
    
    # Translate points so axis passes through origin
    translated = points - axis_start
    
    # Rodrigues' rotation formula
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # For each point
    rotated = []
    for p in translated:
        # where k is the unit axis direction
        k_dot_p = np.dot(axis_direction, p)
        k_cross_p = np.cross(axis_direction, p)
        
        p_rot = (p * cos_theta + 
                 k_cross_p * sin_theta + 
                 axis_direction * k_dot_p * (1 - cos_theta))
        rotated.append(p_rot)
    
    rotated = np.array(rotated)
    
    # Translate back
    rotated = rotated + axis_start
    
    return rotated


def compute_cd_for_rotations(loops_3d, axis_start, axis_direction, surface_points,
                              angle_range=(0, 360), step=1.0):
    """Compute one-sided Chamfer Distance to surface_points for a sweep of
    rotation angles around (axis_start, axis_direction).

    Builds the surface KD-tree once per call (otherwise ``chamfer_distance``
    rebuilds it on every angle, which dominated runtime for dense meshes)."""
    angle_min, angle_max = angle_range
    rotation_angles = np.arange(angle_min, angle_max + step, step)
    all_loop_points = np.vstack([np.array(loop) for loop in loops_3d])
    tree = cKDTree(surface_points)
    cd_values = np.empty(rotation_angles.shape[0])
    for i, angle in enumerate(rotation_angles):
        rotated_points = rotate_points_around_axis(all_loop_points, axis_start, axis_direction, angle)
        dists, _ = tree.query(rotated_points, k=1)
        cd_values[i] = dists.mean()
    return rotation_angles, cd_values


def merge_overlapping_intervals(intervals):
    """
    Merge overlapping or adjacent intervals.
    
    Args:
        intervals: List of interval dictionaries with angle_start, angle_end, etc.
        
    Returns:
        List of merged intervals
    """
    if not intervals:
        return []
    
    # Sort intervals by start angle
    sorted_intervals = sorted(intervals, key=lambda x: x['angle_start'])
    
    merged = [sorted_intervals[0].copy()]
    
    for current in sorted_intervals[1:]:
        last_merged = merged[-1]
        
        # Check if intervals overlap or are adjacent
        if current['angle_start'] <= last_merged['angle_end']:
            # Merge: extend the end and update statistics
            last_merged['angle_end'] = max(last_merged['angle_end'], current['angle_end'])
            last_merged['end_idx'] = max(last_merged['end_idx'], current['end_idx'])
            
            # Update CD statistics
            last_merged['mean_cd'] = (last_merged['mean_cd'] + current['mean_cd']) / 2.0
            last_merged['min_cd'] = min(last_merged['min_cd'], current['min_cd'])
            last_merged['max_cd'] = max(last_merged['max_cd'], current['max_cd'])
        else:
            # No overlap, add as new interval
            merged.append(current.copy())
    
    return merged


def plot_loop_on_mesh_3d(loops_3d, axis_start, axis_direction, mesh, surface_points, sketch_idx, axis_idx, output_folder, rotation_angle=None, save_plots=True):
    """
    Plot loops in 3D on top of normalized mesh and surface point cloud with symmetry axis.
    
    Args:
        loops_3d: List of loops (each loop is Nx3 array of points)
        axis_start: 3D point on the axis
        axis_direction: 3D unit vector defining axis direction
        mesh: Trimesh object (normalized)
        surface_points: Mx3 array of surface sample points
        sketch_idx: Sketch index
        axis_idx: Symmetry axis index or string (e.g., "0_ortho")
        output_folder: Output folder for plots
        rotation_angle: Optional rotation angle to show rotated points
        save_plots: Whether to save visualization plots
    """
    if not save_plots:
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
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
        triangle = np.vstack([triangle, triangle[0]])
        ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 'gray', linewidth=0.3, alpha=0.2)
    
    # Colors for different loops
    colors = ['red', 'blue', 'orange', 'purple', 'green', 'cyan', 'magenta', 'yellow']
    
    # Plot all loops (original position)
    for loop_idx, loop_3d in enumerate(loops_3d):
        loop_array = np.array(loop_3d)
        color = colors[loop_idx % len(colors)]
        
        # Plot loop points
        ax.scatter(loop_array[:, 0], loop_array[:, 1], loop_array[:, 2], 
                   c=color, s=50, alpha=1.0, label=f'Loop {loop_idx}', edgecolors='darkred', linewidths=2)
        
        # Plot loop as connected line
        loop_closed = np.vstack([loop_array, loop_array[0]])
        ax.plot(loop_closed[:, 0], loop_closed[:, 1], loop_closed[:, 2], 
                color=color, linewidth=2, alpha=0.8)
    
    # Plot rotated loops if rotation_angle is provided
    if rotation_angle is not None:
        all_loop_points = np.vstack([np.array(loop) for loop in loops_3d])
        rotated_points = rotate_points_around_axis(all_loop_points, axis_start, axis_direction, rotation_angle)
        
        # Plot rotated points in a different style
        ax.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2],
                   c='lime', s=30, alpha=0.7, marker='o', label=f'Rotated {rotation_angle:.0f}°', 
                   edgecolors='darkgreen', linewidths=1.5)
    
    # Plot symmetry axis
    axis_start_arr = np.array(axis_start)
    axis_direction_arr = np.array(axis_direction)
    axis_direction_arr = axis_direction_arr / np.linalg.norm(axis_direction_arr)
    axis_length = 0.3
    axis_end_arr = axis_start_arr + axis_direction_arr * axis_length
    
    ax.quiver(axis_start_arr[0], axis_start_arr[1], axis_start_arr[2],
              axis_direction_arr[0], axis_direction_arr[1], axis_direction_arr[2],
              length=axis_length, color='darkgreen', arrow_length_ratio=0.3, linewidth=3, label='Rotation Axis')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    title = f'Sketch {sketch_idx}, Axis {axis_idx}: 3D Visualization'
    if rotation_angle is not None:
        title += f' (Rotated {rotation_angle:.0f}°)'
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    
    # Set equal aspect ratio based on mesh bounds
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
    if rotation_angle is not None:
        output_path = f'{output_folder}/sketch_{sketch_idx}_axis_{axis_idx}_3d_rotated_{rotation_angle:.0f}.png'
    else:
        output_path = f'{output_folder}/sketch_{sketch_idx}_axis_{axis_idx}_3d_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    
    print(f"  Saved 3D visualization: {output_path}")


def plot_cd_vs_rotation(sketch_idx, axis_idx, rotation_angles, cd_values, output_folder, save_plots=True, cd_threshold=0.01):
    """
    Plot Chamfer Distance vs rotation angle for a sketch and symmetry axis.
    
    Args:
        sketch_idx: Sketch index
        axis_idx: Symmetry axis index
        rotation_angles: Array of rotation angles
        cd_values: Array of CD values
        output_folder: Output folder for plots
        save_plots: Whether to save visualization plots
        cd_threshold: CD threshold for identifying low-CD intervals
        
    Returns:
        Dictionary with interval data
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Store interval data
    intervals_data = {
        'sketch_idx': sketch_idx,
        'axis_idx': axis_idx,
        'intervals': [],
        'interval_containing_zero': None,
        'all_intervals_min': None,
        'all_intervals_max': None
    }
    
    if save_plots:
        plt.figure(figsize=(12, 7))
        
        plt.plot(rotation_angles, cd_values, linewidth=2, color='blue', alpha=0.7)
    
    # Compute absolute derivative
    abs_derivative = np.abs(np.gradient(cd_values, rotation_angles))
    
    # Find intervals where derivative is low and CD is low
    low_deriv_mask = abs_derivative < 0.2
    
    # Find all transition points
    transitions = []
    
    # Check left boundary
    if abs_derivative[0] < 0.2:
        transitions.append(0)
    
    # Find crossing points
    diff = np.diff(np.concatenate(([False], low_deriv_mask, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for start, end in zip(starts, ends):
        if start > 0:
            transitions.append(start)
        if end < len(abs_derivative):
            transitions.append(end)
    
    # Check right boundary
    if abs_derivative[-1] < 0.2 and len(abs_derivative) not in transitions:
        transitions.append(len(abs_derivative))
    
    transitions = sorted(set(transitions))
    
    # Check each interval between consecutive transitions
    for i in range(len(transitions) - 1):
        interval_start = transitions[i]
        interval_end = transitions[i + 1]
        
        # Skip single-point intervals
        if interval_end - interval_start <= 1:
            continue
        
        # Check if CD values in this interval are all below threshold
        interval_cd_values = cd_values[interval_start:interval_end]
        if np.all(interval_cd_values < cd_threshold):
            # Expand interval by one step on each side
            expanded_start = max(0, interval_start - 1)
            expanded_end = min(len(rotation_angles) - 1, interval_end)
            
            # Save interval data
            intervals_data['intervals'].append({
                'angle_start': float(rotation_angles[expanded_start]),
                'angle_end': float(rotation_angles[expanded_end]),
                'start_idx': int(expanded_start),
                'end_idx': int(expanded_end),
                'mean_cd': float(np.mean(cd_values[expanded_start:expanded_end+1])),
                'min_cd': float(np.min(cd_values[expanded_start:expanded_end+1])),
                'max_cd': float(np.max(cd_values[expanded_start:expanded_end+1]))
            })
            
            if save_plots:
                plt.axvspan(rotation_angles[expanded_start], rotation_angles[expanded_end], 
                           alpha=0.2, color='green', zorder=0)
    
    # Merge overlapping intervals
    intervals_data['intervals'] = merge_overlapping_intervals(intervals_data['intervals'])
    
    # Mark minimum CD
    min_idx = np.argmin(cd_values)
    min_angle = rotation_angles[min_idx]
    min_cd = cd_values[min_idx]
    if save_plots:
        plt.plot(min_angle, min_cd, '*', color='red', markersize=12, 
                 markeredgecolor='black', markeredgewidth=1)
        
        plt.xlabel('Rotation Angle (degrees)', fontsize=12)
        plt.ylabel('Chamfer Distance', fontsize=12)
        plt.title(f'Sketch {sketch_idx}, Axis {axis_idx}: CD vs Rotation Angle', fontsize=14)
        
        plt.tight_layout()
        output_path = f'{output_folder}/sketch_{sketch_idx}_axis_{axis_idx}_cd_rotation.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close('all')
        
        print(f"  Saved CD plot: {output_path}")
        
        # Plot absolute derivative
        plt.figure(figsize=(12, 7))
        plt.plot(rotation_angles, abs_derivative, linewidth=2, color='blue', alpha=0.7)
        plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Threshold (0.2)')
        
        plt.xlabel('Rotation Angle (degrees)', fontsize=12)
        plt.ylabel('|dCD/dθ|', fontsize=12)
        plt.title(f'Sketch {sketch_idx}, Axis {axis_idx}: Absolute Derivative of CD', fontsize=14)
        plt.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        derivative_path = f'{output_folder}/sketch_{sketch_idx}_axis_{axis_idx}_derivative.png'
        plt.close('all')
        
        print(f"  Saved derivative plot: {derivative_path}")
    
    # Compute additional interval statistics
    if intervals_data['intervals']:
        all_angle_starts = [interval['angle_start'] for interval in intervals_data['intervals']]
        all_angle_ends = [interval['angle_end'] for interval in intervals_data['intervals']]
        
        intervals_data['all_intervals_min'] = float(min(all_angle_starts))
        intervals_data['all_intervals_max'] = float(max(all_angle_ends))
        
        # Find interval containing zero
        for interval in intervals_data['intervals']:
            if interval['angle_start'] <= 0 <= interval['angle_end']:
                intervals_data['interval_containing_zero'] = {
                    'angle_min': interval['angle_start'],
                    'angle_max': interval['angle_end'],
                    'mean_cd': interval['mean_cd'],
                    'min_cd': interval['min_cd'],
                    'max_cd': interval['max_cd']
                }
                break
    
    # Save intervals data to JSON
    intervals_path = f'{output_folder}/sketch_{sketch_idx}_axis_{axis_idx}_intervals.json'
    with open(intervals_path, 'w') as f:
        json.dump(intervals_data, f, indent=2)
    print(f"  Saved intervals: {intervals_path}")
    
    return intervals_data


def process_single_sketch(args):
    """
    Worker function to process a single sketch.
    
    Args:
        args: Tuple of (sketch_idx, entry, surface_points, mesh_data, output_folder, angle_range, step, save_plots, cd_threshold)
        
    Returns:
        Dictionary with sketch results
    """
    try:
        sketch_idx, entry, surface_points, mesh_data, output_folder, angle_range, step, save_plots, cd_threshold = args
        
        loops_3d = entry.get('loops_3d', entry.get('loops', []))
        
        if not loops_3d:
            return {
                'sketch_idx': sketch_idx,
                'symmetric_axes': [],
                'results': [],
                'success': False
            }
        
        print(f"\nProcessing Sketch {sketch_idx} ({len(loops_3d)} loops)...")
        
        # Reconstruct mesh from data
        mesh = trimesh.Trimesh(vertices=mesh_data['vertices'], faces=mesh_data['faces'])
        
        # Find symmetry axes
        symmetric_axes = find_symmetry_axes(entry)
        
        if not symmetric_axes:
            print(f"  No symmetry axes found for sketch {sketch_idx}")
            return {
                'sketch_idx': sketch_idx,
                'symmetric_axes': [],
                'results': [],
                'success': False
            }
        
        print(f"  Found {len(symmetric_axes)} symmetry axes")
        
        results = []
        all_intervals = []
        
        for axis_idx, sym_axis in enumerate(symmetric_axes):
            axis_start = np.array(sym_axis['axis_3d_start'])
            axis_direction = np.array(sym_axis['axis_direction'])
            
            # Plot 3D visualization
            plot_loop_on_mesh_3d(loops_3d, axis_start, axis_direction, mesh, surface_points, 
                                sketch_idx, axis_idx, output_folder, save_plots=save_plots)
            
            print(f"  Axis {axis_idx}: angle={sym_axis['angle']:.1f}°, ratio={sym_axis['ratio']:.3f}", end=" - ")
            
            # Compute CD for rotation range around symmetry axis
            rotation_angles, cd_values = compute_cd_for_rotations(
                loops_3d, axis_start, axis_direction, surface_points,
                angle_range=angle_range,
                step=step,
            )
            
            # Check if all CD values are under threshold
            all_under_threshold = np.all(cd_values < cd_threshold)
            
            if all_under_threshold:
                # Compute mean point of all loops
                all_loop_points = np.vstack([np.array(loop) for loop in loops_3d])
                mean_point = np.mean(all_loop_points, axis=0)
                
                # Save simplified JSON
                simplified_data = {
                    'sketch_idx': sketch_idx,
                    'axis_idx': axis_idx,
                    'rotation_type': 'symmetry_axis',
                    'all_cd_under_threshold': True,
                    'cd_threshold': cd_threshold,
                    'mean_point': mean_point.tolist(),
                    'axis_direction': axis_direction.tolist()
                }
                
                simplified_path = f'{output_folder}/sketch_{sketch_idx}_axis_{axis_idx}_simplified.json'
                with open(simplified_path, 'w') as f:
                    json.dump(simplified_data, f, indent=2)
                print(f"All CD < {cd_threshold}, saved simplified: {simplified_path}")
            else:
                # Plot and save results
                intervals_data = plot_cd_vs_rotation(sketch_idx, axis_idx, rotation_angles, cd_values, output_folder, save_plots, cd_threshold)
                
                # Store results
                min_idx = np.argmin(cd_values)
                optimal_angle = rotation_angles[min_idx]
                
                # Plot 3D visualization with rotated points at optimal angle
                if optimal_angle != 0:
                    plot_loop_on_mesh_3d(loops_3d, axis_start, axis_direction, mesh, surface_points, 
                                        sketch_idx, axis_idx, output_folder, rotation_angle=90, save_plots=save_plots)
                
                all_intervals.append(intervals_data)
            
            results.append({
                'sketch_idx': sketch_idx,
                'axis_idx': axis_idx,
                'rotation_type': 'symmetry_axis',
                'axis_angle_2d': sym_axis['angle'],
                'symmetry_ratio': sym_axis['ratio'],
                'min_cd': float(cd_values[np.argmin(cd_values)]),
                'optimal_rotation': float(rotation_angles[np.argmin(cd_values)]),
                'cd_at_zero': float(cd_values[0]),
                'all_cd_under_threshold': bool(all_under_threshold),
                'cd_threshold': cd_threshold
            })
            
            # Now test rotation around orthogonal axis
            # Get sketch normal as orthogonal axis
            origin = entry.get('origin', [0, 0, 0])
            normal = entry.get('normal', [0, 0, 1])
            normal_arr = np.array(normal)
            normal_arr = normal_arr / np.linalg.norm(normal_arr)
            
            # Orthogonal axis passes through the same centroid but points along sketch normal
            ortho_axis_start = axis_start
            ortho_axis_direction = np.cross(normal_arr, axis_direction)
            
            print(f"\n  Axis {axis_idx} (orthogonal): testing rotation around sketch normal", end=" - ")
            
            # Compute CD for rotation range around orthogonal axis
            rotation_angles_ortho, cd_values_ortho = compute_cd_for_rotations(
                loops_3d, ortho_axis_start, ortho_axis_direction, surface_points,
                angle_range=angle_range,
                step=step,
            )
            
            # Check if all orthogonal CD values are under threshold
            all_under_threshold_ortho = np.all(cd_values_ortho < cd_threshold)
            
            if all_under_threshold_ortho:
                # Compute mean point of all loops
                all_loop_points = np.vstack([np.array(loop) for loop in loops_3d])
                mean_point = np.mean(all_loop_points, axis=0)
                
                # Save simplified JSON
                simplified_data_ortho = {
                    'sketch_idx': sketch_idx,
                    'axis_idx': f"{axis_idx}_ortho",
                    'rotation_type': 'orthogonal_to_symmetry',
                    'all_cd_under_threshold': True,
                    'cd_threshold': cd_threshold,
                    'mean_point': mean_point.tolist(),
                    'axis_direction': ortho_axis_direction.tolist()
                }
                
                simplified_path_ortho = f'{output_folder}/sketch_{sketch_idx}_axis_{axis_idx}_ortho_simplified.json'
                with open(simplified_path_ortho, 'w') as f:
                    json.dump(simplified_data_ortho, f, indent=2)
                print(f"All orthogonal CD < {cd_threshold}, saved simplified: {simplified_path_ortho}")
            else:
                # Plot and save results for orthogonal rotation
                intervals_data_ortho = plot_cd_vs_rotation(sketch_idx, f"{axis_idx}_ortho", 
                                                        rotation_angles_ortho, cd_values_ortho, output_folder, save_plots, cd_threshold)
                
                # Store results for orthogonal rotation
                min_idx_ortho = np.argmin(cd_values_ortho)
                optimal_angle_ortho = rotation_angles_ortho[min_idx_ortho]
                
                # Plot 3D visualization with rotated points at optimal angle for orthogonal rotation
                if optimal_angle_ortho != 0:
                    plot_loop_on_mesh_3d(loops_3d, ortho_axis_start, ortho_axis_direction, mesh, surface_points, 
                                        sketch_idx, f"{axis_idx}_ortho", output_folder, rotation_angle=90, save_plots=save_plots)
                
                all_intervals.append(intervals_data_ortho)
            
            results.append({
                'sketch_idx': sketch_idx,
                'axis_idx': axis_idx,
                'rotation_type': 'orthogonal_to_symmetry',
                'axis_angle_2d': sym_axis['angle'],
                'symmetry_ratio': sym_axis['ratio'],
                'min_cd': float(cd_values_ortho[np.argmin(cd_values_ortho)]),
                'optimal_rotation': float(rotation_angles_ortho[np.argmin(cd_values_ortho)]),
                'cd_at_zero': float(cd_values_ortho[0]),
                'all_cd_under_threshold': bool(all_under_threshold_ortho),
                'cd_threshold': cd_threshold
            })
        
        # Clean up memory
        import gc
        gc.collect()
        
        return {
            'sketch_idx': sketch_idx,
            'symmetric_axes': symmetric_axes,
            'results': results,
            'intervals': all_intervals,
            'success': True
        }
    except Exception as e:
        print(f"Error processing sketch {sketch_idx}: {e}")
        return {
            'sketch_idx': sketch_idx,
            'symmetric_axes': [],
            'results': [],
            'success': False
        }


def analyze_all_revolve_axes(loops_json, stl_path, output_folder="revolve_axis_analysis",
                              num_surface_samples=100000, angle_range=(0, 360), step=1.0,
                              num_workers=1, normalize=False, save_plots=True, cd_threshold=0.01):
    """Analyze all sketches in `loops_json` for candidate revolve axes by
    measuring chamfer distance of rotated loop points against the surface."""
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
    all_intervals = []
    
    if num_workers > 1:
        # Parallel processing
        print(f"\nProcessing sketches in parallel with {num_workers} workers...")
        
        worker_args = []
        for sketch_idx, entry in enumerate(data):
            loops_3d = entry.get('loops_3d', entry.get('loops', []))
            if not loops_3d:
                continue
            worker_args.append((sketch_idx, entry, surface_points, mesh_data, output_folder,
                              angle_range, step, save_plots, cd_threshold))
        
        with mp.Pool(num_workers) as pool:
            sketch_results = []
            for i, result in enumerate(pool.imap(process_single_sketch, worker_args)):
                sketch_results.append(result)
                if result['success']:
                    all_results.extend(result['results'])
                    if result.get('intervals'):
                        all_intervals.extend(result['intervals'])
                print(f"Completed {i+1}/{len(worker_args)} sketches")
        
        # Save combined intervals data
        combined_intervals_path = f'{output_folder}/all_intervals.json'
        with open(combined_intervals_path, 'w') as f:
            json.dump(all_intervals, f, indent=2)
        print(f"\nSaved combined intervals to: {combined_intervals_path}")
    
    else:
        # Sequential processing
        print(f"\nProcessing sketches sequentially...")
        
        for sketch_idx, entry in enumerate(data):
            result = process_single_sketch((sketch_idx, entry, surface_points, mesh_data, output_folder,
                                          angle_range, step, save_plots, cd_threshold))
            
            if result['success']:
                all_results.extend(result['results'])
                if result.get('intervals'):
                    all_intervals.extend(result['intervals'])
    
    # Save summary results
    summary_path = f'{output_folder}/revolve_analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, indent=2, fp=f)
    print(f"\nSaved summary results to: {summary_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if all_results:
        min_cds = [r['min_cd'] for r in all_results]
        optimal_rotations = [r['optimal_rotation'] for r in all_results]
        cd_at_zeros = [r['cd_at_zero'] for r in all_results]
        
        print(f"Total axes analyzed: {len(all_results)}")
        print(f"\nMinimum CD statistics:")
        print(f"  Mean: {np.mean(min_cds):.6f}")
        print(f"  Median: {np.median(min_cds):.6f}")
        print(f"  Min: {np.min(min_cds):.6f}")
        print(f"  Max: {np.max(min_cds):.6f}")
        
        print(f"\nOptimal rotation angle statistics:")
        print(f"  Mean: {np.mean(optimal_rotations):.1f}°")
        print(f"  Median: {np.median(optimal_rotations):.1f}°")
        print(f"  Min: {np.min(optimal_rotations):.1f}°")
        print(f"  Max: {np.max(optimal_rotations):.1f}°")
        
        print(f"\nCD at zero rotation:")
        print(f"  Mean: {np.mean(cd_at_zeros):.6f}")
        print(f"  Median: {np.median(cd_at_zeros):.6f}")
        
        # Plot overall distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(optimal_rotations, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Optimal Rotation Angle (degrees)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Optimal Rotation Angles', fontsize=14)
        
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
    parser = argparse.ArgumentParser(description='Analyze revolve axis optimization by computing Chamfer Distance')
    parser.add_argument('loops_json', help='Path to loops_split.json file')
    parser.add_argument('stl_path', help='Path to STL file')
    parser.add_argument('--output-folder', default='revolve_axis_analysis', help='Output folder for results')
    parser.add_argument('--num-surface-samples', type=int, default=100000, help='Number of surface points to sample')
    parser.add_argument('--angle-min', type=float, default=0.0, help='Minimum rotation angle (degrees)')
    parser.add_argument('--angle-max', type=float, default=30.0, help='Maximum rotation angle (degrees)')
    parser.add_argument('--step', type=float, default=30.0, help='Rotation angle step size (degrees)')
    parser.add_argument('--num-workers', type=int, default=25, help='Number of parallel workers')
    parser.add_argument('--normalize', action='store_true', help='Normalize mesh before analysis')
    parser.add_argument('--no-plots', action='store_true', help='Disable saving visualization plots')
    parser.add_argument('--cd-threshold', type=float, default=0.01, help='CD threshold for identifying low-CD intervals')

    args = parser.parse_args()

    analyze_all_revolve_axes(
        args.loops_json,
        args.stl_path,
        output_folder=args.output_folder,
        num_surface_samples=args.num_surface_samples,
        angle_range=(args.angle_min, args.angle_max),
        step=args.step,
        num_workers=args.num_workers,
        normalize=args.normalize,
        save_plots=not args.no_plots,
        cd_threshold=args.cd_threshold,
    )
