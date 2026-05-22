#!/usr/bin/env python3
"""
Parallel Extract planar contours from mesh by sampling surface points.

This script parallelizes plane processing where each worker handles a unique plane:
1. Samples points uniformly on the mesh surface
2. Groups points by their face normals to detect planar regions
3. Requires at least 2 faces to be coplanar to consider them as a plane
4. Extracts contours by sectioning the mesh at detected plane locations (parallelized)
5. Outputs JSON with plane information and their boundary contours

Usage:
  python extract_planar_contours.py test_20.stl --out planar_contours.json --num-workers 8
  python extract_planar_contours.py mesh.stl --samples 20000 --min-faces 3 --num-workers 16

Dependencies: trimesh, numpy, sklearn
"""

import argparse
import json
import math
import multiprocessing as mp
import tempfile
import os
import time
from typing import List, Tuple, Dict
from collections import defaultdict
import pymeshlab
import numpy as np
import trimesh
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_revolve_axis import chamfer_distance
import random


def sample_surface_points(mesh: trimesh.Trimesh, n_samples: int = 10000):
    """
    Sample points uniformly on the mesh surface.
    Returns: (points, face_indices, face_normals)
    """
    # Sample points on surface
    points, face_indices = mesh.sample(n_samples, return_index=True)
    
    # Get corresponding face normals
    face_normals = mesh.face_normals[face_indices]
    
    return points, face_indices, face_normals


def remove_duplicate_planes(planes, normal_tolerance=0.01, distance_tolerance=0.01):
    """
    Remove duplicate planes that have the same normal (or opposite normals) and similar origins.
    
    Args:
        planes: List of plane dictionaries with 'origin', 'normal', etc.
        normal_tolerance: Tolerance for considering normals as the same
        distance_tolerance: Tolerance for considering origins as coplanar
    
    Returns:
        Filtered list of unique planes
    """
    if len(planes) <= 1:
        return planes
    
    unique_planes = []
    
    for i, plane in enumerate(planes):
        origin = np.array(plane['origin'])
        normal = np.array(plane['normal'])
        normal = normal / np.linalg.norm(normal)  # ensure normalized
        
        is_duplicate = False
        
        # Check against all previously accepted planes
        for existing_plane in unique_planes:
            existing_origin = np.array(existing_plane['origin'])
            existing_normal = np.array(existing_plane['normal'])
            existing_normal = existing_normal / np.linalg.norm(existing_normal)  # ensure normalized
            
            # Check if normals are the same or opposite (same plane, different orientation)
            dot_product = abs(np.dot(normal, existing_normal))
            
            if dot_product > (1.0 - normal_tolerance):  # Normals are very similar or opposite
                # Check if both origins lie on both planes (not just parallel planes)
                # Distance from current origin to the existing plane
                origin_diff = origin - existing_origin
                distance_current_to_existing = abs(np.dot(origin_diff, existing_normal))
                
                # Distance from existing origin to the current plane  
                distance_existing_to_current = abs(np.dot(-origin_diff, normal))
                
                # Both distances should be small for the planes to be the same
                if (distance_current_to_existing < distance_tolerance and 
                    distance_existing_to_current < distance_tolerance):
                    # This is a duplicate plane
                    print(f"  Removing duplicate plane {i}: normal similarity = {dot_product:.6f}, "
                          f"distances = {distance_current_to_existing:.6f}, {distance_existing_to_current:.6f}")
                    
                    # Keep the plane with more faces/points (more representative)
                    if plane['num_faces'] > existing_plane['num_faces']:
                        # Replace the existing plane with this better one
                        for j, up in enumerate(unique_planes):
                            if up == existing_plane:
                                unique_planes[j] = plane
                                print(f"    Replaced with plane {i} (more faces: {plane['num_faces']} vs {existing_plane['num_faces']})")
                                break
                    
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_planes.append(plane)
    
    return unique_planes


def detect_planar_regions(points, face_indices, face_normals, mesh, 
                         normal_tolerance=0.05, distance_tolerance=0.1, min_faces=2):
    """
    Detect planar regions by clustering surface points based on face normals and positions.
    
    Args:
        points: sampled surface points
        face_indices: indices of faces the points belong to
        face_normals: normals of the faces
        mesh: the original mesh
        normal_tolerance: tolerance for normal similarity
        distance_tolerance: tolerance for point-to-plane distance
        min_faces: minimum number of unique faces required for a plane
    
    Returns:
        List of plane definitions: [(origin, normal, supporting_points, face_set), ...]
    """
    print(f"Analyzing {len(points)} surface points...")
    
    # First, group points by similar normals using clustering
    # Normalize normals to handle orientation consistency
    normals_normalized = face_normals.copy()
    # Make normals consistent (all point outward)
    for i, normal in enumerate(normals_normalized):
        if np.dot(normal, normal) > 1e-8:  # non-zero normal
            normals_normalized[i] = normal / np.linalg.norm(normal)
    
    # Use DBSCAN to cluster normals
    # Convert normals to spherical coordinates for better clustering
    theta = np.arccos(np.clip(normals_normalized[:, 2], -1, 1))  # polar angle
    phi = np.arctan2(normals_normalized[:, 1], normals_normalized[:, 0])  # azimuth
    
    # Combine spherical coordinates for clustering
    spherical_coords = np.column_stack([theta, phi])
    
    # Cluster by normal direction
    normal_clustering = DBSCAN(eps=normal_tolerance, min_samples=30).fit(spherical_coords)
    normal_labels = normal_clustering.labels_
    
    print(f"Found {len(set(normal_labels)) - (1 if -1 in normal_labels else 0)} normal clusters")
    
    planes = []
    
    # For each normal cluster, check if points are coplanar
    for label in set(normal_labels):
        if label == -1:  # noise points
            continue
            
        # Get points belonging to this normal cluster
        cluster_mask = normal_labels == label
        cluster_points = points[cluster_mask]
        cluster_face_indices = face_indices[cluster_mask]
        cluster_normals = normals_normalized[cluster_mask]
        
        if len(cluster_points) < min_faces:
            continue
            
        # Get unique faces in this cluster
        unique_faces = set(cluster_face_indices)
        if len(unique_faces) < min_faces:
            print(f"  Cluster {label}: only {len(unique_faces)} faces, skipping")
            continue
        
        # Compute average normal for this cluster
        avg_normal = np.mean(cluster_normals, axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        
        # Use DBSCAN to cluster points by their distance to a plane
        # For each point, compute distance to the plane passing through centroid with avg_normal
        centroid = np.mean(cluster_points, axis=0)
        
        # Distance from each point to the plane
        distances = np.abs(np.dot(cluster_points - centroid, avg_normal))
        
        # Cluster points that are close to the same plane
        distance_coords = distances.reshape(-1, 1)
        plane_clustering = DBSCAN(eps=distance_tolerance, min_samples=30).fit(distance_coords)
        plane_labels = plane_clustering.labels_
        
        # For each plane cluster, check if it has enough faces
        for plane_label in set(plane_labels):
            if plane_label == -1:
                continue
                
            plane_mask = plane_labels == plane_label
            plane_points = cluster_points[plane_mask]
            plane_face_indices = cluster_face_indices[plane_mask]
            
            unique_plane_faces = set(plane_face_indices)
            if len(unique_plane_faces) < min_faces:
                continue
                
            # Fit plane to these points using SVD
            plane_centroid = np.mean(plane_points, axis=0)
            centered_points = plane_points - plane_centroid
            
            if len(centered_points) < 3:
                continue
                
            # SVD to get plane normal
            try:
                U, S, Vt = np.linalg.svd(centered_points.T)
                plane_normal = U[:, -1]  # last column is normal to the plane
                
                # Ensure normal points outward (consistent with face normals)
                if np.dot(plane_normal, avg_normal) < 0:
                    plane_normal = -plane_normal
                    
                # Verify planarity by checking residual
                residuals = np.abs(np.dot(centered_points, plane_normal))
                max_residual = np.max(residuals)
                
                if max_residual > distance_tolerance * 2:
                    print(f"  Plane cluster {plane_label}: high residual {max_residual:.4f}, skipping")
                    continue
                
                # Compute x-direction (u vector) for plane coordinate system
                # Choose two orthogonal vectors in the plane
                if abs(plane_normal[2]) < 0.9:
                    u = np.cross(plane_normal, [0, 0, 1])
                else:
                    u = np.cross(plane_normal, [1, 0, 0])
                u = u / np.linalg.norm(u)
                
                # Round origin, normal, and xdir to e-7 precision for consistency
                rounded_origin = [round(float(x), 7) for x in plane_centroid.tolist()]
                rounded_normal = [round(float(x), 7) for x in plane_normal.tolist()]
                rounded_xdir = [round(float(x), 7) for x in u.tolist()]
                
                planes.append({
                    'origin': rounded_origin,
                    'normal': rounded_normal,
                    'xdir': rounded_xdir,
                    'points': plane_points.tolist(),
                    'face_indices': list(unique_plane_faces),
                    'num_faces': len(unique_plane_faces),
                    'num_points': len(plane_points),
                    'max_residual': float(max_residual)
                })
                
                print(f"  Found plane: {len(unique_plane_faces)} faces, {len(plane_points)} points, "
                      f"residual: {max_residual:.4f}")
                
            except np.linalg.LinAlgError:
                print(f"  SVD failed for plane cluster {plane_label}")
                continue
    
    return planes


def slice_mesh_loops(mesh: trimesh.Trimesh, origin: Tuple[float, float, float], 
                    normal: Tuple[float, float, float], samples_per_loop: int = 500, delta = 0.0001):
    """
    Extract 2D and 3D loops from mesh section at given plane.
    Takes multiple slices around the plane to capture all boundaries including holes.
    """
    origin = np.array(origin)
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)  # ensure normalized
    
    # Define slice offsets to capture features slightly above/below the plane
    # Use small offsets relative to mesh size
    bounds = mesh.bounds
    mesh_size = np.linalg.norm(bounds[1] - bounds[0])
    base_offset = mesh_size * 0.001  # 0.1% of mesh size
    slice_offsets = [
            -delta,
            delta
        ]
    
    all_loops_2d = []
    all_loops_3d = []
    all_transforms = []
    
    # Collect loops from multiple slice planes
    for offset in slice_offsets:
        slice_origin = origin + normal * offset
        
        section = mesh.section(plane_origin=slice_origin, plane_normal=normal)
        if section is None:
            continue
        
        try:
            res = section.to_2D()
        except Exception:
            continue
        
        if res is None:
            continue
        
        planar = res[0] if isinstance(res, tuple) else res
        transform = res[1] if isinstance(res, tuple) and len(res) > 1 else None
        
        if transform is None:
            continue
            
        # Extract loops from this slice
        loops_2d, loops_3d = extract_loops_from_planar(planar, transform, samples_per_loop)
        
        if loops_2d:
            all_loops_2d.extend(loops_2d)
            all_loops_3d.extend(loops_3d)
            all_transforms.extend([transform] * len(loops_2d))
            break
    
    if not all_loops_2d:
        return [], []
    
    # Project all 3D loops back onto the original plane
    projected_loops_2d = []
    projected_loops_3d = []
    
    # Create coordinate system for the original plane
    # Choose two orthogonal vectors in the plane
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0, 0, 1])
    else:
        u = np.cross(normal, [1, 0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    for loop_3d in all_loops_3d:
        if not loop_3d:
            continue
            
        # Project each 3D point onto the original plane
        projected_2d = []
        projected_3d = []
        
        for point_3d in loop_3d:
            point = np.array(point_3d)
            
            # Project point onto the plane
            to_point = point - origin
            distance_to_plane = np.dot(to_point, normal)
            projected_point = point - distance_to_plane * normal
            
            # Convert to 2D coordinates in the plane's coordinate system
            relative_pos = projected_point - origin
            x_2d = np.dot(relative_pos, u)
            y_2d = np.dot(relative_pos, v)
            
            projected_2d.append([float(x_2d), float(y_2d)])
            projected_3d.append([float(projected_point[0]), float(projected_point[1]), float(projected_point[2])])
        
        if len(projected_2d) >= 3:  # Only keep loops with at least 3 points
            # Ensure all points in the projected loop are distinct
            projected_2d = remove_duplicate_points(projected_2d, tolerance=1e-6)
            projected_3d = remove_duplicate_points(projected_3d, tolerance=1e-6)
            
            if len(projected_2d) >= 3 and len(projected_3d) >= 3:
                projected_loops_2d.append(projected_2d)
                projected_loops_3d.append(projected_3d)
    
    return projected_loops_2d, projected_loops_3d


def fit_plane_to_loop(loop_3d):
    """Fit a plane to a 3D loop and return its normal and centroid."""
    if len(loop_3d) < 3:
        return None, None
    
    loop_array = np.array(loop_3d)
    centroid = np.mean(loop_array, axis=0)
    centered = loop_array - centroid
    
    try:
        U, S, Vt = np.linalg.svd(centered.T)
        normal = U[:, -1]  # Last column is normal to the plane
        return normal, centroid
    except np.linalg.LinAlgError:
        return None, None


def are_planes_parallel(normal1, normal2, threshold=0.95):
    """Check if two plane normals are parallel (dot product close to ±1)."""
    if normal1 is None or normal2 is None:
        return False
    dot_product = abs(np.dot(normal1, normal2))
    return dot_product > threshold


def project_loop_onto_common_plane(loop_3d, centroid, normal):
    """
    Project a 3D loop onto a plane defined by centroid and normal.
    Returns 2D coordinates in the plane's coordinate system.
    """
    loop_array = np.array(loop_3d)
    
    # Create orthonormal basis for the plane
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0, 0, 1])
    else:
        u = np.cross(normal, [1, 0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    
    # Project each point onto the plane
    projected_2d = []
    for point_3d in loop_array:
        # Project onto plane
        to_point = point_3d - centroid
        distance_to_plane = np.dot(to_point, normal)
        projected_point = point_3d - distance_to_plane * normal
        
        # Convert to 2D coordinates
        relative_pos = projected_point - centroid
        x = np.dot(relative_pos, u)
        y = np.dot(relative_pos, v)
        projected_2d.append([x, y])
    
    return np.array(projected_2d)


def compute_projected_shape_difference(loop1_2d, loop2_2d):
    """
    Compute the difference between two 2D projected loops using Chamfer distance.
    Returns a normalized distance metric.
    """
    if len(loop1_2d) < 3 or len(loop2_2d) < 3:
        return float('inf')
    
    # Compute centroids and center the loops
    centroid1 = np.mean(loop1_2d, axis=0)
    centroid2 = np.mean(loop2_2d, axis=0)
    
    centered1 = loop1_2d - centroid1
    centered2 = loop2_2d - centroid2
    
    # Compute scale (average distance from centroid)
    scale1 = np.mean(np.linalg.norm(centered1, axis=1))
    scale2 = np.mean(np.linalg.norm(centered2, axis=1))
    
    if scale1 < 1e-8 or scale2 < 1e-8:
        return float('inf')
    
    # Normalize by scale
    normalized1 = centered1 / scale1
    normalized2 = centered2 / scale2
    
    # Compute Chamfer distance (average nearest neighbor distance)
    # Average Chamfer distance
    chamfer_dist = (chamfer_distance(normalized1, normalized2, average=True) + chamfer_distance(normalized2, normalized1, average=True)) / 2.0
    
    return chamfer_dist


def are_loops_parallel_duplicates(loop1_3d, loop2_3d, shape_tolerance=0.05, perimeter_tolerance=0.15):
    """
    Check if two loops are parallel duplicates by:
    1. Fitting planes to both loops
    2. Checking if planes are parallel
    3. Checking if centroids are aligned along the plane normal
    4. Projecting both loops onto a common plane and comparing shapes
    
    Args:
        loop1_3d, loop2_3d: 3D loop coordinates
        shape_tolerance: Tolerance for projected shape difference (Chamfer distance)
        perimeter_tolerance: Relative tolerance for perimeter similarity
    
    Returns:
        True if loops are parallel duplicates
    """
    if len(loop1_3d) < 3 or len(loop2_3d) < 3:
        return False
    
    # Calculate perimeters
    loop1_array = np.array(loop1_3d)
    loop2_array = np.array(loop2_3d)
    
    perimeter1 = 0.0
    for j in range(len(loop1_3d)):
        p1 = loop1_array[j]
        p2 = loop1_array[(j + 1) % len(loop1_3d)]
        perimeter1 += np.linalg.norm(p2 - p1)
    
    perimeter2 = 0.0
    for j in range(len(loop2_3d)):
        p1 = loop2_array[j]
        p2 = loop2_array[(j + 1) % len(loop2_3d)]
        perimeter2 += np.linalg.norm(p2 - p1)
    
    # Check if perimeters are similar
    if max(perimeter1, perimeter2) > 1e-8:
        perimeter_ratio = abs(perimeter1 - perimeter2) / max(perimeter1, perimeter2)
        if perimeter_ratio > perimeter_tolerance:
            return False
    else:
        return False
    
    # Fit planes to both loops
    normal1, centroid1 = fit_plane_to_loop(loop1_3d)
    normal2, centroid2 = fit_plane_to_loop(loop2_3d)
    
    if normal1 is None or normal2 is None:
        return False
    
    # Check if planes are parallel
    if not are_planes_parallel(normal1, normal2, threshold=0.95):
        return False
    
    # Ensure normals point in the same direction
    if np.dot(normal1, normal2) < 0:
        normal2 = -normal2
    
    # Check if centroids are aligned along the plane normal
    centroid_vector = centroid2 - centroid1
    centroid_dist = np.linalg.norm(centroid_vector)
    
    # If centroids are too close, already handled by other duplicate detection
    if centroid_dist < 0.01:
        return False
    
    # Check if centroid vector is parallel to the plane normals
    centroid_vector_normalized = centroid_vector / centroid_dist
    parallel_to_normal1 = abs(np.dot(centroid_vector_normalized, normal1))
    parallel_to_normal2 = abs(np.dot(centroid_vector_normalized, normal2))
    
    if parallel_to_normal1 < 0.85 or parallel_to_normal2 < 0.85:
        return False  # Centroids not aligned with plane normals
    
    # Use average normal for common projection plane
    avg_normal = (normal1 + normal2) / 2.0
    avg_normal = avg_normal / np.linalg.norm(avg_normal)
    
    # Use midpoint between centroids as common plane origin
    common_centroid = (centroid1 + centroid2) / 2.0
    
    # Project both loops onto the common plane
    projected1 = project_loop_onto_common_plane(loop1_3d, common_centroid, avg_normal)
    projected2 = project_loop_onto_common_plane(loop2_3d, common_centroid, avg_normal)
    
    # Compute projected shape difference
    shape_diff = compute_projected_shape_difference(projected1, projected2)
    
    if shape_diff < 0.01:
        return True
    
    return False


def remove_duplicate_loops(loops_2d, loops_3d, tolerance=0.01):
    """
    Remove duplicate loops based on similarity of their 3D points.
    Now also detects and removes parallel duplicate loops.
    """
    if not loops_2d or not loops_3d:
        return [], []
        
    filtered_2d = []
    filtered_3d = []
    
    for i, loop_3d in enumerate(loops_3d):
        if len(loop_3d) < 3:
            continue
            
        is_duplicate = False
        loop_3d_array = np.array(loop_3d)
        loop_3d_centroid = np.mean(loop_3d_array, axis=0)
        
        # Calculate 3D perimeter as a size metric
        loop_3d_perimeter = 0.0
        for j in range(len(loop_3d)):
            p1 = np.array(loop_3d[j])
            p2 = np.array(loop_3d[(j + 1) % len(loop_3d)])
            loop_3d_perimeter += np.linalg.norm(p2 - p1)
        
        # Check against already filtered loops
        for j, existing_loop_3d in enumerate(filtered_3d):
            if len(existing_loop_3d) < 3:
                continue
                
            existing_3d_array = np.array(existing_loop_3d)
            existing_3d_centroid = np.mean(existing_3d_array, axis=0)
            
            # Calculate existing loop perimeter
            existing_3d_perimeter = 0.0
            for k in range(len(existing_loop_3d)):
                p1 = np.array(existing_loop_3d[k])
                p2 = np.array(existing_loop_3d[(k + 1) % len(existing_loop_3d)])
                existing_3d_perimeter += np.linalg.norm(p2 - p1)
            
            # Check if 3D centroids are close and perimeters are similar
            centroid_dist_3d = np.linalg.norm(loop_3d_centroid - existing_3d_centroid)
            perimeter_ratio = abs(loop_3d_perimeter - existing_3d_perimeter) / max(loop_3d_perimeter, existing_3d_perimeter, 1e-8)
            
            # Original duplicate check (same location)
            if centroid_dist_3d < tolerance and perimeter_ratio < 0.1:
                is_duplicate = True
                print(f"    Removing duplicate loop {i} (same location as loop {j})")
                break
            
            # New check: parallel duplicate loops (only if perimeters are similar)
            if perimeter_ratio < 0.15:  # Similar size
                if are_loops_parallel_duplicates(loop_3d, existing_loop_3d):
                    is_duplicate = True
                    print(f"    Removing parallel duplicate loop {i} (parallel to loop {j}, distance: {centroid_dist_3d:.4f})")
                    break
        
        if not is_duplicate:
            if i < len(loops_2d):
                filtered_2d.append(loops_2d[i])
            filtered_3d.append(loop_3d)
    
    return filtered_2d, filtered_3d


def compute_loop_area(loop_2d):
    """
    Compute the area of a 2D loop using the shoelace formula.
    """
    if len(loop_2d) < 3:
        return 0.0
    
    points = np.array(loop_2d)
    x = points[:, 0]
    y = points[:, 1]
    
    # Shoelace formula
    area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))
    return area


def project_loop_to_2d_planar(loop_3d, origin, normal):
    """Project a 3D loop onto the 2D plane defined by origin and normal."""
    # Convert to numpy arrays
    origin = np.array(origin)
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)  # normalize
    
    # Create orthonormal basis for the plane
    # Choose an arbitrary vector not parallel to normal
    if abs(normal[0]) < 0.9:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.array([0.0, 1.0, 0.0])
    
    # Make u orthogonal to normal
    u = u - np.dot(u, normal) * normal
    u = u / np.linalg.norm(u)
    
    # Create second basis vector
    v = np.cross(normal, u)
    
    # Project each point
    projected = []
    for point_3d in loop_3d:
        point = np.array(point_3d)
        # Translate to origin
        relative = point - origin
        # Project onto plane
        x = np.dot(relative, u)
        y = np.dot(relative, v)
        projected.append((x, y))
    
    return projected


def is_loop_contained_in_loop_planar(inner_loop, outer_loop, origin, normal, tolerance=1e-6):
    """Check if inner_loop is contained within outer_loop."""
    try:
        # Import shapely here to avoid dependency issues
        from shapely.geometry import Polygon, Point
        from shapely.errors import TopologicalError
        
        # Project both loops to 2D
        inner_2d = project_loop_to_2d_planar(inner_loop, origin, normal)
        outer_2d = project_loop_to_2d_planar(outer_loop, origin, normal)
        
        # Create polygons (ensure they're closed)
        if len(inner_2d) < 3 or len(outer_2d) < 3:
            return False
            
        # Close loops if not already closed
        if inner_2d[0] != inner_2d[-1]:
            inner_2d.append(inner_2d[0])
        if outer_2d[0] != outer_2d[-1]:
            outer_2d.append(outer_2d[0])
        
        # Create shapely polygons
        try:
            outer_poly = Polygon(outer_2d)
            inner_poly = Polygon(inner_2d)
            
            # Make sure polygons are valid
            if not outer_poly.is_valid:
                outer_poly = outer_poly.buffer(0)
            if not inner_poly.is_valid:
                inner_poly = inner_poly.buffer(0)
            
            # Check if inner loop is contained within outer loop
            # We check if the inner polygon is within the outer polygon
            return outer_poly.contains(inner_poly) or outer_poly.covers(inner_poly)
            
        except (TopologicalError, ValueError) as e:
            # Fallback: check if most points of inner loop are inside outer polygon
            try:
                outer_poly = Polygon(outer_2d)
                if not outer_poly.is_valid:
                    outer_poly = outer_poly.buffer(0)
                
                inside_count = 0
                for point in inner_2d[:-1]:  # Skip the duplicate closing point
                    if outer_poly.contains(Point(point)):
                        inside_count += 1
                
                # Consider contained if most points (>80%) are inside
                return inside_count / len(inner_2d[:-1]) > 0.8
            except:
                return False
                
    except ImportError:
        print("Warning: shapely not available, using fallback containment check")
        # Simple fallback: check if centroids are close (not very reliable)
        inner_centroid = np.mean(inner_2d, axis=0)
        outer_centroid = np.mean(outer_2d, axis=0)
        return np.linalg.norm(inner_centroid - outer_centroid) < tolerance
    except Exception as e:
        print(f"Warning: Failed to check containment: {e}")
        return False


def group_contained_loops_planar(loops, origin, normal):
    """Group loops where some are contained within others."""
    if len(loops) <= 1:
        return [[loop] for loop in loops]
    
    # Calculate areas to help determine outer vs inner loops
    loop_areas = []
    for loop in loops:
        try:
            projected = project_loop_to_2d_planar(loop, origin, normal)
            if len(projected) >= 3:
                # Close loop if needed
                if projected[0] != projected[-1]:
                    projected.append(projected[0])
                # Simple area calculation using shoelace formula
                area = 0.5 * abs(sum(projected[i][0] * projected[i+1][1] - projected[i+1][0] * projected[i][1] 
                                   for i in range(-1, len(projected)-1)))
                loop_areas.append(area)
            else:
                loop_areas.append(0)
        except:
            loop_areas.append(0)
    
    # Sort loops by area (largest first - these are likely outer loops)
    loop_indices = list(range(len(loops)))
    loop_indices.sort(key=lambda i: loop_areas[i], reverse=True)
    
    groups = []
    used_indices = set()
    
    for i in loop_indices:
        if i in used_indices:
            continue
            
        # Start a new group with this loop as the potential outer loop
        current_group = [loops[i]]
        current_group_indices = [i]
        used_indices.add(i)
        
        # Find all loops contained within this one
        for j in loop_indices:
            if j in used_indices or j == i:
                continue
                
            # Check if loop j is contained in loop i
            if is_loop_contained_in_loop_planar(loops[j], loops[i], origin, normal):
                contained = False
                # Also check if it's already contained in any other loop in the current group
                for loop in current_group[1:]:
                    if is_loop_contained_in_loop_planar(loops[j], loop, origin, normal):
                        contained = True
                        break
                if not contained:
                    current_group.append(loops[j])
                    current_group_indices.append(j)
                    used_indices.add(j)
        
        # Sort the group by area (largest first) to ensure consistent ordering
        if len(current_group) > 1:
            # Create pairs of (area, loop) and sort by area descending
            group_with_areas = [(loop_areas[idx], loops[idx]) for idx in current_group_indices]
            group_with_areas.sort(key=lambda x: x[0], reverse=True)
            current_group = [loop for area, loop in group_with_areas]
        
        groups.append(current_group)
    
    # Handle any remaining ungrouped loops
    for i in range(len(loops)):
        if i not in used_indices:
            groups.append([loops[i]])
    
    return groups


def remove_duplicate_points(points, tolerance=1e-6):
    """
    Remove duplicate points from a list/array of points, keeping points distinct.
    
    Args:
        points: List or array of points [[x,y], [x,y], ...] or [[x,y,z], [x,y,z], ...]
        tolerance: Minimum distance between points to consider them distinct
    
    Returns:
        List of distinct points
    """
    if not points or len(points) <= 1:
        return points
    
    distinct_points = []
    points_array = np.array(points, dtype=float)
    
    for i, point in enumerate(points_array):
        # Check if this point is close to any already selected point
        is_duplicate = False
        
        if distinct_points:
            existing_points = np.array(distinct_points)
            distances = np.linalg.norm(existing_points - point, axis=1)
            if np.min(distances) <= tolerance:
                is_duplicate = True
        
        if not is_duplicate:
            if len(point) == 2:
                distinct_points.append([float(point[0]), float(point[1])])
            else:  # 3D points
                distinct_points.append([float(point[0]), float(point[1]), float(point[2])])
    
    return distinct_points


def extract_loops_from_planar(planar, transform, samples_per_loop):
    """
    Extract loops from a planar section.
    """
    loops = []
    loops3d = []

    def _densify_2d_loop(loop_pts2d, target_samples=None):
        """Return a densified closed 2D loop (list of [x,y])."""
        if target_samples is None:
            target_samples = samples_per_loop
        if not loop_pts2d:
            return loop_pts2d
        
        # ensure closed
        if loop_pts2d[0] != loop_pts2d[-1]:
            loop_pts2d = list(loop_pts2d) + [loop_pts2d[0]]
        
        pts = np.array(loop_pts2d, dtype=float)
        segs = pts[1:] - pts[:-1]
        lens = np.linalg.norm(segs, axis=1)
        perim = float(lens.sum())
        
        if perim <= 0 or len(pts) <= 1:
            return loop_pts2d
        
        # compute samples per segment proportional to length
        target = max(int(target_samples), len(pts))
        parts = np.maximum(1, np.round(lens / perim * target).astype(int))
        new_pts = []
        
        for k in range(len(segs)):
            p0 = pts[k]
            p1 = pts[k + 1]
            nsub = int(parts[k])
            # generate nsub points along segment excluding last point
            for t in np.linspace(0.0, 1.0, nsub, endpoint=False):
                pt = p0 * (1.0 - t) + p1 * t
                new_pts.append([float(pt[0]), float(pt[1])])
        
        # ensure closed by appending final point equal to first
        if not new_pts or new_pts[0] != new_pts[-1]:
            new_pts.append([float(new_pts[0][0]), float(new_pts[0][1])])
        
        return new_pts

    # prefer polygons if available
    if hasattr(planar, 'polygons_full') and planar.polygons_full:
        for poly in planar.polygons_full:
            # outer ring
            coords = list(poly.exterior.coords)
            if len(coords) >= 2:
                loop_pts = [[float(x), float(y)] for (x, y) in coords]
                # densify loop for more samples
                loop_pts = _densify_2d_loop(loop_pts)
                # ensure all points in loop are distinct
                loop_pts = remove_duplicate_points(loop_pts, tolerance=1e-6)
                if len(loop_pts) >= 3:  # Only keep loops with at least 3 distinct points
                    loops.append(loop_pts)
                
                    # compute 3D loop if transform provided
                    if transform is not None:
                        pts3 = []
                        for (x, y) in loop_pts:
                            v = np.array([float(x), float(y), 0.0, 1.0])
                            p3 = transform.dot(v)[:3]
                            pts3.append([float(p3[0]), float(p3[1]), float(p3[2])])
                        if pts3 and (pts3[0] != pts3[-1]):
                            pts3.append(pts3[0])
                        loops3d.append(pts3)
            
            # interior rings (holes)
            for interior in poly.interiors:
                icoords = list(interior.coords)
                if len(icoords) >= 2:
                    iloop = [[float(x), float(y)] for (x, y) in icoords]
                    iloop = _densify_2d_loop(iloop)
                    # ensure all points in loop are distinct
                    iloop = remove_duplicate_points(iloop, tolerance=1e-6)
                    if len(iloop) >= 3:  # Only keep loops with at least 3 distinct points
                        loops.append(iloop)
                        if transform is not None:
                            pts3 = []
                            for (x, y) in iloop:
                                v = np.array([float(x), float(y), 0.0, 1.0])
                                p3 = transform.dot(v)[:3]
                                pts3.append([float(p3[0]), float(p3[1]), float(p3[2])])
                            if pts3 and (pts3[0] != pts3[-1]):
                                pts3.append(pts3[0])
                            loops3d.append(pts3)
    else:
        # fallback: collect continuous entities into loops
        segments = []
        for ent in planar.entities:
            pts = np.asarray(ent.discrete())
            if pts.shape[0] >= 2:
                segments.append([tuple(map(float, p)) for p in pts])
        
        # link segments by endpoints
        used = [False] * len(segments)
        for i, seg in enumerate(segments):
            if used[i]:
                continue
            used[i] = True
            loop = list(seg)
            changed = True
            while changed:
                changed = False
                end = loop[-1]
                for j, s2 in enumerate(segments):
                    if used[j]:
                        continue
                    if np.allclose(end, s2[0]):
                        loop.extend(s2[1:])
                        used[j] = True
                        changed = True
                        break
                    if np.allclose(end, s2[-1]):
                        loop.extend(reversed(s2[:-1]))
                        used[j] = True
                        changed = True
                        break
                # try to close to start
                if not changed and len(loop) > 2 and np.allclose(loop[0], loop[-1]):
                    break
            
            # ensure closed and filter degenerate, then densify
            loop_pts = [[float(x), float(y)] for (x, y) in loop]
            if len(loop_pts) >= 2 and (loop_pts[0] != loop_pts[-1]):
                loop_pts.append(loop_pts[0])
            if len(loop_pts) >= 3:
                loop_pts = _densify_2d_loop(loop_pts)
                # ensure all points in loop are distinct
                loop_pts = remove_duplicate_points(loop_pts, tolerance=1e-6)
                if len(loop_pts) >= 3:  # Only keep loops with at least 3 distinct points
                    loops.append(loop_pts)
                    if transform is not None:
                        pts3 = []
                        for (x, y) in loop_pts:
                            v = np.array([float(x), float(y), 0.0, 1.0])
                            p3 = transform.dot(v)[:3]
                            pts3.append([float(p3[0]), float(p3[1]), float(p3[2])])
                        if pts3 and (pts3[0] != pts3[-1]):
                            pts3.append(pts3[0])
                        loops3d.append(pts3)
    
    return loops, loops3d


def process_plane_worker(args):
    """
    Worker function to process a single plane.
    
    Args:
        args: Tuple containing (plane_idx, plane_data, mesh_path, loop_samples)
    
    Returns:
        Dictionary with plane processing results
    """
    plane_idx, plane_data, mesh_path, loop_samples, delta = args
    
    try:
        print(f"Worker processing plane {plane_idx + 1}")
        
        # Load mesh in worker
        mesh = trimesh.load(mesh_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            return {'plane_idx': plane_idx, 'success': False, 'error': 'Failed to load mesh'}
        
        origin = plane_data['origin']
        normal = plane_data['normal']
        xdir = plane_data['xdir']
        
        # Extract contours for this plane
        loops2d, loops3d = slice_mesh_loops(
            mesh, 
            origin=origin, 
            normal=normal,
            samples_per_loop=loop_samples,
            delta=delta
        )
        
        if not loops2d:
            return {'plane_idx': plane_idx, 'success': True, 'entries': []}
        
        print(f"  Worker {plane_idx + 1}: Extracted {len(loops2d)} contour loops")
        
        entries = []
        
        # 1. Add each individual loop as its own entry
        for j, (loop2d, loop3d) in enumerate(zip(loops2d, loops3d)):
            individual_entry = {
                'plane_id': plane_idx,
                'loop_id': j,
                'entry_type': 'individual_loop',
                'origin': origin,
                'normal': normal,
                'xdir': xdir,
                'num_faces': plane_data['num_faces'],
                'num_sample_points': plane_data['num_points'],
                'max_residual': plane_data['max_residual'],
                'loops': [loop2d],
                'loops_3d': [loop3d] if loop3d else None
            }
            entries.append(individual_entry)
        
        # 2. If multiple loops, check for containment relationships
        if len(loops2d) > 1:
            # Group loops where some are contained within others
            loop_groups = group_contained_loops_planar(loops3d, origin, normal)
            
            # Add grouped entries (outer loop + inner loops)
            for group_idx, group in enumerate(loop_groups):
                if len(group) > 1:  # Only create grouped entries for multi-loop groups
                    # Create corresponding 2D loops for this group
                    group_loops_2d = []
                    for loop_3d in group:
                        # Find matching 2D loop
                        for k, check_loop_3d in enumerate(loops3d):
                            if loop_3d == check_loop_3d:
                                group_loops_2d.append(loops2d[k])
                                break
                    
                    grouped_entry = {
                        'plane_id': plane_idx,
                        'group_id': group_idx,
                        'entry_type': 'grouped_loops',
                        'origin': origin,
                        'normal': normal,
                        'xdir': xdir,
                        'num_faces': plane_data['num_faces'],
                        'num_sample_points': plane_data['num_points'],
                        'max_residual': plane_data['max_residual'],
                        'loops': group_loops_2d,
                        'loops_3d': group,
                        'group_description': f'outer_loop_with_{len(group)-1}_inner_loops'
                    }
                    entries.append(grouped_entry)
                    print(f"    Worker {plane_idx + 1}: Added grouped entry: 1 outer + {len(group)-1} inner loops")
        
        return {'plane_idx': plane_idx, 'success': True, 'entries': entries}
        
    except Exception as e:
        print(f"Error processing plane {plane_idx + 1}: {e}")
        return {'plane_idx': plane_idx, 'success': False, 'error': str(e)}


def remesh(mesh):
    """Remesh the mesh to increase resolution"""
    V = mesh.vertices
    F = mesh.faces

    edge_midpoint_cache = {}
    new_vertices = list(V)
    new_faces = []

    def midpoint(i, j):
        key = tuple(sorted((i, j)))
        if key not in edge_midpoint_cache:
            vi = V[i]
            vj = V[j]
            vm = 0.5 * (vi + vj)
            edge_midpoint_cache[key] = len(new_vertices)
            new_vertices.append(vm)
        return edge_midpoint_cache[key]

    for f in F:
        a, b, c = f
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)

        # 4 new triangles per original
        new_faces.append([a, ab, ca])
        new_faces.append([ab, b, bc])
        new_faces.append([ca, bc, c])
        new_faces.append([ab, bc, ca])

    new_vertices = np.array(new_vertices)
    new_faces = np.array(new_faces)

    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    return mesh


def main():
    parser = argparse.ArgumentParser(description='Parallel extract planar contours from mesh by surface sampling')
    parser.add_argument('input', help='Input STL file')
    parser.add_argument('--out', '-o', default='planar_contours.json', help='Output JSON file')
    parser.add_argument('--samples', type=int, default=10000, help='Number of surface points to sample')
    parser.add_argument('--min-faces', type=int, default=5, help='Minimum faces required for a plane')
    parser.add_argument('--normal-tol', type=float, default=0.03, help='Normal similarity tolerance')
    parser.add_argument('--distance-tol', type=float, default=0.01, help='Point-to-plane distance tolerance')
    parser.add_argument('--loop-samples', type=int, default=200, help='Approximate samples per contour loop')
    parser.add_argument('--normalize', action='store_true', help='Normalize mesh to [-10, 10] box')
    parser.add_argument('--num-workers', type=int, default=30, help='Number of worker processes (default: 30)')
    parser.add_argument('--delta', type=float, default=0.03, help='Offset for slice planes (default: 0.0001)')

    args = parser.parse_args()

    # Load mesh
    mesh = trimesh.load(args.input, force='mesh')
    in_path = args.input
    if not isinstance(mesh, trimesh.Trimesh):
        raise SystemExit('Failed to load mesh')

    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Remesh the mesh
    n_vertices = len(mesh.vertices)
    while n_vertices < 50000:
        mesh = remesh(mesh)
        n_vertices = len(mesh.vertices)
    if n_vertices > 60000:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(in_path)
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=30000,
            preservenormal=True,
            preservetopology=True,
            optimalplacement=True,
            planarquadric=True,
            qualitythr=1.0,
        )
        mesh = ms.current_mesh()
        ms.save_current_mesh("decimated_mesh.stl")

        mesh = trimesh.load("decimated_mesh.stl", force='mesh')
        
    print(f"Remeshed mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Normalize mesh if requested
    if args.normalize:
        try:
            bb_min = mesh.bounds[0]
            bb_max = mesh.bounds[1]
            center = 0.5 * (bb_min + bb_max)
            max_dim = float((bb_max - bb_min).max())
            if max_dim > 0:
                scale = 2.0 / max_dim
                mesh.apply_translation(-center)
                mesh.apply_scale(scale)
                print(f'Normalized mesh: center={center.tolist()}, max_dim={max_dim:.6g}, scale={scale:.6g}')
            else:
                print('Mesh has zero size; skipping normalization')
        except Exception as e:
            print(f'Failed to normalize mesh: {e}')

    # Sample surface points
    print(f"Sampling {args.samples} points on surface...")
    points, face_indices, face_normals = sample_surface_points(mesh, args.samples)
    print(f"Sampled {len(points)} surface points")

    # Detect planar regions
    print("Detecting planar regions...")
    planes = detect_planar_regions(
        points, face_indices, face_normals, mesh,
        normal_tolerance=args.normal_tol,
        distance_tolerance=args.distance_tol,
        min_faces=args.min_faces
    )
    
    print(f"Found {len(planes)} planar regions")

    # Remove duplicate planes (same normal direction or opposite)
    print("Removing duplicate planes...")
    unique_planes = remove_duplicate_planes(planes, normal_tolerance=args.normal_tol, distance_tolerance=args.distance_tol)
    print(f"After deduplication: {len(unique_planes)} unique planar regions")

    if not unique_planes:
        print("No planes found, saving empty results")
        with open(args.out, 'w') as f:
            json.dump([], f, indent=2)
        return

    # Save mesh to temporary file for worker access
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_mesh:
        mesh.export(tmp_mesh.name)
        mesh_path = tmp_mesh.name

    try:
        # Prepare arguments for parallel processing
        worker_args = []
        for plane_idx, plane_data in enumerate(unique_planes):
            args_tuple = (plane_idx, plane_data, mesh_path, args.loop_samples, args.delta)
            worker_args.append(args_tuple)

        print(f"Processing {len(unique_planes)} planes using {args.num_workers} workers...")
        
        # Process planes in parallel
        start_time = time.time()
        with mp.Pool(args.num_workers) as pool:
            worker_results = pool.map(process_plane_worker, worker_args)
        
        processing_time = time.time() - start_time
        print(f"Parallel processing completed in {processing_time:.2f} seconds")

        # Collect results
        results = []
        total_individual_loops = 0
        total_grouped_loops = 0
        failed_planes = 0

        for worker_result in worker_results:
            if worker_result['success']:
                results.extend(worker_result['entries'])
                
                # Count loop types
                for entry in worker_result['entries']:
                    if entry['entry_type'] == 'individual_loop':
                        total_individual_loops += 1
                    elif entry['entry_type'] == 'grouped_loops':
                        total_grouped_loops += 1
            else:
                failed_planes += 1
                print(f"Failed to process plane {worker_result['plane_idx'] + 1}: {worker_result.get('error', 'Unknown error')}")
        
        print(f"\nCollected {len(results)} total entries before parallel duplicate removal")
        print(f"  - {total_individual_loops} individual loop entries")
        print(f"  - {total_grouped_loops} grouped loop entries")

        if len(results) > 30:
            # Remove parallel duplicate loops across all planes
            print("\nRemoving parallel duplicate loops across all entries...")
            filtered_results = []
            removed_count = 0
            
            for i, entry in enumerate(results):
                if entry['entry_type'] != 'individual_loop':
                    # Keep all grouped entries as-is
                    filtered_results.append(entry)
                    continue
                
                # Check individual loops for parallel duplicates
                loop_3d = entry['loops_3d'][0] if entry['loops_3d'] else None
                if loop_3d is None or len(loop_3d) < 3:
                    continue
                
                is_duplicate = False
                
                # Compare against already filtered individual loops
                for existing_entry in filtered_results:
                    if existing_entry['entry_type'] != 'individual_loop':
                        continue
                    
                    existing_loop_3d = existing_entry['loops_3d'][0] if existing_entry['loops_3d'] else None
                    if existing_loop_3d is None or len(existing_loop_3d) < 3:
                        continue
                    
                    # Check for same-location duplicates
                    loop_centroid = np.mean(np.array(loop_3d), axis=0)
                    existing_centroid = np.mean(np.array(existing_loop_3d), axis=0)
                    centroid_dist = np.linalg.norm(loop_centroid - existing_centroid)
                    
                    if centroid_dist < 0.01:
                        is_duplicate = True
                        removed_count += 1
                        print(f"  Removing duplicate at same location: entry {i}")
                        break
                    
                    # Check for parallel duplicates
                    if are_loops_parallel_duplicates(loop_3d, existing_loop_3d):
                        is_duplicate = True
                        removed_count += 1
                        print(f"  Removing parallel duplicate: entry {i} (parallel to existing entry, distance: {centroid_dist:.4f})")
                        break
                
                if not is_duplicate:
                    filtered_results.append(entry)
        
            results = filtered_results
            print(f"Removed {removed_count} parallel duplicate loops")
            print(f"Remaining entries: {len(results)}")
        
        if len(results) > 100:
            # randomly sample 150 entries to save
            results = random.sample(results, 100)
        # Save results
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)

        # Recount after filtering
        final_individual_loops = sum(1 for r in results if r.get('entry_type') == 'individual_loop')
        final_grouped_loops = sum(1 for r in results if r.get('entry_type') == 'grouped_loops')

        print(f"\nSaved {len(results)} entries to {args.out}")
        print(f"  - {final_individual_loops} individual loop entries")
        print(f"  - {final_grouped_loops} grouped loop entries")
        print(f"  - {failed_planes} failed planes")
        
        # Print summary by plane
        print("\nSummary by plane:")
        for i in range(len(unique_planes)):
            individual_count = sum(1 for r in results if r.get('plane_id') == i and r.get('entry_type') == 'individual_loop')
            grouped_count = sum(1 for r in results if r.get('plane_id') == i and r.get('entry_type') == 'grouped_loops')
            if individual_count > 0 or grouped_count > 0:
                print(f"  Plane {i+1}: {individual_count} individual + {grouped_count} grouped entries")

        # Performance summary
        avg_time_per_plane = processing_time / len(unique_planes) if unique_planes else 0
        print(f"\nPerformance summary:")
        print(f"  Total processing time: {processing_time:.2f} seconds")
        print(f"  Average time per plane: {avg_time_per_plane:.2f} seconds")
        print(f"  Workers used: {args.num_workers}")

    finally:
        # Clean up temporary mesh file
        try:
            os.unlink(mesh_path)
        except:
            pass


if __name__ == '__main__':
    main()
