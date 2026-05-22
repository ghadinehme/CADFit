#!/usr/bin/env python3
"""
simplified extract_sketches.py

Loads an STL, takes axis-aligned slices and slices at detected flat-face
planes, and extracts the intersection edges as ordered loops of 2D points
(in the slice plane coordinates). Outputs JSON with entries:
  { origin: [x,y,z], normal: [nx,ny,nz], xdir: [ux,uy,uz], loops: [ [ [x,y], ... ], ... ] }

Usage:
  python extract_sketches.py 00000472.stl --out loops.json
  python3 extract_sketches.py 3.stl

Dependencies: trimesh, numpy
"""

import argparse
import json
import math
from typing import List, Tuple

import numpy as np
import trimesh
from iou import normalize_mesh


def is_rectangular_loop(loop_2d, parallel_tolerance=0.01):
    """
    Check if a 2D loop is approximately rectangular by detecting two pairs of parallel edges.
    
    Strategy: Compute the oriented bounding box using PCA. If the loop fits tightly
    to this bounding box and the box has perpendicular sides, it's a rectangle.
    This handles rectangles with or without filleted corners.
    
    Args:
        loop_2d: List of [x, y] coordinates
        parallel_tolerance: Tolerance for parallel edge detection
    
    Returns:
        True if the loop appears to be rectangular
    """
    if len(loop_2d) < 4:
        return False
    
    center = np.mean(loop_2d, axis=0)
    distances = np.linalg.norm(loop_2d - center, axis=1)
    if np.allclose(distances, distances[0], atol=1e-2):
        return False
    
    # Remove duplicate closing point if present
    loop = np.array(loop_2d)
    if len(loop) > 0 and np.allclose(loop[0], loop[-1]):
        loop = loop[:-1]
    
    if len(loop) < 4:
        return False
    
    # Center the points
    centroid = np.mean(loop, axis=0)
    centered = loop - centroid
    
    # Compute covariance matrix and get principal axes via PCA
    try:
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort by eigenvalue (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Check if one eigenvalue is much larger than the other (indicates a line, not a rectangle)
        if eigenvalues[1] < 1e-6 or eigenvalues[0] / eigenvalues[1] > 100:
            return False
        
        # Project points onto principal axes
        projected = centered @ eigenvectors
        
        # Compute bounding box in principal component space
        min_proj = np.min(projected, axis=0)
        max_proj = np.max(projected, axis=0)
        
        # Rectangle dimensions
        width = max_proj[0] - min_proj[0]
        height = max_proj[1] - min_proj[1]
        
        if width < 1e-6 or height < 1e-6:
            return False
        
        # Create rectangle corners in principal space
        rect_corners = np.array([
            [min_proj[0], min_proj[1]],
            [max_proj[0], min_proj[1]],
            [max_proj[0], max_proj[1]],
            [min_proj[0], max_proj[1]]
        ])
        
        # Transform back to original space
        rect_corners_orig = rect_corners @ eigenvectors.T + centroid
        
        # For each loop point, find distance to nearest rectangle edge
        total_dist = 0
        sample_points = loop[::max(1, len(loop)//100)]  # Sample ~100 points
        
        for point in sample_points:
            # Find minimum distance to the 4 rectangle edges
            min_dist = float('inf')
            for i in range(4):
                p1 = rect_corners_orig[i]
                p2 = rect_corners_orig[(i + 1) % 4]
                
                # Distance from point to line segment
                edge = p2 - p1
                edge_len_sq = np.dot(edge, edge)
                if edge_len_sq > 1e-12:
                    t = max(0, min(1, np.dot(point - p1, edge) / edge_len_sq))
                    closest = p1 + t * edge
                    dist = np.linalg.norm(point - closest)
                    min_dist = min(min_dist, dist)
            
            total_dist += min_dist
        
        # Compute average distance from points to rectangle boundary
        avg_dist = total_dist / max(1, len(sample_points))
        
        # Compute characteristic dimension (average of width and height)
        char_dim = (width + height) / 2.0
        
        # If average distance is small relative to the rectangle size, it's rectangular
        # This allows for filleted corners which will have slightly higher deviation
        if char_dim > 0 and avg_dist / char_dim < parallel_tolerance:
            return True
            
    except Exception as e:
        return False
    
    return False

def quantile_slices(bounds_min, bounds_max, count=10):
    val_range = bounds_max - bounds_min
    return list(np.linspace(bounds_min, bounds_max-1e-6, count))


def detect_flat_planes(mesh: trimesh.Trimesh, angle_tol=1e-2, area_thresh=1e-6):
    normals = mesh.face_normals
    areas = mesh.area_faces
    big = areas > area_thresh
    if not np.any(big):
        return []
    normals = normals[big]
    face_indices = np.nonzero(big)[0]
    n = normals.copy()
    flip = n[:, 2] < 0
    n[flip] *= -1
    rounded = np.round(n / angle_tol) * angle_tol
    groups = {}
    for i, r in enumerate(rounded):
        key = tuple((r / angle_tol).astype(int))
        groups.setdefault(key, []).append(face_indices[i])
    planes = []
    for inds in groups.values():
        if len(inds) == 0:
            continue
        verts_idx = mesh.faces[inds].ravel()
        pts = mesh.vertices[verts_idx]
        centroid = pts.mean(axis=0)
        uu, dd, vv = np.linalg.svd(pts - centroid)
        normal = vv[-1]
        normal = normal / np.linalg.norm(normal)
        planes.append((tuple(centroid.tolist()), tuple(normal.tolist())))
    # unique by normal
    uniq = []
    seen = []
    for origin, normal in planes:
        n = np.array(normal)
        if n[2] < 0:
            n = -n
        found = False
        for s in seen:
            if np.linalg.norm(n - s) < 1e-3:
                found = True
                break
        if not found:
            seen.append(n)
            uniq.append((origin, tuple(n.tolist())))
    return uniq


def compute_xdir(normal: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Compute x-direction vector (u vector) for plane coordinate system.
    Choose two orthogonal vectors in the plane.
    
    Args:
        normal: Normal vector of the plane
        
    Returns:
        X-direction unit vector in the plane
    """
    normal_array = np.array(normal)
    normal_array = normal_array / np.linalg.norm(normal_array)  # Ensure normalized
    
    # Choose a vector that's not parallel to the normal
    if abs(normal_array[2]) < 0.9:
        u = np.cross(normal_array, [0, 0, 1])
    else:
        u = np.cross(normal_array, [1, 0, 0])
    
    # Normalize the u vector
    u = u / np.linalg.norm(u)
    
    # Round to e-7 precision for consistency
    rounded_xdir = [round(float(x), 7) for x in u.tolist()]
    
    return tuple(rounded_xdir)


def slice_mesh_loops(mesh: trimesh.Trimesh, origin: Tuple[float, float, float], normal: Tuple[float, float, float], samples_per_loop: int = 500):
    """Return tuple (loops2d, loops3d).

    loops2d: list of loops where each loop is list of [x,y]
    loops3d: list of loops where each loop is list of [x,y,z] in mesh coords
    """
    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is None:
        return [], []
    try:
        res = section.to_planar()
    except Exception:
        return [], []
    if res is None:
        return [], []
    planar = res[0] if isinstance(res, tuple) else res
    transform = res[1] if isinstance(res, tuple) and len(res) > 1 else None

    loops = []
    loops3d = []

    def _densify_2d_loop(loop_pts2d, target_samples=None):
        """Return a densified closed 2D loop (list of [x,y]).

        target_samples: approximate desired total points along perimeter. If
        None, uses the outer `samples_per_loop` value.
        """
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
            # generate nsub points along segment excluding last point (will be added by next)
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
                loops.append(loop_pts)
                # compute 3D loop if transform provided (map densified 2D points)
                if transform is not None:
                    pts3 = []
                    for (x, y) in loop_pts:
                        v = np.array([float(x), float(y), 0.0, 1.0])
                        p3 = transform.dot(v)[:3]
                        pts3.append([float(p3[0]), float(p3[1]), float(p3[2])])
                    if pts3 and (pts3[0] != pts3[-1]):
                        pts3.append(pts3[0])
                    loops3d.append(pts3)
            # interior rings (holes): export each as its own loop
            for interior in poly.interiors:
                icoords = list(interior.coords)
                if len(icoords) >= 2:
                    iloop = [[float(x), float(y)] for (x, y) in icoords]
                    iloop = _densify_2d_loop(iloop)
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
        # Path2D.entities are sequences; entity.discrete() gives coordinates
        # We'll attempt to link segments by endpoints to form closed loops
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', help='Input STL file')
    p.add_argument('--out', '-o', default='loops.json', help='Output JSON file')
    p.add_argument('--axis-slices', type=int, default=5, help='Number of axis-aligned slices per axis')
    p.add_argument('--samples', type=int, default=3000, help='Approx samples per extracted loop')
    p.add_argument('--normalize', action='store_true', help='Normalize mesh into [-10,10] box before slicing')
    args = p.parse_args()

    mesh = trimesh.load(args.input, force='mesh')
    if args.normalize:
        mesh = normalize_mesh(mesh)
    if not isinstance(mesh, trimesh.Trimesh):
        raise SystemExit('failed to load mesh')

    #     if max_dim > 0:
    #     else:

    results = []
    results_with_rects = []  # Keep all loops including rectangles

    bounds = mesh.bounds
    for axis in range(3):
        mins = bounds[0][axis]
        maxs = bounds[1][axis]
        slices = quantile_slices(mins, maxs, args.axis_slices)
        for s in slices:
            origin = [0.0, 0.0, 0.0]
            origin[axis] = float(s)
            normal = [0.0, 0.0, 0.0]
            normal[axis] = 1.0
            try:
                loops2d, loops3d = slice_mesh_loops(mesh, origin=origin, normal=normal, samples_per_loop=args.samples)
                if loops2d:
                    # Separate rectangular and non-rectangular loops
                    filtered_loops2d = []
                    filtered_loops3d = []
                    
                    for i, loop2d in enumerate(loops2d):
                        is_rect = is_rectangular_loop(loop2d)
                        if is_rect:
                            print(f'  Filtered out rectangular loop {i} at {origin}')
                        if not is_rect:
                            filtered_loops2d.append(loop2d)
                            if i < len(loops3d):
                                filtered_loops3d.append(loops3d[i])
                    
                    # Always save the full (unfiltered) entry for fallback
                    xdir = compute_xdir(tuple(normal))
                    full_entry = {
                        'origin': tuple(origin),
                        'normal': tuple(normal),
                        'xdir': xdir,
                        'loops': [l for l in loops2d]
                    }
                    if loops3d:
                        full_entry['loops_3d'] = [l for l in loops3d]
                    results_with_rects.append(full_entry)
                    
                    # Only add filtered entry if there are non-rectangular loops
                    if filtered_loops2d:
                        entry = {
                            'origin': tuple(origin), 
                            'normal': tuple(normal), 
                            'xdir': xdir,
                            'loops': filtered_loops2d
                        }
                        if filtered_loops3d:
                            entry['loops_3d'] = filtered_loops3d
                        results.append(entry)
            except Exception as e:
                print(f'Error processing slice at {origin}: {e}')

    # If fewer than 50 non-rectangular sketches, keep rectangles too
    total_non_rect = sum(len(e['loops']) for e in results)
    if total_non_rect < 5 and results_with_rects:
        print(f'  Only {total_non_rect} non-rectangular loops found (<50), keeping rectangles too ({sum(len(e["loops"]) for e in results_with_rects)} total)')
        results = results_with_rects

    #     if loops2d:
    #         
    #             'origin': origin, 
    #             'normal': normal, 
    #             'xdir': xdir,
    #             'loops': loops2d
    #         if loops3d:

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'wrote {len(results)} sections to {args.out}')


if __name__ == '__main__':
    main()
