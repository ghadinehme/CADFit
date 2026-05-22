#!/usr/bin/env python3
"""
fit_loops_to_primitives.py

Read `loops.json` (prefer `loops_3d`) and fit simple sketch primitives to each
loop: line segments, circles/arcs, or splines. Outputs nuCAD-style action
snippets (AddSketch / StartLoop / AddLine / AddArc / AddCircle / AddSpline
 / CloseProfile / MakeFace) that approximate each loop.

Usage:
  python3 fit_loops_to_primitives.py loops.json --out actions.txt

This is intentionally dependency-light (numpy + math). Thresholds are
conservative but tunable via command-line flags.
"""

import argparse
import json
import math
import time
from typing import List, Tuple, Optional
import os

import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def plane_basis(normal: Tuple[float, float, float], xdir: Optional[Tuple[float, float, float]] = None):
    n = np.array(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)

    if xdir is not None and xdir[0] is not None:
        # Use provided x-direction
        u = np.array(xdir, dtype=float)
        u = u / (np.linalg.norm(u) + 1e-12)
        
        # Ensure u is orthogonal to n
        u = u - np.dot(u, n) * n
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-12:
            # xdir is parallel to normal, fall back to automatic selection
            xdir = None
        else:
            u = u / u_norm
            v = np.cross(n, u)
            v = v / np.linalg.norm(v)
            return u, v, n
    
    # Automatic x-direction selection (original logic)
    if n[2] > 0.9:
        u = np.array([-1.0, 0.0, 0.0])
        v = np.array([0.0, -1.0, 0.0])
    elif n[1] > 0.9:
        v = np.array([0.0, 0.0, -1.0])
        u = np.array([1.0, 0.0, 0.0])
    elif n[0] > 0.9:
        u = np.array([0.0, -1.0, 0.0])
        v = np.array([0.0, 0.0, -1.0])
    else:
        # pick helper
        if abs(n[0]) < 0.9:
            a = np.array([1.0, 0.0, 0.0])
        else:
            a = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, a)
        if np.linalg.norm(u) < 1e-8:
            a = np.array([0.0, 0.0, 1.0])
            u = np.cross(n, a)
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
    return u, v, n


def project_points_to_plane(points3d: np.ndarray, origin: Tuple[float, float, float], normal: Tuple[float, float, float], xdir: Optional[Tuple[float, float, float]] = None):
    """Project Nx3 points into 2D plane coordinates (u,v) relative to origin.
    Returns (N,2) array.
    """
    u, v, n = plane_basis(normal, xdir)
    print(u, v, n)
    origin = np.array(origin, dtype=float)
    rel = points3d - origin.reshape((1, 3))
    x = rel.dot(u)
    y = rel.dot(v)
    return np.vstack((x, y)).T


def fit_line_2d(pts: np.ndarray):
    """Fit a line to 2D points using PCA. Return direction unit vector, point on line, and rms error (perp dist).
    """
    mean = pts.mean(axis=0)
    U, S, Vt = np.linalg.svd(pts - mean)
    direction = Vt[0]
    # perpendicular distances
    proj = (pts - mean).dot(direction)
    recon = np.outer(proj, direction) + mean
    dists = np.linalg.norm(pts - recon, axis=1)
    rms = math.sqrt(np.mean(dists ** 2))
    return direction, mean, rms


def fit_circle_2d(pts: np.ndarray):
    """Kasa circle fit (algebraic). Returns center (cx,cy), radius, rms of radial residuals.
    Requires at least 3 points.
    """
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    cx, cy, c = sol
    r = math.sqrt(cx * cx + cy * cy + c)
    dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    resid = dists - r
    rms = math.sqrt(np.mean(resid ** 2))
    return (cx, cy), r, rms


def angular_span(angles: np.ndarray):
    # unwrap and compute span
    a = np.unwrap(angles)
    return a.max() - a.min()


def angular_coverage(angles: np.ndarray) -> float:
    """Compute angular coverage (radians) of a set of angles in [-pi,pi).

    Returns 2*pi - max_gap where max_gap is the largest gap between consecutive
    angles after sorting. This is robust to wrap-around and unordered samples.
    """
    if angles.size == 0:
        return 0.0
    a = np.mod(angles, 2 * math.pi)
    a_sorted = np.sort(a)
    # include wrap-around gap
    gaps = np.diff(a_sorted)
    if a_sorted.size > 1:
        wrap_gap = (a_sorted[0] + 2 * math.pi) - a_sorted[-1]
        max_gap = max(gaps.max() if gaps.size > 0 else 0.0, wrap_gap)
    else:
        max_gap = 2 * math.pi
    return 2 * math.pi - max_gap


def chamfer_distance(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """Compute chamfer distance between two point sets.
    
    Chamfer distance is the average of minimum distances from each point in set1
    to the closest point in set2, plus the average of minimum distances from 
    each point in set2 to the closest point in set1.
    """
    if len(pts1) == 0 or len(pts2) == 0:
        return float('inf')
    
    # Distance matrix between all pairs
    dists = cdist(pts1, pts2)
    
    # Forward direction: for each point in pts1, find closest in pts2
    min_dists_1_to_2 = np.min(dists, axis=1)
    
    # Chamfer distance is average of both directions
    chamfer_dist = np.mean(min_dists_1_to_2)
    
    return chamfer_dist


def sample_arc_points(cx: float, cy: float, r: float, a0: float, a1: float, n_samples: int = 50) -> np.ndarray:
    """Sample points along an arc from angle a0 to a1."""
    angles = np.linspace(a0, a1, n_samples)
    
    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    return np.column_stack([x, y])


def try_circle_arc(pts: np.ndarray, rms_tol: float, min_pts: int = 7):
    if pts.shape[0] < min_pts:
        return None
    res = fit_circle_2d(pts)
    if res is None:
        return None
    (cx, cy), r, rms = res
    if rms > rms_tol:
        return None
    # compute angles in point order
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    span = angular_coverage(angles)
    try:
        dist_ends = float(np.linalg.norm(pts[0] - pts[-1]))
    except Exception:
        dist_ends = 0.0
    start = angles[0]
    end = angles[-1]
    # ensure minimal angular span by unwrapping properly
    angs = np.unwrap(angles)
    return ('arc', (cx, cy, r, angs[0], angs[-1]), rms)


def remove_duplicate_points_2d(pts: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """
    Remove duplicate/overlapping points from 2D point array.
    
    Args:
        pts: Nx2 numpy array of 2D points
        tolerance: Minimum distance between points to consider them distinct
    
    Returns:
        Filtered numpy array with distinct points
    """
    if pts.shape[0] <= 1:
        return pts
    
    distinct_points = []
    
    for i, point in enumerate(pts):
        # Check if this point is close to any already selected point
        is_duplicate = False
        
        if distinct_points:
            existing_points = np.array(distinct_points)
            distances = np.linalg.norm(existing_points - point, axis=1)
            if np.min(distances) <= tolerance:
                is_duplicate = True
        
        if not is_duplicate:
            distinct_points.append(point)
    
    return np.array(distinct_points, dtype=float) if distinct_points else pts[:0]  # Return empty array of same type


def segment_greedy_fit(pts2: np.ndarray, scale: float, line_tol: float, circ_tol: float):
    """Greedy segmentation of pts2 (Nx2) into primitives. Returns list of primitives.
    Each primitive is a tuple: ('line', x0,y0,x1,y1) or ('circle', cx,cy,r) or ('arc', cx,cy,r,a0,a1) or ('spline', [pt...]).
    """
    # Remove overlapping/duplicate points to ensure all points are distinct
    pts2 = remove_duplicate_points_2d(pts2, tolerance=1e-6)
    n = pts2.shape[0]
    # check if circle:
    fit_circle_2d_res = fit_circle_2d(pts2)
    if fit_circle_2d_res is not None:
        if fit_circle_2d_res[2] <= 1e-3:
            return [('circle', (float(fit_circle_2d_res[0][0]), float(fit_circle_2d_res[0][1]), float(fit_circle_2d_res[1])), 0.0)]

    if n < 2:
        return []  # Not enough distinct points to create any primitives
    
    i = 0
    primitives = []
    min_line_pts = 2
    # require a longer run (>6 points) before considering circle/arc to avoid false positives
    min_circle_pts = 7
    while i < n - 1:
        best_j = None
        best_model = None
        best_err = float('inf')
        # try to extend j from i+1 up to n-1
        for j in range(i + min_line_pts - 1, n):
            seg = pts2[i:j + 1]
            # line fit
            dir_vec, pt_on, line_rms = fit_line_2d(seg)
            # circle/arc fit
            circ = try_circle_arc(seg, rms_tol=circ_tol*scale, min_pts=min_circle_pts)
            # evaluate

            if circ is not None and circ[2] < line_rms and circ[1][2] < 1000:
                typ = circ[0]
                err = circ[2]
            else:
                typ = 'line'
                err = line_rms
            # accept if error below thresholds
            if typ == 'line' and err <= line_tol * scale:
                best_j = j
                best_model = ('line', dir_vec, pt_on, seg[0], seg[-1], err)
                best_err = err
                # keep extending to get longer line
                continue
            if typ in ('circle', 'arc'):
                # allow acceptance if absolute rms small relative to scale OR relative to radius
                r = circ[1][2] if len(circ[1]) > 2 else 0.0
                rel_ok = (r > 1e-12 and (err / r) <= circ_tol)
                abs_ok = (err <= circ_tol * scale)
                if rel_ok or abs_ok:
                    best_j = j
                    best_model = (typ, circ[1], err)
                    best_err = err
                    continue
            if best_j is not None:
                break
            # else keep trying
        if best_j is None:
            # fallback: take a small spline segment of next few points
            j = min(i + 5, n - 1)
            seg = pts2[i:j + 1]
            primitives.append(('spline', [(float(x), float(y)) for x, y in seg.tolist()]))
            i = j + 1
        else:
            # record best_model
            if best_model[0] == 'line':
                # extract start/end in coordinates
                start = tuple(float(x) for x in best_model[3])
                end = tuple(float(x) for x in best_model[4])
                primitives.append(('line', start, end))
            elif best_model[0] in ('circle', 'arc'):
                typ = best_model[0]
                data = best_model[1]
                if typ == 'circle':
                    cx, cy, r = data
                    primitives.append(('circle', (float(cx), float(cy), float(r))))
                else:
                    cx, cy, r, a0, a1 = data
                    primitives.append(('arc', (float(cx), float(cy), float(r), float(a0), float(a1))))
            else:
                primitives.append(('spline', [(round(float(x), 3), round(float(y), 3)) for x, y in seg.tolist()]))
            i = best_j + 1
    primitives = _merge_adjacent_lines(primitives)

    return primitives


def _merge_adjacent_lines(primitives: List[Tuple]) -> List[Tuple]:
    """Merge consecutive 'line' primitives when they share the same x, same y, or same slope.

    Checks for every 2 consecutive lines:
    1. If all starts and ends have the same x, merge by taking first start and second end
    2. If all starts and ends have the same y, merge by taking first start and second end
    3. If the slope of first line equals slope of second line, merge by taking first start and second end
    4. If a line has length <= 0.01, remove it and replace the start of the next primitive with the start of that tiny line
    """
    if not primitives:
        return primitives
    
    out = []
    i = 0
    
    while i < len(primitives):
        prim = primitives[i]
        
        # Non-line primitives are added directly
        if prim[0] != 'line':
            out.append(prim)
            i += 1
            continue
        
        # Start of current line
        first_start = np.array(prim[1], dtype=float)
        current_end = np.array(prim[2], dtype=float)
        
        line_length = np.linalg.norm(current_end - first_start)
        if line_length < 0.01:
            i += 1
            continue
        #     if i + 1 < len(primitives):
        #         if next_prim[0] == 'line':
        #         elif next_prim[0] == 'arc':
        #             pass
        #         elif next_prim[0] == 'spline':
        #             if spline_pts:
        #     continue
        
        # Try to merge with subsequent consecutive lines
        j = i + 1
        while j < len(primitives):
            next_prim = primitives[j]
            
            # Stop if next primitive is not a line
            if next_prim[0] != 'line':
                break
            
            next_start = np.array(next_prim[1], dtype=float)
            next_end = np.array(next_prim[2], dtype=float)
            
            # Update next_start to match current_end if they differ
            if not np.allclose(current_end, next_start, atol=1e-9):
                next_start = current_end.copy()
                # Update the primitive in the list with new start point
                primitives[j] = ('line', tuple(next_start), next_prim[2])

            # Check if lines can be merged
            can_merge = False
            
            # Check 1: All x-coordinates are the same
            if (first_start[0] == current_end[0] == next_start[0] == next_end[0]):
                can_merge = True
            
            # Check 2: All y-coordinates are the same
            elif (first_start[1] == current_end[1] == next_start[1] == next_end[1]):
                can_merge = True
            
            # Check 3: Same slope
            else:
                # Calculate slopes
                # Current merged line: from first_start to current_end
                dx1 = current_end[0] - first_start[0]
                dy1 = current_end[1] - first_start[1]
                
                # Next line: from next_start to next_end
                dx2 = next_end[0] - next_start[0]
                dy2 = next_end[1] - next_start[1]
                
                # Check if slopes are equal using cross product (avoids division by zero)
                # slope1 = dy1/dx1, slope2 = dy2/dx2
                # slope1 == slope2 iff dy1*dx2 == dy2*dx1
                if dx1 * dy2 == dy1 * dx2:
                    can_merge = True
            
            # If can merge, update the end point and continue checking next line
            if can_merge:
                current_end = next_end  # Extend to the end of next line
                j += 1
            else:
                break
        
        # Add the merged line (from first start to final end)
        out.append(('line', (float(first_start[0]), float(first_start[1])), 
                           (float(current_end[0]), float(current_end[1]))))
        
        # Move to the next unprocessed primitive
        i = j
    
    return out


def primitives_to_nucad(actions_acc: List[str], primitives: List[Tuple], original_pts: np.ndarray = None):
    """Append nuCAD style action lines for primitives to actions_acc.
    
    Args:
        actions_acc: List to append action strings to
        primitives: List of fitted primitives
        original_pts: Original 2D points for optimal arc midpoint calculation
    """
    for prim in primitives:
        t = prim[0]
        if t == 'line':
            s = prim[1]
            e = prim[2]
            actions_acc.append(f"    AddLine(start=({s[0]:.6g}, {s[1]:.6g}), end=({e[0]:.6g}, {e[1]:.6g})),")
        elif t == 'circle':
            cx, cy, r = prim[1]
            actions_acc.append(f"    AddCircle(center=({cx:.6g}, {cy:.6g}), radius={r:.6g}),")
        elif t == 'arc':
            cx, cy, r, a0, a1 = prim[1]
            p1 = (cx + r * math.cos(a0), cy + r * math.sin(a0))
            p3 = (cx + r * math.cos(a1), cy + r * math.sin(a1))
            
            # Use optimal midpoint if original points are available

            # Determine which arc direction to plot using optimal midpoint logic
            if original_pts is not None and len(original_pts) > 0:
                # Sample both possible arc directions
                a1 = a1 % (2 * math.pi)
                a0 = a0 % (2 * math.pi)
                
                if a1 > a0:
                    arc_pts_short = sample_arc_points(cx, cy, r, a0, a1, n_samples=50)
                    arc_pts_long = sample_arc_points(cx, cy, r, a1, a0 + 2*math.pi, n_samples=50)
                else:
                    arc_pts_short = sample_arc_points(cx, cy, r, a1, a0, n_samples=50)
                    arc_pts_long = sample_arc_points(cx, cy, r, a0, a1 + 2*math.pi, n_samples=50)
                
                # Choose the arc with better chamfer distance
                chamfer_short = chamfer_distance(arc_pts_short, original_pts)
                chamfer_long = chamfer_distance(arc_pts_long, original_pts)
                
                if chamfer_short <= chamfer_long:
                    p2 = arc_pts_short[arc_pts_short.shape[0] // 2]
                else:
                    p2 = arc_pts_long[arc_pts_long.shape[0] // 2]
            else:
                # Fallback to original method if no original points available
                if a1 < a0:  # Handle wrap-around
                    angles = np.linspace(a0, a1 + 2*np.pi, 50)
                else:
                    angles = np.linspace(a0, a1, 50)
                p2_x = cx + r * np.cos(angles[25])
                p2_y = cy + r * np.sin(angles[25])
                p2 = (p2_x, p2_y)

            actions_acc.append(f"    AddArc(p1=({p1[0]:.6g}, {p1[1]:.6g}), p2=({p2[0]:.6g}, {p2[1]:.6g}), p3=({p3[0]:.6g}, {p3[1]:.6g})),")
        elif t == 'spline':
            pts = prim[1]
            pts_str = ', '.join(f"({float(x):.6g}, {float(y):.6g})" for (x, y) in pts)
            actions_acc.append(f"    AddSpline(control_points=[{pts_str}]),")
        else:
            # unknown -> plot as point sequence via AddSpline
            pts = prim[1]
            pts_str = ', '.join(f"({float(x):.6g}, {float(y):.6g})" for (x, y) in pts)
            actions_acc.append(f"    AddSpline(control_points=[{pts_str}]),")


def _prim_start_end(prim):
    """Return (start, end) 2D points for primitive if available; for closed-only primitives return (p,p) representative."""
    t = prim[0]
    if t == 'line':
        return tuple(prim[1]), tuple(prim[2])
    if t == 'spline':
        pts = prim[1]
        if not pts:
            return None, None
        return tuple(pts[0]), tuple(pts[-1])
    if t == 'circle':
        return None, None  # closed; no distinct start/end
    if t == 'arc':
        cx, cy, r, a0, a1 = prim[1]
        p0 = (float(cx + r * math.cos(a0)), float(cy + r * math.sin(a0)))
        p1 = (float(cx + r * math.cos(a1)), float(cy + r * math.sin(a1)))
        return p0, p1
    return None, None


def close_primitives_loop(primitives: List[Tuple], tol: float = 1e-6) -> List[Tuple]:
    """Ensure consecutive primitives connect and the loop is closed.

    If gaps exist between consecutive primitives their endpoints are bridged by
    inserting short 'line' primitives. Circles are treated as closed;
    representative points are used for connectivity.
    """
    if not primitives:
        return primitives
    out = []
    # iterate and insert connectors when needed
    prev_end = None
    for prim in primitives:
        start, end = _prim_start_end(prim)
        if start is None or end is None:
            # cannot reason about primitive; just append
            out.append(prim)
            prev_end = end
            continue
        if prev_end is None:
            out.append(prim)
            prev_end = end
            continue
        # compute gap between prev_end and start
        dx = start[0] - prev_end[0]
        dy = start[1] - prev_end[1]
        dist = math.hypot(dx, dy)
        if dist > tol:
            # insert connector line
            out.append(('line', (prev_end[0], prev_end[1]), (start[0], start[1])))
        out.append(prim)
        prev_end = end

    # ensure loop closure: connect last end to first start
    first_start, _ = _prim_start_end(out[0]) if out else (None, None)
    _, last_end = _prim_start_end(out[-1]) if out else (None, None)
    if first_start is not None and last_end is not None:
        dx = first_start[0] - last_end[0]
        dy = first_start[1] - last_end[1]
        if math.hypot(dx, dy) > tol:
            out.append(('line', (last_end[0], last_end[1]), (first_start[0], first_start[1])))
    return out


def plot_primitives_and_points(primitives: List[Tuple], pts2: np.ndarray, entry_idx: int = 0, loop_idx: int = 0, output_dir: str = "primitive_plots"):
    """
    Plot fitted primitives and original loop points using matplotlib.
    
    Args:
        primitives: List of fitted primitives (line, arc, circle)
        pts2: Original 2D points of the loop (Nx2 array)
        entry_idx: Index of the entry (for filename)
        loop_idx: Index of the loop within the entry (for filename)
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: Fitted primitives
    ax1.set_title(f'Fitted Primitives - Entry {entry_idx}, Loop {loop_idx}')
    ax1.set_aspect('equal')
    
    # Plot each primitive
    for i, prim in enumerate(primitives):
        prim_type = prim[0]
        
        if prim_type == 'line':
            start, end = prim[1], prim[2]
            ax1.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=2, 
                    label=f'Line {i}' if i == 0 else "")
            
        elif prim_type == 'circle':
            cx, cy, r = prim[1]
            circle = plt.Circle((cx, cy), r, fill=False, color='red', linewidth=2)
            ax1.add_patch(circle)
            ax1.plot(cx, cy, 'ro', markersize=4)  # Center point
            if i == 0:  # Add label only once
                ax1.plot([], [], 'r-', linewidth=2, label='Circle')
                
        elif prim_type == 'arc':
            cx, cy, r, a0, a1 = prim[1]
            
            # Determine which arc direction to plot using optimal midpoint logic
            if pts2 is not None and len(pts2) > 0:
                # Sample both possible arc directions
                a1 = a1 % (2 * math.pi)
                a0 = a0 % (2 * math.pi)
                
                if a1 > a0:
                    arc_pts_short = sample_arc_points(cx, cy, r, a0, a1, n_samples=50)
                    arc_pts_long = sample_arc_points(cx, cy, r, a1, a0 + 2*math.pi, n_samples=50)
                else:
                    arc_pts_short = sample_arc_points(cx, cy, r, a1, a0, n_samples=50)
                    arc_pts_long = sample_arc_points(cx, cy, r, a0, a1 + 2*math.pi, n_samples=50)
                
                # Choose the arc with better chamfer distance
                chamfer_short = chamfer_distance(arc_pts_short, pts2)
                chamfer_long = chamfer_distance(arc_pts_long, pts2)

                if chamfer_short <= chamfer_long:
                    x_arc, y_arc = arc_pts_short[:, 0], arc_pts_short[:, 1]
                else:
                    x_arc, y_arc = arc_pts_long[:, 0], arc_pts_long[:, 1]
            else:
                # Fallback to original method if no original points available
                if a1 < a0:  # Handle wrap-around
                    angles = np.linspace(a0, a1 + 2*np.pi, 50)
                else:
                    angles = np.linspace(a0, a1, 50)
                x_arc = cx + r * np.cos(angles)
                y_arc = cy + r * np.sin(angles)
            
            ax1.plot(x_arc, y_arc, 'g-', linewidth=2, 
                    label=f'Arc {i}' if i == 0 else "")
    
    # Set equal aspect and add legend
    ax1.legend()
    
    # Plot 2: Original points
    ax2.set_title(f'Original Loop Points - Entry {entry_idx}, Loop {loop_idx}')
    ax2.set_aspect('equal')
    
    # Scatter plot of original points
    ax2.scatter(pts2[:, 0], pts2[:, 1], c='blue', s=20, alpha=0.7, label='Original Points')
    
    # Connect points to show the loop
    if len(pts2) > 1:
        # Close the loop for visualization
        loop_pts = np.vstack([pts2, pts2[0:1]])  # Add first point at end
        ax2.plot(loop_pts[:, 0], loop_pts[:, 1], 'k--', alpha=0.5, linewidth=1, label='Point Connection')
    
    ax2.legend()
    
    # Set equal limits for both plots
    all_x = pts2[:, 0]
    all_y = pts2[:, 1]
    
    # Add some padding
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    padding = max(x_range, y_range) * 0.1
    
    x_min, x_max = all_x.min() - padding, all_x.max() + padding
    y_min, y_max = all_y.min() - padding, all_y.max() + padding
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    plt.tight_layout()
    
    # Save the plot
    filename = f"entry_{entry_idx:03d}_loop_{loop_idx:03d}_primitives.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved primitive plot: {filepath}")
    
    return filepath


def primitives_to_instances(primitives: List[Tuple], original_pts: np.ndarray = None):
    """Convert primitive specification to nuCAD action dataclass instances (2D coords).
    Returns a list of instances in sketch-local 2D coordinates.
    
    Args:
        primitives: List of fitted primitives
        original_pts: Original 2D points for optimal arc midpoint calculation
    """
    inst = []
    if len(primitives)>0:
        first_primitive = primitives[0]
    else:
        return []
    prev_point = None
    
    if first_primitive[0] == 'line':
        s = first_primitive[1]
        inst.append("sketch.moveTo({}, {})".format(round(float(s[0]), 5), round(float(s[1]), 5)))
        prev_point = np.array(s, dtype=float)
    elif first_primitive[0] == 'arc':
        cx, cy, r, a0, a1 = first_primitive[1]
        # sample three points on the arc: start, mid, end
        p1 = (float(cx + r * math.cos(a0)), float(cy + r * math.sin(a0)))
        inst.append("sketch.moveTo({}, {})".format(round(float(p1[0]), 5), round(float(p1[1]), 5)))
        prev_point = np.array(p1, dtype=float)

    for prim in primitives:
        t = prim[0]
        if t == 'line':
            e = prim[2]
            e_point = np.array(e, dtype=float)
            # Skip if endpoint is same as previous point
            if prev_point is not None and np.allclose(e_point, prev_point, atol=1e-2):
                continue
            inst.append("sketch.lineTo({}, {})".format(round(float(e[0]), 5), round(float(e[1]), 5)))
            prev_point = e_point
        elif t == 'circle':
            cx, cy, r = prim[1]
            # skip invalid circles
            if r is None or r <= 1e-8 or not math.isfinite(r):
                continue
            inst.append("sketch.moveTo({}, {})".format(round(float(cx), 5), round(float(cy), 5)))
            inst.append("sketch.circle({})".format(round(float(r), 5)))
        elif t == 'arc':
            cx, cy, r, a0, a1 = prim[1]
            # sample three points on the arc: start, mid, end
            p1 = (float(cx + r * math.cos(a0)), float(cy + r * math.sin(a0)))
            p3 = (float(cx + r * math.cos(a1)), float(cy + r * math.sin(a1)))
            p3_point = np.array(p3, dtype=float)
            
            # Skip if arc endpoint is same as previous point
            if prev_point is not None and np.allclose(p3_point, prev_point, atol=1e-2):
                continue
            
            # Use same optimal midpoint logic as primitives_to_nucad
            if original_pts is not None and len(original_pts) > 0:
                # Sample both possible arc directions
                a1_norm = a1 % (2 * math.pi)
                a0_norm = a0 % (2 * math.pi)
                
                if a1_norm > a0_norm:
                    arc_pts_short = sample_arc_points(cx, cy, r, a0_norm, a1_norm, n_samples=50)
                    arc_pts_long = sample_arc_points(cx, cy, r, a1_norm, a0_norm + 2*math.pi, n_samples=50)
                else:
                    arc_pts_short = sample_arc_points(cx, cy, r, a1_norm, a0_norm, n_samples=50)
                    arc_pts_long = sample_arc_points(cx, cy, r, a0_norm, a1_norm + 2*math.pi, n_samples=50)
                
                # Choose the arc with better chamfer distance
                chamfer_short = chamfer_distance(arc_pts_short, original_pts)
                chamfer_long = chamfer_distance(arc_pts_long, original_pts)
                
                if chamfer_short <= chamfer_long:
                    p2 = arc_pts_short[arc_pts_short.shape[0] // 2]
                    p2 = (float(p2[0]), float(p2[1]))
                else:
                    p2 = arc_pts_long[arc_pts_long.shape[0] // 2]
                    p2 = (float(p2[0]), float(p2[1]))
            else:
                # Fallback to original method if no original points available
                if a1 < a0:  # Handle wrap-around
                    angles = np.linspace(a0, a1 + 2*np.pi, 50)
                else:
                    angles = np.linspace(a0, a1, 50)
                p2_x = cx + r * np.cos(angles[25])
                p2_y = cy + r * np.sin(angles[25])
                p2 = (float(p2_x), float(p2_y))
            
            #     continue
            inst.append("sketch = sketch.threePointArc(({}, {}), ({}, {}))".format(round(float(p2[0]), 5), round(float(p2[1]), 5), round(float(p3[0]), 5), round(float(p3[1]), 5)))
            prev_point = p3_point
    return inst


def build_action_instances_from_entry(entry, line_tol=1e-4, circ_tol=1e-4, plot_primitives=False, entry_idx=0, loop_idx=0, plot_dir="primitive_plots"):
    """Build a list of nuCAD action dataclass instances from the entry (uses loops_3d).
    Returns a list of instances (AddSketch, StartLoop, primitives..., CloseProfile, MakeFace)
    """
    origin = tuple(entry.get('origin', (0.0, 0.0, 0.0)))
    normal = tuple(entry.get('normal', (0.0, 0.0, 1.0)))
    xdir = tuple(entry.get('xdir', (None, None, None)))
    loops3d = entry.get('loops_3d') or []
    if not loops3d:
        return []
    actions = []
    if xdir[0] is None:
        actions.append(f"plane = cq.Plane(origin=({round(origin[0], 5)}, {round(origin[1], 5)}, {round(origin[2], 5)}), normal=({round(normal[0], 5)}, {round(normal[1], 5)}, {round(normal[2], 5)}))\nsketch = cq.Workplane(plane)")
    else:
        actions.append(f"plane = cq.Plane(origin=({round(origin[0], 5)}, {round(origin[1], 5)}, {round(origin[2], 5)}), normal=({round(normal[0], 5)}, {round(normal[1], 5)}, {round(normal[2], 5)}), xDir=({round(xdir[0], 5)}, {round(xdir[1], 5)}, {round(xdir[2], 5)}))")
        actions.append(f"sketch = cq.Workplane(plane)")
    for loop3 in loops3d:
        if not loop3:
            continue
        pts3 = np.array(loop3, dtype=float)
        if pts3.shape[0] > 2 and np.allclose(pts3[0], pts3[-1]):
            pts3 = pts3[:-1]
        
        # Project to 2D and ensure distinct points
        pts2 = project_points_to_plane(pts3, origin, normal, xdir)
        pts2 = remove_duplicate_points_2d(pts2, tolerance=1e-6)

        if pts2.shape[0] < 2:
            print(f"  Warning: Loop has less than 2 distinct points after duplicate removal, skipping")
            continue

        bbox = pts2.max(axis=0) - pts2.min(axis=0)
        scale = math.hypot(bbox[0], bbox[1])
        if scale <= 0:
            scale = 1.0
        primitives = segment_greedy_fit(pts2, scale, line_tol, circ_tol)

        # ensure primitives connect and convert to nuCAD instances
        primitives_closed = close_primitives_loop(primitives)
        if plot_primitives:
            plot_primitives_and_points(primitives_closed, pts2, entry_idx, loop_idx, 
                                     output_dir=plot_dir)

        instances = primitives_to_instances(primitives_closed, pts2)
        if not instances:
            # skip empty/degenerate loop
            continue
        for a in instances:
            actions.append(a)
        # append CloseProfile and MakeFace for this loop
        if "circle" not in actions[-1]:
            actions.append("sketch = sketch.close()")
    return actions


def process_entry(entry, line_tol=1e-4, circ_tol=1e-4, plot_primitives=False, entry_idx=0, plot_dir="primitive_plots"):
    origin = tuple(entry.get('origin', (0.0, 0.0, 0.0)))
    normal = tuple(entry.get('normal', (0.0, 0.0, 1.0)))
    xdir = tuple(entry.get('xdir', (None, None, None)))
    loops3d = entry.get('loops_3d') or []
    if not loops3d:
        return None
    actions = []
    for loop_idx, loop3 in enumerate(loops3d):
        if not loop3:
            continue
        # AddSketch before every StartLoop
        if xdir[0] is None:
            actions.append(f"AddSketch(origin=({round(origin[0], 5)}, {round(origin[1], 5)}, {round(origin[2], 5)}), normal=({round(normal[0], 5)}, {round(normal[1], 5)}, {round(normal[2], 5)})),")
        else:
            actions.append(f"AddSketch(origin=({round(origin[0], 5)}, {round(origin[1], 5)}, {round(origin[2], 5)}), normal=({round(normal[0], 5)}, {round(normal[1], 5)}, {round(normal[2], 5)}), xdir=({round(xdir[0], 5)}, {round(xdir[1], 5)}, {round(xdir[2], 5)})),")
        pts3 = np.array(loop3, dtype=float)
        # If loop is closed with duplicate last point, drop last for fitting
        if pts3.shape[0] > 2 and np.allclose(pts3[0], pts3[-1]):
            pts3 = pts3[:-1]
        
        # Project to 2D plane coordinates
        pts2 = project_points_to_plane(pts3, origin, normal, xdir)
        
        # Ensure all points are distinct in 2D space
        pts2 = remove_duplicate_points_2d(pts2, tolerance=1e-6)
        
        if pts2.shape[0] < 2:
            print(f"  Warning: Loop has less than 2 distinct points after duplicate removal, skipping")
            continue
        
        # compute scale (diagonal) for tolerance normalization
        bbox = pts2.max(axis=0) - pts2.min(axis=0)
        scale = math.hypot(bbox[0], bbox[1])
        if scale <= 0:
            scale = 1.0
        primitives = segment_greedy_fit(pts2, scale, line_tol, circ_tol)

        # emit StartLoop and primitives (ensure closed)
        primitives_closed = close_primitives_loop(primitives)
        
        # Generate plot if requested
        if plot_primitives:
            plot_primitives_and_points(primitives_closed, pts2, entry_idx, loop_idx, 
                                     output_dir=plot_dir)
        
        actions.append("    StartLoop(),")
        primitives_to_nucad(actions, primitives_closed, pts2)
        # append CloseProfile and MakeFace after this loop
        actions.append("    CloseProfile(),")
        actions.append("    MakeFace(),")
    return actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', nargs='?', default='loops.json')
    p.add_argument('--out', '-o', default='fitted_actions.txt')
    p.add_argument('--line-tol', type=float, default=1e-4, help='line RMS tolerance (relative to scale)')
    p.add_argument('--circ-tol', type=float, default=1e-4, help='circle RMS tolerance (relative to scale)')
    p.add_argument('--plot', action='store_true', help='generate plots showing fitted primitives vs original points')
    p.add_argument('--plot-dir', default='primitive_plots', help='directory to save plots (default: primitive_plots)')
    p.add_argument('--exec', action='store_true', help='execute the fitted actions in the nuCAD environment (requires nuCAD imports)')
    p.add_argument('--export-stl', default=None, help='when --exec is used, export resulting mesh to given filename (without extension)')
    args = p.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)

    # Set up plotting if requested
    if args.plot:
        print(f"Generating plots in directory: {args.plot_dir}")

    all_actions = []
    for i, entry in enumerate(data):
        res = process_entry(entry, line_tol=args.line_tol, circ_tol=args.circ_tol, 
                          plot_primitives=args.plot, entry_idx=i, plot_dir=args.plot_dir)
        if res is None:
            continue
        # collect action string lines
        all_actions.extend(res)

    # Post-process the combined actions to merge trivial consecutive line segments
    # and convert nearly-full arcs to circles.
    def _parse_addline(s: str):
        m = re.search(r"AddLine\(start=\(\s*([\-0-9.eE]+),\s*([\-0-9.eE]+)\s*\),\s*end=\(\s*([\-0-9.eE]+),\s*([\-0-9.eE]+)\s*\)\)", s)
        if not m:
            return None
        return (float(m.group(1)), float(m.group(2))), (float(m.group(3)), float(m.group(4)))


    def _dist(p, q):
        return math.hypot(p[0]-q[0], p[1]-q[1])

    def fix_actions_list(lines: List[str]) -> List[str]:
        out = []
        i = 0
        while i < len(lines):
            s = lines[i]
            # try merge consecutive AddLine when contiguous and colinear
            cur = _parse_addline(s)
            if cur and out:
                prev = out[-1]
                prev_parsed = _parse_addline(prev)
                if prev_parsed:
                    p0, p1 = prev_parsed
                    q0, q1 = cur
                    # check adjacency (prev end == cur start)
                    if _dist(p1, q0) < 1e-9:
                        # check colinearity by comparing direction vectors
                        v1 = (p1[0]-p0[0], p1[1]-p0[1])
                        v2 = (q1[0]-q0[0], q1[1]-q0[1])
                        n1 = math.hypot(v1[0], v1[1])
                        n2 = math.hypot(v2[0], v2[1])
                        if n1 > 1e-12 and n2 > 1e-12:
                            dot = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1 * n2)
                            if abs(1.0 - abs(dot)) <= 1e-3:
                                # merge into previous: start=prev.start, end=cur.end
                                new_line = f"    AddLine(start=({p0[0]:.6g}, {p0[1]:.6g}), end=({q1[0]:.6g}, {q1[1]:.6g})),"
                                out[-1] = new_line
                                i += 1
                                continue
            # (no arc->circle post-processing)
            out.append(s)
            i += 1
        return out

    all_actions = fix_actions_list(all_actions)

    # write a single combined actions list
    with open(args.out, 'w') as f:
        f.write("actions = [\n")
        for a in all_actions:
            f.write(a + "\n")
        f.write("]\n")

    print(f'wrote combined actions list ({len(all_actions)} lines) to {args.out}')

    if args.exec:
        # attempt to import env
        try:
            from nuCAD.env.base_env import nuCADEnv
        except Exception as e:
            print('failed to import nuCADEnv:', e)
            return
        # build all instances together and run once
        insts_all = []
        for i, entry in enumerate(data):
            try:
                insts = build_action_instances_from_entry(entry, line_tol=args.line_tol, circ_tol=args.circ_tol)
            except Exception as e:
                print(f'entry {i}: failed to build instances: {e}')
                continue
            if not insts:
                print(f'entry {i}: no instances')
                continue
            insts_all.extend(insts)

        if not insts_all:
            print('no action instances to execute')
            return

        env = nuCADEnv(viz=False)
        env.reset()
        try:
            for act in insts_all:
                env.step(act)
        except Exception as e:
            print(f'runtime error during env.step: {e}')
            return

        if args.export_stl:
            outname = f"{args.export_stl}.stl"
            try:
                env.export(type='stl', filename=outname)
                print(f'exported combined actions to {outname}')
            except Exception as e:
                print(f'failed to export STL: {e}')


if __name__ == '__main__':
    main()
