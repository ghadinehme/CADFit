#!/usr/bin/env python3
"""
Iterative CAD reconstruction using pre-computed analysis results.

This script uses pre-computed results from analyze_loop_translation.py and
analyze_revolve_axis.py to avoid the expensive search phase, then applies
iterative refinement similar to process_all_stls_iterative.py.

Usage:
    python run_pipeline.py [--num-workers N] [--alpha A]
"""
import os
import re
from random import random
import sys
import glob
import time
import json
import argparse
import subprocess
import numpy as np
import trimesh
from pathlib import Path

# Import matplotlib / pyexpat-dependent modules BEFORE anything that loads
# OCP/cadquery — those bundle a libexpat that breaks Python's pyexpat if
# pulled in first.
import matplotlib  # noqa: F401
import xml.parsers.expat  # noqa: F401

from iou import normalize_mesh, mesh_iou_cpu_boolean
from greedy_search import run_greedy_search
from refine_edges import optimize_edges, render_refined_script, load_solid_from_script


def _drop_nested_entries(entries, plane_dist_tol=1e-3, contain_frac=0.9):
    """Drop entries whose outer loop is contained in another entry's outer loop
    on the same plane. An entry's "outer loop" is the largest-area loop in
    `loops_3d`. Two entries are co-planar iff their normals are parallel (or
    anti-parallel) and the origin offset is perpendicular to that normal
    within `plane_dist_tol`.
    """
    if len(entries) < 2:
        return entries
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return entries

    outers_world = []
    outer_areas = []
    origins = []
    normals = []
    for e in entries:
        loops = e.get('loops_3d') or []
        if not loops:
            outers_world.append(None); outer_areas.append(0); origins.append(None); normals.append(None)
            continue
        areas = e.get('loop_areas') or []
        if areas and len(areas) == len(loops):
            i_outer = int(np.argmax(areas))
        else:
            i_outer = max(range(len(loops)), key=lambda i: len(loops[i]))
        loop = np.array(loops[i_outer], dtype=float)
        if loop.shape[0] > 2 and np.allclose(loop[0], loop[-1]):
            loop = loop[:-1]
        outers_world.append(loop)
        outer_areas.append(float(areas[i_outer]) if areas and len(areas) == len(loops) else 0.0)
        origins.append(np.array(e.get('origin', [0, 0, 0]), dtype=float))
        normals.append(np.array(e.get('normal', [0, 0, 1]), dtype=float))

    keep = [True] * len(entries)
    n = len(entries)
    for i in range(n):
        if outers_world[i] is None or not keep[i]:
            continue
        ni = normals[i]
        ni_n = ni / (np.linalg.norm(ni) + 1e-12)
        # Build a 2D basis on plane i
        if abs(ni_n[2]) < 0.9:
            u = np.cross(ni_n, [0, 0, 1.0])
        else:
            u = np.cross(ni_n, [1.0, 0, 0])
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(ni_n, u)
        oi = origins[i]
        try:
            poly_i = Polygon((outers_world[i] - oi) @ np.column_stack([u, v]))
            if not poly_i.is_valid:
                poly_i = poly_i.buffer(0)
            if poly_i.is_empty or poly_i.area <= 0:
                continue
        except Exception:
            continue
        for j in range(n):
            if i == j or not keep[j] or outers_world[j] is None:
                continue
            # Co-planarity check
            nj = normals[j]; nj_n = nj / (np.linalg.norm(nj) + 1e-12)
            if abs(abs(np.dot(ni_n, nj_n)) - 1.0) > 1e-3:
                continue
            if abs(np.dot(origins[j] - oi, ni_n)) > plane_dist_tol:
                continue
            # Smaller-area only — keep the bigger one
            if outer_areas[j] >= outer_areas[i] - 1e-9:
                continue
            pts_j_local = (outers_world[j] - oi) @ np.column_stack([u, v])
            try:
                poly_j = Polygon(pts_j_local)
                if not poly_j.is_valid:
                    poly_j = poly_j.buffer(0)
                if poly_j.is_empty:
                    continue
                inter = poly_j.intersection(poly_i).area
                if inter / max(poly_j.area, 1e-12) >= contain_frac:
                    keep[j] = False
            except Exception:
                continue
    return [e for k, e in zip(keep, entries) if k]


# Resolve subprocess scripts from this file's directory so cadfit_release
# is exercised end-to-end regardless of where the pipeline was launched from.
_RELEASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _release_script(name: str) -> str:
    return os.path.join(_RELEASE_DIR, name)


def _release_env():
    """env with PYTHONPATH=_RELEASE_DIR so subprocess scripts import their
    siblings (cadquery_ops, iou, etc.) from cadfit_release/ rather than cwd."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _RELEASE_DIR + (os.pathsep + existing if existing else "")
    return env


def run_command(cmd, description="", timeout=None):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, timeout=timeout, env=_release_env())
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ SUCCESS ({duration:.1f}s)")
            if result.stdout:
                print(result.stdout[-500:])  # Last 500 chars
            return True
        else:
            print(f"✗ FAILED ({duration:.1f}s)")
            if result.stderr:
                print(result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"✗ TIMEOUT after {duration:.1f}s")
        return False
    except Exception as e:
        duration = time.time() - start_time
        print(f"✗ ERROR ({duration:.1f}s): {e}")
        return False


def compute_mesh_volume(mesh, normalize=False):
    """Compute volume of a mesh"""
    try:
        if isinstance(mesh, str):
            mesh = trimesh.load(mesh, force='mesh')
        if normalize:
            mesh = normalize_mesh(mesh)
        return abs(mesh.volume)
    except Exception as e:
        print(f"Warning: Could not compute volume: {e}")
        return 0.0

def watertight(mesh, stl_path):
    if not mesh.is_watertight:
        print(f"⚠️  Mesh is not watertight, applying cleanup...")
        try:
            import pymeshlab
            
            # Create temporary path for cleaned mesh
            temp_cleaned_path = stl_path + ".tmp_cleaned.stl"
            
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(stl_path)
            
            ms.meshing_remove_duplicate_vertices()
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_remove_null_faces()
            
            # Save cleaned mesh
            ms.save_current_mesh(temp_cleaned_path)
            
            # Replace original with cleaned version
            import shutil
            shutil.move(temp_cleaned_path, stl_path)
            
            # Reload to verify
            mesh = trimesh.load(stl_path, force='mesh')
            if mesh.is_watertight:
                print(f"✓ Mesh repaired and is now watertight")
            else:
                print(f"⚠️  Mesh still not watertight after cleanup")
        except ImportError:
            print(f"⚠️  pymeshlab not available for mesh repair")
        except Exception as e:
            print(f"⚠️  Mesh repair failed: {e}")
    else:
        print(f"✓ Mesh is watertight")
    return mesh
    

def compute_difference_mesh(mesh_a_path, mesh_b_path, output_path, normalize_a=False, normalize_b=False):
    """Compute difference A - B of two meshes and remove surface-like components"""
    try:
        mesh_a = trimesh.load(mesh_a_path, force='mesh')
        mesh_b = trimesh.load(mesh_b_path, force='mesh')

        if normalize_a:
            mesh_a = normalize_mesh(mesh_a)
        if normalize_b:
            mesh_b = normalize_mesh(mesh_b)

        mesh_a = watertight(mesh_a, stl_path=mesh_a_path)
        mesh_b = watertight(mesh_b, stl_path=mesh_b_path)
        
        # Compute difference (A - B)
        difference = mesh_a.difference(mesh_b)
        
        if difference is not None and hasattr(difference, 'export'):
            # Filter out surface-like components
            if hasattr(difference, 'split'):
                components = difference.split()
                volume_components = [c for c in components if abs(c.volume) > 0.01]
                
                if volume_components:
                    if len(volume_components) == 1:
                        difference = volume_components[0]
                    else:
                        difference = trimesh.util.concatenate(volume_components)
            
            difference.export(output_path)
            return True
        return False
    except Exception as e:
        print(f"Error computing difference: {e}")
        return False


def compute_union_mesh(mesh_a_path, mesh_b_path, output_path):
    """Compute union of two meshes"""
    try:
        mesh_a = trimesh.load(mesh_a_path, force='mesh')
        mesh_b = trimesh.load(mesh_b_path, force='mesh')
        
        # Compute union
        union = mesh_a.union(mesh_b)
        
        if union is not None and hasattr(union, 'export'):
            union.export(output_path)
            return True
        return False
    except Exception as e:
        print(f"Error computing union: {e}")
        return False


def combine_cadquery_scripts(base_script_path, operation_script_path, output_script_path, operation='cut'):
    """
    Combine two CadQuery scripts using cut or union operation.
    
    Args:
        base_script_path: Path to the base CadQuery script (solid_n)
        operation_script_path: Path to the operation CadQuery script (solid_m)
        output_script_path: Path to save the combined script
        operation: 'cut' for solid_n.cut(solid_m) or 'union' for solid_n.union(solid_m)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read both scripts
        with open(base_script_path, 'r') as f:
            base_lines = f.readlines()
        
        with open(operation_script_path, 'r') as f:
            operation_lines = f.readlines()
        
        # Process base script - extract body and find result variable
        base_body = []
        base_result_var = "result"
        skip_imports = True
        
        for i, line in enumerate(base_lines):
            # Skip imports at the beginning
            if skip_imports and (line.strip().startswith('import ') or line.strip() == ''):
                if line.strip() != '' or i < 3:  # Skip first few lines including imports
                    continue
                else:
                    skip_imports = False
                    continue
            
            # Stop at export or print statements
            if '.exportStl' in line or line.strip().startswith('print('):
                break
            
            # Track the last result variable assignment
            if ' = ' in line and '.val().exportStl' not in line:
                stripped = line.strip()
                if stripped.startswith('final_result'):
                    base_result_var = stripped.split('=')[0].strip()
                elif stripped.startswith('result'):
                    base_result_var = stripped.split('=')[0].strip()
            
            base_body.append(line)
        
        # Process operation script - extract body and find result variable
        operation_body = []
        operation_result_var = "result"
        skip_imports = True
        in_union_block = False
        
        for i, line in enumerate(operation_lines):
            # Skip imports at the beginning
            if skip_imports and (line.strip().startswith('import ') or line.strip() == ''):
                if line.strip() != '' or i < 3:
                    continue
                else:
                    skip_imports = False
                    continue
            
            # Stop at export or print statements
            if '.exportStl' in line or line.strip().startswith('print('):
                break
            
            # Skip comments about union/export
            if line.strip().startswith('#') and ('Union all solids' in line or 'Export' in line):
                in_union_block = True
                continue
            
            # Track the result variable
            if ' = ' in line and '.val().exportStl' not in line:
                stripped = line.strip()
                if stripped.startswith('result'):
                    operation_result_var = stripped.split('=')[0].strip()
            
            operation_body.append(line)
        
        # Rename variables in operation script to avoid conflicts
        operation_body_renamed = []
        
        for line in operation_body:
            renamed_line = line
            # Add _op suffix to main variables
            renamed_line = renamed_line.replace('solid_', 'solid_op_')
            renamed_line = renamed_line.replace('plane_', 'plane_op_')
            renamed_line = renamed_line.replace('sketch_', 'sketch_op_')
            
            # Fix result variable references more carefully
            if 'result_op_op' in renamed_line:
                renamed_line = renamed_line.replace('result_op_op', 'result_op')
            elif operation_result_var in renamed_line:
                # Use regex-like replacement to handle result variable properly
                import re
                # Replace 'result =' with 'result_op ='
                renamed_line = re.sub(r'\bresult\s*=', 'result_op =', renamed_line)
                # Replace 'result.union' or 'result.cut' with 'result_op.union' or 'result_op.cut'
                renamed_line = re.sub(r'\bresult\.', 'result_op.', renamed_line)
            
            operation_body_renamed.append(renamed_line)
        
        # Build combined script
        combined_lines = []
        
        # Add imports
        combined_lines.append("import cadquery as cq\n")
        combined_lines.append("import math\n")
        combined_lines.append("\n")
        
        # Add base script body
        combined_lines.extend(base_body)
        
        # Add result variable assignment if not already present
        if base_result_var == "result" and not any("result =" in line for line in base_body):
            # Find the last solid variable in base_body
            last_solid_var = None
            for line in reversed(base_body):
                if 'solid_' in line and ' = ' in line and '.extrude' in line:
                    last_solid_var = line.split('=')[0].strip()
                    break
            
            if last_solid_var:
                combined_lines.append(f"result = {last_solid_var}\n")
            elif base_body:
                # Look for result assignment that might have been excluded
                for line in base_lines:
                    if line.strip().startswith('result =') and '.exportStl' not in line:
                        combined_lines.append(line)
                        break
        
        # Add separator comment
        combined_lines.append("\n# ===== Operation Script =====\n")
        
        # Add operation script body
        combined_lines.extend(operation_body_renamed)
        
        # Add the cut/union operation
        combined_lines.append(f"\n# ===== Combine using {operation} =====\n")
        if operation == 'cut':
            combined_lines.append(f"final_result = {base_result_var}.cut(result_op)\n")
        else:  # union — use `.add()` (Compound) and let manifold3d clean it.
            combined_lines.append(f"final_result = {base_result_var}.add(result_op)\n")

        
        # Add export
        stl_output = output_script_path.replace('.py', '.stl')
        combined_lines.append(f"cq.exporters.export(final_result, '{stl_output}')\n")

        combined_lines.append(f"print('Successfully exported combined result to {stl_output}')\n")
        
        # Write combined script
        with open(output_script_path, 'w') as f:
            f.writelines(combined_lines)
        
        return True
        
    except Exception as e:
        print(f"Error combining CadQuery scripts: {e}")
        import traceback
        traceback.print_exc()
        return False


def execute_cadquery_script_file(script_path, timeout=300):
    """Execute a CadQuery script file"""
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"CadQuery script execution timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"Error executing CadQuery script: {e}")
        return False


def refine_with_fillet_chamfer(src_script: str, src_stl: str, gt_stl: str,
                               work_dir: str, tag: str = "refine",
                               normalize: bool = True,
                               timeout: int = 600):
    """Per-edge fillet/chamfer refinement, run in a subprocess for crash isolation.

    OCP's `fillet` / `chamfer` can segfault on degenerate inputs, which would
    kill the main pipeline if run in-process. We invoke `refine_edges.py`
    as a separate Python process; if it crashes or runs over `timeout`, we
    log the failure and keep the unrefined solid.

    Returns `(new_iou, num_ops, new_script_path, new_stl_path)` on improvement,
    else `None`.
    """
    os.makedirs(work_dir, exist_ok=True)
    output_stl = f"{work_dir}/{tag}_refined.stl"
    output_log = f"{work_dir}/{tag}_ops.json"

    worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "refine_edges.py")
    cmd = [
        sys.executable, worker_path,
        "--base-script", src_script,
        "--gt-stl", gt_stl,
        "--work-dir", work_dir,
        "--output-stl", output_stl,
        "--output-log", output_log,
        "--operation-type", "both",
    ]
    if not normalize:
        cmd.append("--no-normalize")

    try:
        result = subprocess.run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [refine] worker exceeded {timeout}s timeout; reverting")
        return None
    except Exception as e:
        print(f"  [refine] worker launch failed: {e}; reverting")
        return None

    if result.returncode != 0:
        print(f"  [refine] worker exited with code {result.returncode} "
              "(possibly OCP segfault); reverting")
        return None

    if not os.path.exists(output_log):
        return None

    try:
        with open(output_log) as f:
            log_data = json.load(f)
    except Exception as e:
        print(f"  [refine] could not parse ops log: {e}")
        return None

    ops = log_data.get("ops", [])
    if not ops:
        return None  # worker decided refinement didn't help

    if not (os.path.exists(output_stl) and os.path.getsize(output_stl) > 0):
        return None

    # Sanity-check on disk: did the refinement actually beat the base?
    disk_iou = mesh_iou_cpu_boolean(gt_stl, output_stl, normalize=normalize)
    base_iou = mesh_iou_cpu_boolean(gt_stl, src_stl, normalize=normalize)
    if disk_iou <= base_iou + 1e-5:
        print(f"  [refine] disk IoU {disk_iou:.6f} ≤ base {base_iou:.6f}; reverting")
        return None

    # Best-effort: emit a self-contained refined CadQuery script too.
    new_script = f"{work_dir}/{tag}_refined.py"
    ops_log_objs = [
        # Reconstruct EdgeOp-like tuples for the renderer.
        type("EdgeOpLite", (), {"edge_index": o["edge_index"],
                                "op_kind": o["op_kind"],
                                "size": o["size"]})()
        for o in ops
    ]
    if not render_refined_script(src_script, ops_log_objs, new_script, output_stl):
        new_script = src_script

    return disk_iou, len(ops), new_script, output_stl


def run_analysis_steps(stl_path, output_folder, n_slices=5, delta=0.05, normalize=True):
    """Run analysis steps to generate translation and revolve analysis"""
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Define paths
    planar_contours_path = f"{output_folder}/planar_contours.json"
    planar_loops_split_path = f"{output_folder}/planar_loops_split.json"
    axis_sketches_path = f"{output_folder}/axis_sketches.json"
    axis_loops_split_path = f"{output_folder}/axis_loops_split.json"
    loops_split_path = f"{output_folder}/loops_split.json"
    loops_split_revolve_path = f"{output_folder}/loops_split_revolve.json"
    translation_analysis_folder = f"{output_folder}/translation_analysis"
    revolve_analysis_folder = f"{output_folder}/revolve_analysis"

    # Check and repair mesh if not watertight
    print(f"\n🔧 Checking mesh integrity...")
    mesh = trimesh.load(stl_path, force='mesh')
    
    if not mesh.is_watertight:
        print(f"⚠️  Mesh is not watertight, applying cleanup...")
        try:
            import pymeshlab
            
            # Create temporary path for cleaned mesh
            temp_cleaned_path = stl_path + ".tmp_cleaned.stl"
            
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(stl_path)
            
            ms.meshing_remove_duplicate_vertices()
            ms.meshing_remove_duplicate_faces()
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_remove_null_faces()
            
            # Save cleaned mesh
            ms.save_current_mesh(temp_cleaned_path)
            
            # Replace original with cleaned version
            import shutil
            shutil.move(temp_cleaned_path, stl_path)
            
            # Reload to verify
            mesh = trimesh.load(stl_path, force='mesh')
            if mesh.is_watertight:
                print(f"✓ Mesh repaired and is now watertight")
            else:
                print(f"⚠️  Mesh still not watertight after cleanup")
        except ImportError:
            print(f"⚠️  pymeshlab not available for mesh repair")
        except Exception as e:
            print(f"⚠️  Mesh repair failed: {e}")
    else:
        print(f"✓ Mesh is watertight")
    
    # Step 1: Extract planar contours
    print(f"\n📐 Step 1/7: Extracting planar contours...")
    if normalize:
        cmd1 = [sys.executable, _release_script("extract_planar_contours.py"), stl_path, "--out", planar_contours_path, "--delta", str(delta), "--normalize"]
    else:
        cmd1 = [sys.executable, _release_script("extract_planar_contours.py"), stl_path, "--out", planar_contours_path, "--delta", str(delta)]
    if not run_command(cmd1, "Extract planar contours", timeout=300):
        return False
    
    # Step 2: Split planar contours loops
    print(f"\n🔄 Step 2/7: Splitting planar contours loops...")
    cmd2 = [sys.executable, _release_script("split_loops.py"), planar_contours_path, "--group-contained", "--out", planar_loops_split_path]
    if not run_command(cmd2, "Split planar loops", timeout=120):
        return False
    
    # Step 3: Extract axis-aligned sketches
    print(f"\n📏 Step 3/7: Extracting axis-aligned sketches...")
    if normalize:
        cmd3 = [sys.executable, _release_script("extract_sketches.py"), stl_path, "--out", axis_sketches_path, "--axis-slices", str(n_slices), "--normalize"]
    else:
        cmd3 = [sys.executable, _release_script("extract_sketches.py"), stl_path, "--out", axis_sketches_path, "--axis-slices", str(n_slices)]
    if not run_command(cmd3, "Extract axis-aligned sketches", timeout=300):
        return False
    
    # Step 4: Split axis-aligned sketches loops
    print(f"\n🔄 Step 4/7: Splitting axis-aligned sketches loops...")
    cmd4 = [sys.executable, _release_script("split_loops.py"), axis_sketches_path, "--group-contained", "--out", axis_loops_split_path]
    if not run_command(cmd4, "Split axis loops", timeout=120):
        print("failed")
    
    # Step 4b: Combine both split loop JSONs for extrusion
    print(f"\n🔗 Step 4b/7: Combining split loops from both sources...")
    if os.path.exists(planar_loops_split_path) and os.path.exists(axis_loops_split_path):
        with open(planar_loops_split_path, 'r') as f:
            planar_loops = json.load(f)
        with open(axis_loops_split_path, 'r') as f:
            axis_loops = json.load(f)

        combined_loops = planar_loops + axis_loops
        n_pre = len(combined_loops)
        combined_loops = _drop_nested_entries(combined_loops)

        with open(loops_split_path, 'w') as f:
            json.dump(combined_loops, f, indent=2)

        n_dropped = n_pre - len(combined_loops)
        print(f"✓ Combined {len(planar_loops)} planar + {len(axis_loops)} axis "
              f"→ {len(combined_loops)} ({n_dropped} dropped as nested)")
    elif os.path.exists(planar_loops_split_path):
        loops_split_path = planar_loops_split_path
    elif os.path.exists(axis_loops_split_path):
        loops_split_path = axis_loops_split_path
    else:
        with open(loops_split_path, 'w') as f:
            json.dump([], f, indent=2)
        
    
    # Step 4c: Create revolve-specific JSON (axis sketches without splitting)
    print(f"\n🔗 Step 4c/7: Preparing axis sketches for revolve detection (unsplit)...")
    try:
        with open(axis_sketches_path, 'r') as f:
            axis_data = json.load(f)
        
        # Save as loops_split_revolve.json
        with open(loops_split_revolve_path, 'w') as f:
            json.dump(axis_data, f, indent=2)
        
        print(f"✓ Saved {len(axis_data)} sketches for revolve analysis")
        
    except Exception as e:
        print(f"✗ Error preparing revolve data: {e}")
        return False
    
    # Step 5: Run translation analysis
    print(f"\n📊 Step 5/7: Running translation analysis...")
    # CADFIT_INNER_WORKERS overrides everything (useful when wrapping the
    # pipeline in an outer process pool — keep inner at 1 to avoid
    # over-subscription). Otherwise fall back to the auto-sized default.
    if "CADFIT_INNER_WORKERS" in os.environ:
        inner_workers = max(1, int(os.environ["CADFIT_INNER_WORKERS"]))
    else:
        inner_workers = max(4, os.cpu_count() // max(int(os.environ.get("CADFIT_MAX_PARALLEL", "2")), 1))
    cmd5 = [sys.executable, _release_script("analyze_loop_translation.py"), loops_split_path, stl_path,
            "--output-folder", translation_analysis_folder, "--num-workers", str(inner_workers), "--no-plots"]
    if normalize:
        cmd5.append("--normalize")
    if not run_command(cmd5, "Translation analysis", timeout=3600):
        return False
    
    # Step 6: Run revolve analysis
    print(f"\n🔄 Step 6/7: Running revolve analysis...")
    cmd6 = [sys.executable, _release_script("analyze_revolve_axis.py"), loops_split_revolve_path, stl_path,
            "--output-folder", revolve_analysis_folder,
            "--num-workers", str(inner_workers),
            "--angle-min", "0", "--angle-max", "90", "--step", "90", "--cd-threshold", "0.01", "--no-plots"]
    if normalize:
        cmd6.append("--normalize")
    if not run_command(cmd6, "Revolve analysis", timeout=3600):
        return False
    
    # Step 7: Run greedy CAD reconstruction with precomputed operations
    print(f"\n🏗️  Step 7/7: Running greedy CAD reconstruction with precomputed operations...")
    cmd7 = [sys.executable, _release_script("greedy_search.py"),
            loops_split_path, loops_split_revolve_path, stl_path,
            translation_analysis_folder, revolve_analysis_folder,
            "--output-folder", output_folder,
            "--num-workers", str(inner_workers)]
    if normalize:
        cmd7.append("--normalize")
    if not run_command(cmd7, "Greedy CAD reconstruction", timeout=3000):
        return False
    
    return True


def _decimate_input_if_needed(stl_path: str, output_folder: str,
                               target_faces: int = 50_000) -> str:
    """If the input STL has more than `target_faces`, write a decimated copy
    into `output_folder/decimated_input.stl` and return that path. Otherwise
    return the original path unchanged. All downstream sketch-extraction
    steps then operate on a much smaller mesh, with massive runtime savings
    on dense CAD STLs (100k–900k faces).

    The decimated mesh is used for sketch extraction only; the IoU function
    does its own alpha-wrap on whatever STL it's given, so accuracy of the
    final IoU is unaffected as long as decimation preserves overall shape.
    """
    try:
        m = trimesh.load(stl_path, force='mesh')
    except Exception as e:
        print(f"⚠️  Could not load input STL for decimation check: {e}")
        return stl_path
    if len(m.faces) <= target_faces:
        return stl_path
    try:
        import pymeshlab
        out_path = os.path.join(output_folder, "decimated_input.stl")
        os.makedirs(output_folder, exist_ok=True)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=np.asarray(m.vertices, dtype=np.float64),
            face_matrix=np.asarray(m.faces, dtype=np.int32),
        ))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preservenormal=True,
            preserveboundary=True,
        )
        ms.save_current_mesh(out_path)
        print(f"📉 Decimated input from {len(m.faces):,} → {target_faces:,} faces "
              f"→ {out_path}")
        return out_path
    except Exception as e:
        print(f"⚠️  Decimation failed ({e}); using original {len(m.faces):,}-face mesh")
        return stl_path


def process_single_stl(stl_path, alpha=0.01, folder_name="search_greedy_parallel",
                                   over_threshold=0.02, under_threshold=0.01, max_iterations=5,
                                   apply_fillet_chamfer=False, keep_history=False):
    """Process a single STL file with iterative refinement using precomputed analysis"""
    stl_name = os.path.basename(stl_path)
    stl_id = os.path.splitext(stl_name)[0]

    print(f"\n{'#'*80}")
    print(f"PROCESSING: {stl_name}")
    print(f"STL ID: {stl_id}")
    print(f"Full path: {stl_path}")
    print(f"{'#'*80}")

    total_start_time = time.time()

    # Get the output folder path
    output_folder = f"{folder_name}/{stl_id}"
    os.makedirs(output_folder, exist_ok=True)

    # Decimate the input STL once if it's too dense — every downstream step
    # then loads this smaller mesh. The original stl_path is still used as
    # the canonical ground truth in the final IoU report.
    original_stl_path = stl_path
    stl_path = _decimate_input_if_needed(stl_path, output_folder)

    # If the input STL is non-watertight or composed of multiple bodies, save
    # the alpha-wrap surface alongside the outputs — that is what `iou.py`
    # actually feeds into manifold3d, so inspecting it is the cleanest way to
    # diagnose IoU failures.
    alphawrap_path = None
    try:
        _m = trimesh.load(stl_path, force='mesh')
        _multi = getattr(_m, 'body_count', 1) > 1
        if (not _m.is_watertight) or _multi:
            from iou import _alpha_wrap
            wrapped = _alpha_wrap(_m)
            alphawrap_path = f"{output_folder}/{stl_id}_alphawrap.stl"
            wrapped.export(alphawrap_path)
            print(f"⚠️  Input mesh is not single-body watertight "
                  f"(watertight={_m.is_watertight}, bodies={getattr(_m, 'body_count', 1)}); "
                  f"saved alpha-wrap → {alphawrap_path}")
    except Exception as e:
        print(f"⚠️  Could not check / write alpha-wrap: {e}")

    # Initial reconstruction (steps 1-7)
    print(f"\n🚀 INITIAL RECONSTRUCTION")
    print(f"{'='*80}")

    if not run_analysis_steps(stl_path, output_folder, normalize=True):
        print(f"✗ Failed to complete analysis steps for {stl_name}")
        return
    
    # Get the reconstructed mesh path
    reconstructed_path = f"{output_folder}/best_greedy_parallel.stl"
    
    if not os.path.exists(reconstructed_path):
        print(f"✗ Reconstructed mesh not found: {reconstructed_path}")
        return
    
    current_reconstruction = reconstructed_path
    current_cadquery_script = f"{output_folder}/best_greedy_parallel.py"
    iteration = 0
    
    # Check if initial CadQuery script exists
    if not os.path.exists(current_cadquery_script):
        print(f"✗ CadQuery script not found: {current_cadquery_script}")
        return
    
    # Iterative refinement loop
    print(f"\n🔧 REFINEMENT PHASE: Removing over-reconstruction and adding under-reconstruction")
    print(f"{'='*80}")

    under_volume = 10
    
    while iteration < max_iterations and under_volume > under_threshold:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")
        
        # Compute over-reconstruction (reconstructed - ground_truth)
        over_recon_path = f"{output_folder}/over_reconstruction_{iteration}.stl"
        
        print(f"\nComputing over-reconstruction...")
        if not compute_difference_mesh(current_reconstruction, stl_path, over_recon_path, 
                                      normalize_a=False, normalize_b=True):
            print("✗ Could not compute over-reconstruction")
            break
        
        if not os.path.exists(over_recon_path):
            print("✗ Over-reconstruction file not created")
            break
        
        over_volume = compute_mesh_volume(over_recon_path)
        gt_mesh_volume = compute_mesh_volume(stl_path, normalize=True)
        over_ratio = over_volume / gt_mesh_volume if gt_mesh_volume > 0 else 0
        print(f"Over-reconstruction volume: {over_ratio:.6f}")

        
        
        #     break
        
        # Handle over-reconstruction if volume > threshold
        if over_volume > over_threshold:
            print(f"\n🔨 Processing over-reconstruction (volume = {over_volume:.6f} > {over_threshold})")
            
            # Run analysis on over-reconstruction
            over_output = f"{output_folder}/refine_over_{iteration}"
            
            if not run_analysis_steps(over_recon_path, over_output, normalize=False):
                print("✗ Failed to analyze over-reconstruction")
                break
            
            cutting_solid_path = f"{over_output}/best_greedy_parallel.stl"
            cutting_script_path = f"{over_output}/best_greedy_parallel.py"
            
            if not os.path.exists(cutting_solid_path):
                print("✗ Cutting solid not created")
                break
            
            # Combine scripts with cut operation
            cut_script_path = f"{output_folder}/reconstruction_cut_{iteration}.py"
            
            if not combine_cadquery_scripts(current_cadquery_script, cutting_script_path,
                                          cut_script_path, operation='cut'):
                print("✗ Failed to combine scripts for cut")
                break
            
            # Execute combined script
            if not execute_cadquery_script_file(cut_script_path, timeout=300):
                print("✗ Failed to execute cut script")
                break
            
            # Update current reconstruction
            new_reconstruction = f"{output_folder}/reconstruction_cut_{iteration}.stl"
            
            if os.path.exists(new_reconstruction):
                current_reconstruction = new_reconstruction
                current_cadquery_script = cut_script_path
                print(f"✓ Applied cut operation")
            else:
                print("✗ Cut operation did not produce output")
                break

        # Compute under-reconstruction (ground_truth - reconstructed)
        under_recon_path = f"{output_folder}/under_reconstruction_{iteration}.stl"
        
        print(f"\nComputing under-reconstruction...")
        if not compute_difference_mesh(stl_path, current_reconstruction, under_recon_path,
                                      normalize_a=True, normalize_b=False):
            print("✗ Could not compute under-reconstruction")
            break
        
        if not os.path.exists(under_recon_path):
            print("✗ Under-reconstruction file not created")
            break
        
        under_volume = compute_mesh_volume(under_recon_path)
        gt_mesh_volume = compute_mesh_volume(stl_path, normalize=True)
        under_ratio = under_volume / gt_mesh_volume if gt_mesh_volume > 0 else 0
        print(f"Under-reconstruction volume: {under_ratio:.6f}")
        
        # Handle under-reconstruction if volume > threshold
        if under_volume > under_threshold:
            print(f"\n➕ Processing under-reconstruction (volume = {under_volume:.6f} > {under_threshold})")
            
            # Run analysis on under-reconstruction
            under_output = f"{output_folder}/refine_under_{iteration}"
            
            if not run_analysis_steps(under_recon_path, under_output, normalize=False):
                print("✗ Failed to analyze under-reconstruction")
                # Continue anyway to try next iteration
                continue
            
            additional_solid_path = f"{under_output}/best_greedy_parallel.stl"
            additional_script_path = f"{under_output}/best_greedy_parallel.py"
            
            if not os.path.exists(additional_solid_path):
                print("✗ Additional solid not created")
                continue
            
            # Combine scripts with union operation
            union_script_path = f"{output_folder}/reconstruction_union_{iteration}.py"
            
            if not combine_cadquery_scripts(current_cadquery_script, additional_script_path,
                                          union_script_path, operation='union'):
                print("✗ Failed to combine scripts for union")
                continue
            
            # Execute combined script
            if not execute_cadquery_script_file(union_script_path, timeout=300):
                print("✗ Failed to execute union script")
                continue
            
            # Update current reconstruction
            new_reconstruction = f"{output_folder}/reconstruction_union_{iteration}.stl"
            
            if os.path.exists(new_reconstruction):
                current_reconstruction = new_reconstruction
                current_cadquery_script = union_script_path
                print(f"✓ Applied union operation")
            else:
                print("✗ Union operation did not produce output")
                continue

        # Fillet / chamfer refinement on the current solid.
        if apply_fillet_chamfer:
            print(f"\n✨ Fillet/chamfer refinement (iteration {iteration})")
            refine_dir = f"{output_folder}/refine_fc_{iteration}"
            try:
                refined = refine_with_fillet_chamfer(
                    src_script=current_cadquery_script,
                    src_stl=current_reconstruction,
                    gt_stl=stl_path,
                    work_dir=refine_dir,
                    tag=f"iter{iteration}",
                    normalize=True,
                )
            except Exception as e:
                print(f"  ✗ Fillet/chamfer refinement raised: {e}")
                refined = None
            if refined is not None:
                best_iou, num_ops, new_script, new_stl = refined
                current_reconstruction = new_stl
                current_cadquery_script = new_script
                print(f"  ✓ Applied {num_ops} edge op(s) → IoU {best_iou:.6f}")
            else:
                print(f"  No fillet/chamfer improvement; keeping current solid")

    # Save final result. The saved .py is for human inspection — strip any
    # cq.exporters.export(...) and trailing print(...) calls so the file is
    # just the build sequence.
    final_output = f"{output_folder}/best_greedy_parallel_iterative.stl"
    final_script_output = f"{output_folder}/best_greedy_parallel_iterative.py"

    def _copy_script_stripped(src, dst):
        with open(src) as f:
            lines = f.readlines()
        out = [ln for ln in lines
               if 'cq.exporters.export' not in ln
               and not ln.lstrip().startswith('print(')]
        with open(dst, 'w') as f:
            f.writelines(out)

    import shutil
    if current_reconstruction != reconstructed_path:
        shutil.copy(current_reconstruction, final_output)
        _copy_script_stripped(current_cadquery_script, final_script_output)
        print(f"\n✓ Saved final result: {final_output}")
    else:
        shutil.copy(reconstructed_path, final_output)
        _copy_script_stripped(current_cadquery_script, final_script_output)
        print(f"\n✓ No refinement needed, saved initial result: {final_output}")
    
    # Compute final IoU via CPU mesh booleans
    print(f"\n📊 COMPUTING FINAL IOU (CPU mesh booleans)")
    print(f"{'='*80}")

    iou_start = time.time()
    # Final IoU is reported against the ORIGINAL (un-decimated) GT — the
    # decimated copy was only used to accelerate sketch extraction.
    final_iou = mesh_iou_cpu_boolean(original_stl_path, final_output, normalize=True, verbose=True)

    # Safety net: a candidate from greedy's evaluated_operations may score
    # higher than the greedy combination when evaluated against the
    # un-decimated GT. If so, swap in the best candidate.
    eval_path = os.path.join(output_folder, "evaluated_operations.json")
    if os.path.exists(eval_path):
        try:
            with open(eval_path, 'r') as _f:
                evaluated = json.load(_f)
        except Exception:
            evaluated = []
        best_candidate = None
        best_candidate_iou = final_iou
        for op in evaluated:
            stl_p = op.get('stl_path')
            if not stl_p or not os.path.exists(stl_p):
                continue
            try:
                iou = mesh_iou_cpu_boolean(original_stl_path, stl_p, normalize=True, verbose=False)
            except Exception:
                continue
            if iou is not None and iou > best_candidate_iou + 1e-6:
                best_candidate_iou = iou
                best_candidate = op
        if best_candidate is not None:
            import shutil
            print(f"⚠️  Single candidate beats greedy: {best_candidate_iou:.6f} > {final_iou:.6f} — swapping in {best_candidate.get('script_path', '?')}")
            shutil.copy2(best_candidate['stl_path'], final_output)
            sp = best_candidate.get('script_path')
            if sp and os.path.exists(sp):
                _copy_script_stripped(sp, final_script_output)
            final_iou = best_candidate_iou
    iou_duration = time.time() - iou_start

    print(f"✓ Final IoU: {final_iou:.6f}  (CPU boolean took {iou_duration:.2f}s)")

    total_duration = time.time() - total_start_time

    # Save IoU to JSON file
    iou_output_path = f"{output_folder}/final_iou.json"
    iou_data = {
        "final_iou": final_iou,
        "stl_id": stl_id,
        "ground_truth": original_stl_path,
        "reconstruction": final_output,
        "num_iterations": iteration,
        "duration": total_duration
    }
    with open(iou_output_path, 'w') as f:
        json.dump(iou_data, f, indent=2)
    print(f"✓ IoU data saved to {iou_output_path}")

    # Clean up intermediate artifacts unless history is requested.
    if not keep_history:
        greedy_stl = f"{output_folder}/best_greedy_parallel.stl"
        greedy_py = f"{output_folder}/best_greedy_parallel.py"
        greedy_iou_json = f"{output_folder}/greedy_final_iou.json"
        _cleanup_output_folder(output_folder, final_output, final_script_output,
                               iou_output_path, alphawrap_path,
                               greedy_stl, greedy_py, greedy_iou_json)

    print(f"\n🎉 COMPLETED: {stl_name} in {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    if final_iou is not None:
        print(f"   Final IoU: {final_iou:.6f}")

    return


def _cleanup_output_folder(output_folder, *keep_paths):
    """Delete every file in `output_folder` (recursively) except the named
    artifacts. Removes JSONs, intermediate STLs/scripts, analysis caches —
    leaves only the final STL, .py, and final_iou.json."""
    keep_abs = {os.path.abspath(p) for p in keep_paths if p}
    try:
        for root, dirs, files in os.walk(output_folder, topdown=False):
            for name in files:
                p = os.path.abspath(os.path.join(root, name))
                if p in keep_abs:
                    continue
                try:
                    os.unlink(p)
                except Exception:
                    pass
            for name in dirs:
                d = os.path.join(root, name)
                try:
                    if not os.listdir(d):
                        os.rmdir(d)
                except Exception:
                    pass
    except Exception as e:
        print(f"⚠️  Cleanup failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='CADFit pipeline — reconstruct every STL in a folder.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('input_folder',
                        help='Folder containing input STLs to reconstruct (non-recursive).')
    parser.add_argument('--output-folder', default=None,
                        help='Folder for per-STL outputs. Defaults to "<input>_runs/".')
    parser.add_argument('--alpha', type=float, default=0.01, help='Alpha for IoU computation')
    parser.add_argument('--over-threshold', type=float, default=0.02,
                        help='Volume threshold for over-reconstruction')
    parser.add_argument('--under-threshold', type=float, default=0.02,
                        help='Volume threshold for under-reconstruction')
    parser.add_argument('--max-iterations', type=int, default=1,
                        help='Max residual-refinement iterations')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip STLs whose final_iou.json already exists')
    parser.add_argument('--start-from', type=str,
                        help='Skip STLs alphabetically before this filename')
    parser.add_argument('--limit', type=int, help='Process at most this many STLs')
    parser.add_argument('--fillet-chamfer', action='store_true',
                        help='Enable the per-iteration fillet/chamfer refinement step (off by default)')
    parser.add_argument('--keep-history', action='store_true',
                        help='Keep all intermediate files in the output folder. '
                             'By default, only the final STL, .py, and final_iou.json are kept.')
    args = parser.parse_args()

    in_dir = args.input_folder.rstrip('/')
    if not os.path.isdir(in_dir):
        print(f"Input folder not found: {in_dir}")
        return 1
    out_dir = args.output_folder or f"{os.path.basename(in_dir)}_runs"
    os.makedirs(out_dir, exist_ok=True)

    stl_files = sorted(glob.glob(os.path.join(in_dir, "*.stl")))
    if not stl_files:
        print(f"No STL files found in {in_dir}/")
        return 1
    print(f"🔍 Found {len(stl_files)} STL files in {in_dir}/")
    print(f"📁 Output → {out_dir}/")

    if args.start_from:
        try:
            start_idx = [os.path.basename(f) for f in stl_files].index(args.start_from)
            stl_files = stl_files[start_idx:]
            print(f"Starting from {args.start_from} ({len(stl_files)} remaining)")
        except ValueError:
            print(f"Start file {args.start_from} not found in list")
            return 1

    if args.limit:
        stl_files = stl_files[:args.limit]
        print(f"Limited to {args.limit} files")

    successful, failed, failed_files = 0, 0, []
    overall_start_time = time.time()

    for i, stl_path in enumerate(stl_files, 1):
        stl_name = os.path.basename(stl_path)
        stl_id = os.path.splitext(stl_name)[0]
        final_iou_json = f"{out_dir}/{stl_id}/final_iou.json"

        if args.skip_existing and os.path.exists(final_iou_json):
            print(f"\n[{i}/{len(stl_files)}] ⏭️  SKIPPING {stl_name} (cached)")
            successful += 1
            continue

        print(f"\n{'='*80}")
        print(f"[{i}/{len(stl_files)}] Processing: {stl_name}")
        print(f"{'='*80}")

        try:
            process_single_stl(
                stl_path=stl_path,
                folder_name=out_dir,
                alpha=args.alpha,
                over_threshold=args.over_threshold,
                under_threshold=args.under_threshold,
                max_iterations=args.max_iterations,
                apply_fillet_chamfer=args.fillet_chamfer,
                keep_history=args.keep_history,
            )
            successful += 1
        except Exception as e:
            print(f"\n✗ EXCEPTION while processing {stl_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            failed_files.append(stl_name)

    total_duration = time.time() - overall_start_time
    print(f"\n{'='*80}\n🏁 FINAL SUMMARY\n{'='*80}")
    print(f"Total files: {len(stl_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_duration/60:.1f} minutes")
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    print(f"\n🎯 SUCCESS RATE: {successful}/{len(stl_files)} "
          f"({100*successful/len(stl_files):.1f}%)")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
