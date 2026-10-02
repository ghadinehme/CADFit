#!/usr/bin/env python3
"""
Parallel mirror of run_pipeline.py.

Identical reconstruction algorithm to ``run_pipeline.py`` — it imports and
reuses every step from that module verbatim. 

Usage:
    python run_pipeline_parallel.py INPUT_FOLDER --n-procs P --n-threads N [...]
    python run_pipeline_parallel.py INPUT_FOLDER --n-procs P --n-threads N --verbose
"""
import os
import sys
import glob
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from run_pipeline import process_single_stl


_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _pin_threads(n_threads: int):
    """Force the inner worker count and BLAS thread budget for this process and
    everything it spawns. ``run_pipeline`` reads ``CADFIT_INNER_WORKERS`` to set
    each analysis subprocess's ``--num-workers``; the BLAS vars cap native
    threading inside any subprocess that does not already override them."""
    n = str(max(1, int(n_threads)))
    os.environ["CADFIT_INNER_WORKERS"] = n
    for var in _BLAS_ENV_VARS:
        os.environ[var] = n


def _silence_fds():

    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)   # stdout
    os.dup2(devnull, 2)   # stderr
    os.close(devnull)

    sys.stdout = open(os.devnull, "w")
    sys.stderr = sys.stdout


def _worker_init(n_threads: int, silent: bool):
    """ProcessPoolExecutor initializer: runs once per worker process."""
    _pin_threads(n_threads)
    if silent:
        _silence_fds()


def _process_one(stl_path: str, kwargs: dict):
    """Top-level (picklable) wrapper around process_single_stl that returns a
    status tuple instead of raising, so one bad STL doesn't kill the pool."""
    stl_name = os.path.basename(stl_path)
    try:
        process_single_stl(stl_path=stl_path, **kwargs)
        return stl_name, True, None
    except Exception:
        import traceback
        return stl_name, False, traceback.format_exc()


def main():
    parser = argparse.ArgumentParser(
        description='CADFit pipeline (parallel mirror) — reconstruct every STL '
                    'in a folder, N STLs at a time via multiprocessing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('input_folder',
                        help='Folder containing input STLs to reconstruct (non-recursive).')
    parser.add_argument('--output-folder', default=None,
                        help='Folder for per-STL outputs. Defaults to "<input>_runs/".')
    # --- parallelism controls (the only knobs added vs run_pipeline.py) ---
    parser.add_argument('--n-procs', type=int, default=4,
                        help='Number of STLs to process concurrently (outer '
                             'ProcessPoolExecutor workers).')
    parser.add_argument('--n-threads', type=int, default=8,
                        help='Forced inner worker count per STL. Exported as '
                             'CADFIT_INNER_WORKERS and pinned onto OMP/OPENBLAS/'
                             'MKL/NUMEXPR/VECLIB thread env vars for every '
                             'internal subprocess.')
    # --- identical to run_pipeline.py ---
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
    parser.add_argument('--verbose', action='store_true',
                        help='Show the full per-STL pipeline output. By default '
                             'all worker/subprocess output is suppressed and only '
                             'a progress bar is shown.')
    args = parser.parse_args()

    silent = not args.verbose

    def info(msg):
        """Setup-time logging that is hidden in silent mode (tqdm.write keeps it
        from clobbering the progress bar when verbose)."""
        if not silent:
            print(msg)

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
    info(f"🔍 Found {len(stl_files)} STL files in {in_dir}/")
    info(f"📁 Output → {out_dir}/")

    if args.start_from:
        try:
            start_idx = [os.path.basename(f) for f in stl_files].index(args.start_from)
            stl_files = stl_files[start_idx:]
            info(f"Starting from {args.start_from} ({len(stl_files)} remaining)")
        except ValueError:
            print(f"Start file {args.start_from} not found in list")
            return 1

    if args.limit:
        stl_files = stl_files[:args.limit]
        info(f"Limited to {args.limit} files")

    n_procs = max(1, int(args.n_procs))
    n_threads = max(1, int(args.n_threads))
    info(f"⚙️  Parallelism: {n_procs} procs × {n_threads} inner threads "
         f"(≈{n_procs * n_threads} live processes during analysis)")
    info(f"⚙️  Pinning CADFIT_INNER_WORKERS={n_threads} and "
         f"{'/'.join(_BLAS_ENV_VARS)}={n_threads} in every worker")

    _pin_threads(n_threads)

    common_kwargs = dict(
        folder_name=out_dir,
        alpha=args.alpha,
        over_threshold=args.over_threshold,
        under_threshold=args.under_threshold,
        max_iterations=args.max_iterations,
        apply_fillet_chamfer=args.fillet_chamfer,
        keep_history=args.keep_history,
    )

    pending = []
    skipped = 0
    for stl_path in stl_files:
        stl_id = os.path.splitext(os.path.basename(stl_path))[0]
        final_iou_json = f"{out_dir}/{stl_id}/final_iou.json"
        if args.skip_existing and os.path.exists(final_iou_json):
            info(f"⏭️  SKIPPING {os.path.basename(stl_path)} (cached)")
            skipped += 1
            continue
        pending.append(stl_path)

    total = len(stl_files)
    successful, failed, failed_files = skipped, 0, []
    overall_start_time = time.time()

    info(f"\n🚀 Launching {len(pending)} STL(s) across {n_procs} worker process(es)...")

    fail_log = os.path.join(out_dir, "failures.log")

    with ProcessPoolExecutor(max_workers=n_procs,
                             initializer=_worker_init,
                             initargs=(n_threads, silent)) as pool:
        futures = {pool.submit(_process_one, stl_path, common_kwargs): stl_path
                   for stl_path in pending}
        bar = tqdm(as_completed(futures), total=len(pending), unit="stl",
                   desc="Reconstructing", dynamic_ncols=True)
        for fut in bar:
            stl_name, ok, tb = fut.result()
            if ok:
                successful += 1
            else:
                failed += 1
                failed_files.append(stl_name)
                # Record the traceback without disturbing the bar.
                with open(fail_log, "a") as fh:
                    fh.write(f"===== {stl_name} =====\n{tb}\n")
                tqdm.write(f"✗ FAILED {stl_name} (see {fail_log})")
            bar.set_postfix(ok=successful, fail=failed)
        bar.close()

    total_duration = time.time() - overall_start_time
    # Always print the one-line summary, even in silent mode.
    print(f"🏁 {successful}/{total} succeeded, {failed} failed "
          f"in {total_duration/60:.1f} min "
          f"({100*successful/total:.1f}% success rate)")
    if failed_files:
        print(f"   Failures ({len(failed_files)}) logged to {fail_log}:")
        for f in failed_files:
            print(f"     - {f}")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
