#!/usr/bin/env python3
"""Run the pipeline on 10 random STLs from old_bench and report IoU + runtime."""
import glob
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

from run_pipeline import process_single_stl

OLD_BENCH = '/media/ghadinehme/6d6c61c1-0d00-4011-afaa-35a7f387fd96/CADEvolve/train/old_bench'
SAMPLE_SIZE = int(os.environ.get('SAMPLE_SIZE', '10'))
SEED = int(os.environ.get('SAMPLE_SEED', '42'))
OUT_ROOT = 'random_samples_runs'


def main():
    random.seed(SEED)
    stls = sorted(glob.glob(os.path.join(OLD_BENCH, '**', '*.stl'), recursive=True))
    print(f'Total STLs in pool: {len(stls)}')
    sample = random.sample(stls, SAMPLE_SIZE)

    results = []
    overall_t0 = time.time()
    for i, stl_path in enumerate(sample, 1):
        parts = Path(stl_path).parts
        bench = parts[-3] if len(parts) >= 3 else 'unk'
        level = parts[-2] if len(parts) >= 2 else 'unk'
        stl_id = Path(stl_path).stem
        folder_name = f'{OUT_ROOT}/{bench}_{level}'
        os.makedirs(folder_name, exist_ok=True)

        print(f'\n{"="*80}')
        print(f'[{i}/{SAMPLE_SIZE}] {bench}/{level}/{stl_id}.stl')
        print(f'{"="*80}')

        t0 = time.time()
        try:
            process_single_stl(
                stl_path=stl_path,
                folder_name=folder_name,
                alpha=0.01,
                over_threshold=0.02,
                under_threshold=0.02,
                max_iterations=1,
            )
            duration = time.time() - t0

            iou_json = Path(folder_name) / stl_id / 'final_iou.json'
            iou = None
            if iou_json.exists():
                try:
                    iou = json.loads(iou_json.read_text()).get('final_iou')
                except Exception:
                    pass
            results.append({
                'idx': i,
                'stl_id': stl_id,
                'bench': bench,
                'level': level,
                'stl_path': stl_path,
                'duration_seconds': duration,
                'final_iou': iou,
                'status': 'ok' if iou is not None else 'no_iou',
            })
            print(f'\n>>> [{i}/{SAMPLE_SIZE}] IoU={iou} duration={duration:.1f}s')
        except Exception as e:
            duration = time.time() - t0
            print(f'EXCEPTION: {e}')
            traceback.print_exc()
            results.append({
                'idx': i,
                'stl_id': stl_id,
                'bench': bench,
                'level': level,
                'stl_path': stl_path,
                'duration_seconds': duration,
                'final_iou': None,
                'status': f'error: {e}',
            })

    total = time.time() - overall_t0

    # Summary
    print('\n' + '=' * 100)
    print('FINAL SUMMARY')
    print('=' * 100)
    print(f'{"#":<4}{"bench/level":<22}{"stl_id":<46}{"IoU":>10}{"time (s)":>12}  status')
    print('-' * 100)
    for r in results:
        iou_s = f'{r["final_iou"]:.4f}' if r['final_iou'] is not None else '  --  '
        print(f'{r["idx"]:<4}{r["bench"]+"/"+r["level"]:<22}{r["stl_id"]:<46}{iou_s:>10}{r["duration_seconds"]:>12.1f}  {r["status"]}')
    ious = [r['final_iou'] for r in results if r['final_iou'] is not None]
    durs = [r['duration_seconds'] for r in results]
    print('-' * 100)
    if ious:
        print(f'mean IoU       : {sum(ious)/len(ious):.4f}  ({len(ious)}/{len(results)} computed)')
        print(f'min/max IoU    : {min(ious):.4f} / {max(ious):.4f}')
    if durs:
        print(f'mean duration  : {sum(durs)/len(durs):.1f}s')
        print(f'total wall time: {total:.1f}s')

    summary_path = f'{OUT_ROOT}/summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'seed': SEED,
            'sample_size': SAMPLE_SIZE,
            'overall_wall_time_seconds': total,
            'results': results,
        }, f, indent=2)
    print(f'\nSaved summary to {summary_path}')


if __name__ == '__main__':
    sys.exit(main())
