"""Image → mesh → CAD pipeline.

For every image in `--images-folder`:
1. `shape_gen.image_to_mesh` runs Hunyuan3D-2 to produce a raw mesh.
2. `mesh_preprocess.preprocess_mesh` applies Taubin smoothing and decimates
   to 50,000 faces.
3. CADFit's `process_single_stl` reconstructs the CAD program.

Outputs are organized as:

    <output-folder>/<image_id>/
        raw.glb              # Hunyuan output
        clean.stl            # smoothed + decimated
        cadfit/              # CADFit per-STL output folder
            best_greedy_parallel_iterative.stl
            best_greedy_parallel_iterative.py
            final_iou.json

Usage:
    python run_image_to_cad.py --images-folder ../../images
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Allow `from run_pipeline import ...` even when this script is run from
# inside `image_to_cad/`.
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser(
        description="Image → Hunyuan3D mesh → Taubin + decimate → CADFit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--images-folder", required=True, help="Folder containing input images.")
    ap.add_argument("--output-folder", default=None,
                    help='Defaults to "<images>_cad/".')
    ap.add_argument("--target-faces", type=int, default=50_000)
    ap.add_argument("--taubin-steps", type=int, default=20)
    ap.add_argument("--max-iterations", type=int, default=1,
                    help="CADFit residual refinement iterations.")
    ap.add_argument("--fillet-chamfer", action="store_true",
                    help="Enable CADFit's per-edge fillet/chamfer pass (off by default).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip an image whose final_iou.json already exists.")
    ap.add_argument("--keep-history", action="store_true",
                    help="Keep CADFit intermediate files. Default: cleanup, "
                         "keeping only the final STL, .py and final_iou.json.")
    ap.add_argument("--shape-gen-steps", type=int, default=5,
                    help="Hunyuan3D inference steps (higher = slower, better).")
    args = ap.parse_args()

    images_dir = args.images_folder.rstrip("/")
    if not os.path.isdir(images_dir):
        print(f"images folder not found: {images_dir}")
        return 1
    out_dir = args.output_folder or f"{os.path.basename(images_dir)}_cad"
    os.makedirs(out_dir, exist_ok=True)

    image_paths = sorted(
        p for p in Path(images_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        print(f"no images found in {images_dir}/")
        return 1
    print(f"🖼️  Found {len(image_paths)} image(s); output → {out_dir}/")

    # Lazy imports: Hunyuan3D pulls torch + diffusers + HF hub, so don't
    # load until we actually need it.
    from image_to_cad.shape_gen import image_to_mesh
    from image_to_cad.mesh_preprocess import preprocess_mesh
    from run_pipeline import process_single_stl

    overall_t0 = time.time()
    for i, img_path in enumerate(image_paths, 1):
        image_id = img_path.stem
        item_dir = Path(out_dir) / image_id
        item_dir.mkdir(parents=True, exist_ok=True)
        final_iou_json = item_dir / "cadfit" / image_id / "final_iou.json"
        if args.skip_existing and final_iou_json.exists():
            print(f"\n[{i}/{len(image_paths)}] ⏭️  {image_id} (cached)")
            continue

        print(f"\n{'='*78}\n[{i}/{len(image_paths)}] {image_id}\n{'='*78}")
        t0 = time.time()

        raw_path = item_dir / "raw.glb"
        clean_path = item_dir / f"{image_id}.stl"
        processed_img_path = item_dir / f"{image_id}_processed.png"

        try:
            if not raw_path.exists():
                print(f"  → Hunyuan3D → {raw_path}")
                image_to_mesh(str(img_path), str(raw_path),
                              num_inference_steps=args.shape_gen_steps,
                              processed_image_path=str(processed_img_path))
                print(f"     processed image → {processed_img_path}")
            else:
                print(f"  ↺ reusing existing {raw_path}")

            print(f"  → Taubin smooth + decimate to {args.target_faces} faces → {clean_path}")
            preprocess_mesh(str(raw_path), str(clean_path),
                            target_faces=args.target_faces,
                            taubin_steps=args.taubin_steps)

            print(f"  → CADFit (max_iterations={args.max_iterations})")
            cadfit_dir = item_dir / "cadfit"
            cadfit_dir.mkdir(parents=True, exist_ok=True)
            process_single_stl(
                stl_path=str(clean_path),
                folder_name=str(cadfit_dir),
                max_iterations=args.max_iterations,
                apply_fillet_chamfer=args.fillet_chamfer,
                keep_history=args.keep_history,
            )
            print(f"  ✓ done in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  ✗ failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*78}\n🏁 finished in {(time.time()-overall_t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
