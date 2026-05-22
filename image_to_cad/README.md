# Image → Mesh → CAD

End-to-end pipeline:

1. **`shape_gen.py`** — runs [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) (mini-turbo DiT shape pipeline) to predict a mesh from an RGB image.
2. **`mesh_preprocess.py`** — Taubin smoothing + quadric decimation to a fixed face budget (50 000 by default).
3. **`run_image_to_cad.py`** — orchestrator: loops over every image in a folder, runs (1) → (2) → CADFit.


## Setup

After installing CADFit's main `requirements.txt`:

```bash
# Hunyuan3D's Python deps (torch, diffusers, transformers, etc.)
pip install -r image_to_cad/Hunyuan3D-2/requirements.txt

# Register the bundled hy3dgen package
pip install -e image_to_cad/Hunyuan3D-2
```

## Run

```bash
python image_to_cad/run_image_to_cad.py \
    --images-folder /path/to/images \
    --output-folder images_cad \
    --max-iterations 1
```

Per-image outputs:

```
images_cad/<image_id>/
├── raw.glb                # Hunyuan3D prediction
├── <image_id>.stl         # smoothed + decimated to 50k faces
└── cadfit/<image_id>/
    ├── best_greedy_parallel_iterative.stl
    ├── best_greedy_parallel_iterative.py
    └── final_iou.json
```

Common flags:

| flag | effect |
|---|---|
| `--target-faces N` | decimation target (default `50000`) |
| `--taubin-steps N` | Taubin smoothing iterations (default `20`) |
| `--shape-gen-steps N` | Hunyuan3D diffusion steps (default `5`) |
| `--max-iterations N` | CADFit residual cut/union passes (default `1`) |
| `--fillet-chamfer` | enable CADFit's per-edge fillet/chamfer pass (off by default) |
| `--skip-existing` | resume — skip an image whose `final_iou.json` exists |

## License

Hunyuan3D-2 is distributed under the **Tencent Hunyuan Community License** — see [`Hunyuan3D-2/LICENSE`](./Hunyuan3D-2/LICENSE) and [`Hunyuan3D-2/NOTICE`](./Hunyuan3D-2/NOTICE). It is non-commercial. The CADFit code itself is licensed separately.
