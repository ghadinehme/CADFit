"""Hunyuan3D image-to-mesh inference.

Thin wrapper around Tencent's Hunyuan3D-2 (mini-turbo) DiT shape pipeline.
The model checkpoint is pulled on first use from `tencent/Hunyuan3D-2mini`
via Hugging Face Hub.

Adapted from the upstream example at:
    https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/examples/shape_gen.py
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


# Cached singletons so we don't reload weights for every image.
_PIPELINE = None
_REMBG = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mini",
            subfolder="hunyuan3d-dit-v2-mini-turbo",
            use_safetensors=False,
        )
    return _PIPELINE


def _get_rembg():
    global _REMBG
    if _REMBG is None:
        _REMBG = BackgroundRemover()
    return _REMBG


def _has_real_alpha(image: Image.Image) -> bool:
    """True if the image has a non-trivial alpha channel (i.e. background
    already cut out). A fully-opaque alpha is treated as no alpha."""
    if image.mode not in ("RGBA", "LA"):
        return False
    alpha = image.split()[-1]
    lo, hi = alpha.getextrema()
    return lo < 250


def image_to_mesh(image_path: str,
                   output_path: str,
                   num_inference_steps: int = 5,
                   octree_resolution: int = 380,
                   num_chunks: int = 20000,
                   guidance: float = 5,
                   seed: int = 12345,
                   processed_image_path: str | None = None):
    """Run Hunyuan3D-2 on `image_path` and save the predicted mesh to `output_path`.

    The output is whatever extension `output_path` carries — `.glb`, `.obj`,
    `.stl`, etc. — supported by trimesh's exporter.

    If the source image lacks a non-trivial alpha channel, the BackgroundRemover
    is applied to cut out the foreground object. The processed (background-
    removed) image is written to `processed_image_path` for inspection; if
    None, it is saved next to `output_path` as `<stem>_processed.png`.
    """
    src = Image.open(image_path)
    if _has_real_alpha(src):
        image = src.convert("RGBA")
    else:
        image = _get_rembg()(src.convert("RGB"))
        if image.mode != "RGBA":
            image = image.convert("RGBA")

    out_abs = os.path.abspath(output_path)
    out_dir = os.path.dirname(out_abs) or "."
    os.makedirs(out_dir, exist_ok=True)

    if processed_image_path is None:
        stem = Path(output_path).stem
        processed_image_path = os.path.join(out_dir, f"{stem}_processed.png")
    image.save(processed_image_path)

    pipeline = _get_pipeline()
    mesh = pipeline(
        image=image,
        num_inference_steps=num_inference_steps,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
        generator=torch.manual_seed(seed),
        output_type="trimesh",
        guidance=guidance,
    )[0]
    mesh.export(output_path)
    return output_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Image-to-mesh via Hunyuan3D-2 mini-turbo.")
    ap.add_argument("image_path")
    ap.add_argument("output_path")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--octree-resolution", type=int, default=380)
    ap.add_argument("--num-chunks", type=int, default=20000)
    ap.add_argument("--guidance", type=float, default=5)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    image_to_mesh(args.image_path, args.output_path,
                  num_inference_steps=args.steps,
                  octree_resolution=args.octree_resolution,
                  num_chunks=args.num_chunks,
                  guidance=args.guidance,
                  seed=args.seed)
