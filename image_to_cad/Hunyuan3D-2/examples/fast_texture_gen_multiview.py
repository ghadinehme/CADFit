import time

import torch
from PIL import Image
import trimesh

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline

images_path = [
    "assets/example_mv_images/1/front.png",
    "assets/example_mv_images/1/left.png",
    "assets/example_mv_images/1/back.png"
]

images = []
for image_path in images_path:
    image = Image.open(image_path)
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    images.append(image)

pipeline = Hunyuan3DPaintPipeline.from_pretrained(
    'tencent/Hunyuan3D-2',
    subfolder='hunyuan3d-paint-v2-0-turbo'
)

mesh = trimesh.load('assets/1.glb')

mesh = pipeline(mesh, image=images)
mesh.export('demo_textured.glb')