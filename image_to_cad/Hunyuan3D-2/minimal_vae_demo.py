import torch

from hy3dgen.shapegen.models.autoencoders import ShapeVAE
from hy3dgen.shapegen.surface_loaders import SharpEdgeSurfaceLoader

# vae = ShapeVAE.from_pretrained(
#     'tencent/Hunyuan3D-2',
#     subfolder='hunyuan3d-vae-v2-0-withencoder',
#     use_safetensors=False,
#     pc_size = 30720,
#     pc_sharpedge_size= 30720,
# )
# loader = SharpEdgeSurfaceLoader(
#     num_sharp_points=30720,
#     num_uniform_points=30720,
# )

vae = ShapeVAE.from_pretrained(
    'tencent/Hunyuan3D-2mini',
    subfolder='hunyuan3d-vae-v2-mini-withencoder',
    use_safetensors=False,
)
loader = SharpEdgeSurfaceLoader(
    num_sharp_points=5120,
    num_uniform_points=5120,
)
surface = loader('demo.glb').to('cuda', dtype=torch.float16)

latents = vae.encode(surface)
latents = vae.decode(latents)
mesh = vae.latents2mesh(
    latents,
    output_type='trimesh',
    bounds=1.01,
    mc_level=0.0,
    num_chunks=20000,
    octree_resolution=256,
    mc_algo='mc',
    enable_pbar=True
)
from hy3dgen.shapegen.pipelines import export_to_trimesh

mesh = export_to_trimesh(mesh)[0]
mesh.export('output.glb')
