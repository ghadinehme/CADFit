"""Image-to-CAD orchestration: Hunyuan3D → Taubin smooth + decimate → CADFit."""
from .shape_gen import image_to_mesh  # noqa: F401
from .mesh_preprocess import preprocess_mesh  # noqa: F401
