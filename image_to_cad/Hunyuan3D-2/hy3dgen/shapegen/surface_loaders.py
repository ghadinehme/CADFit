import numpy as np

import torch
import trimesh


def normalize_mesh(mesh, scale=0.9999):
    bbox = mesh.bounds
    center = (bbox[1] + bbox[0]) / 2
    scale_ = (bbox[1] - bbox[0]).max()

    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale_ * 2 * scale)

    return mesh


def sample_pointcloud(mesh, num=200000):
    points, face_idx = mesh.sample(num, return_index=True)
    normals = mesh.face_normals[face_idx]
    points = torch.from_numpy(points.astype(np.float32))
    normals = torch.from_numpy(normals.astype(np.float32))
    return points, normals


def load_surface(mesh, num_points=8192):
    mesh = normalize_mesh(mesh, scale=0.98)
    surface, normal = sample_pointcloud(mesh)

    rng = np.random.default_rng(seed=0)
    ind = rng.choice(surface.shape[0], num_points, replace=False)
    surface = torch.FloatTensor(surface[ind])
    normal = torch.FloatTensor(normal[ind])

    surface = torch.cat([surface, normal], dim=-1).unsqueeze(0)

    return surface, mesh


def sharp_sample_pointcloud(mesh, num=16384):
    V = mesh.vertices
    N = mesh.face_normals
    VN = mesh.vertex_normals
    F = mesh.faces
    VN2 = np.ones(V.shape[0])
    for i in range(3):
        dot = np.stack((VN2[F[:, i]], np.sum(VN[F[:, i]] * N, axis=-1)), axis=-1)
        VN2[F[:, i]] = np.min(dot, axis=-1)

    sharp_mask = VN2 < 0.985
    # collect edge
    edge_a = np.concatenate((F[:, 0], F[:, 1], F[:, 2]))
    edge_b = np.concatenate((F[:, 1], F[:, 2], F[:, 0]))
    sharp_edge = ((sharp_mask[edge_a] * sharp_mask[edge_b]))
    edge_a = edge_a[sharp_edge > 0]
    edge_b = edge_b[sharp_edge > 0]

    sharp_verts_a = V[edge_a]
    sharp_verts_b = V[edge_b]
    sharp_verts_an = VN[edge_a]
    sharp_verts_bn = VN[edge_b]

    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights /= np.sum(weights)

    random_number = np.random.rand(num)
    w = np.random.rand(num, 1)
    index = np.searchsorted(weights.cumsum(), random_number)
    samples = w * sharp_verts_a[index] + (1 - w) * sharp_verts_b[index]
    normals = w * sharp_verts_an[index] + (1 - w) * sharp_verts_bn[index]
    return samples, normals


def load_surface_sharpegde(mesh, num_points=4096, num_sharp_points=4096, sharpedge_flag=True):
    try:
        mesh_full = trimesh.util.concatenate(mesh.dump())
    except Exception as err:
        mesh_full = trimesh.util.concatenate(mesh)
    mesh_full = normalize_mesh(mesh_full)

    origin_num = mesh_full.faces.shape[0]
    original_vertices = mesh_full.vertices
    original_faces = mesh_full.faces

    mesh = trimesh.Trimesh(vertices=original_vertices, faces=original_faces[:origin_num])
    mesh_fill = trimesh.Trimesh(vertices=original_vertices, faces=original_faces[origin_num:])
    area = mesh.area
    area_fill = mesh_fill.area
    sample_num = 499712 // 2
    num_fill = int(sample_num * (area_fill / (area + area_fill)))
    num = sample_num - num_fill

    random_surface, random_normal = sample_pointcloud(mesh, num=num)
    if num_fill == 0:
        random_surface_fill, random_normal_fill = np.zeros((0, 3)), np.zeros((0, 3))
    else:
        random_surface_fill, random_normal_fill = sample_pointcloud(mesh_fill, num=num_fill)
    random_sharp_surface, sharp_normal = sharp_sample_pointcloud(mesh, num=sample_num)

    # save_surface
    surface = np.concatenate((random_surface, random_normal), axis=1).astype(np.float16)
    surface_fill = np.concatenate((random_surface_fill, random_normal_fill), axis=1).astype(np.float16)
    sharp_surface = np.concatenate((random_sharp_surface, sharp_normal), axis=1).astype(np.float16)
    surface = np.concatenate((surface, surface_fill), axis=0)
    if sharpedge_flag:
        sharpedge_label = np.zeros((surface.shape[0], 1))
        surface = np.concatenate((surface, sharpedge_label), axis=1)
        sharpedge_label = np.ones((sharp_surface.shape[0], 1))
        sharp_surface = np.concatenate((sharp_surface, sharpedge_label), axis=1)
    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], num_points, replace=False)
    surface = torch.FloatTensor(surface[ind])
    ind = rng.choice(sharp_surface.shape[0], num_sharp_points, replace=False)
    sharp_surface = torch.FloatTensor(sharp_surface[ind])

    return torch.cat([surface, sharp_surface], dim=0).unsqueeze(0), mesh_full


class SurfaceLoader:
    def __init__(self, num_points=8192):
        self.num_points = num_points

    def __call__(self, mesh_or_mesh_path, num_points=None):
        if num_points is None:
            num_points = self.num_points

        mesh = mesh_or_mesh_path
        if isinstance(mesh, str):
            mesh = trimesh.load(mesh, force="mesh", merge_primitives=True)
        if isinstance(mesh, trimesh.scene.Scene):
            for idx, obj in enumerate(mesh.geometry.values()):
                if idx == 0:
                    temp_mesh = obj
                else:
                    temp_mesh = temp_mesh + obj
            mesh = temp_mesh
        surface, mesh = load_surface(mesh, num_points=num_points)
        return surface


class SharpEdgeSurfaceLoader:
    def __init__(self, num_uniform_points=8192, num_sharp_points=8192, **kwargs):
        self.num_uniform_points = num_uniform_points
        self.num_sharp_points = num_sharp_points
        self.num_points = num_uniform_points + num_sharp_points

    def __call__(self, mesh_or_mesh_path, num_uniform_points=None, num_sharp_points=None):
        if num_uniform_points is None:
            num_uniform_points = self.num_uniform_points
        if num_sharp_points is None:
            num_sharp_points = self.num_sharp_points

        mesh = mesh_or_mesh_path
        if isinstance(mesh, str):
            mesh = trimesh.load(mesh, force="mesh", merge_primitives=True)
        if isinstance(mesh, trimesh.scene.Scene):
            for idx, obj in enumerate(mesh.geometry.values()):
                if idx == 0:
                    temp_mesh = obj
                else:
                    temp_mesh = temp_mesh + obj
            mesh = temp_mesh
        surface, mesh = load_surface_sharpegde(mesh, num_points=num_uniform_points, num_sharp_points=num_sharp_points)
        return surface
