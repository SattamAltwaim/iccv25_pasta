import numpy as np
import trimesh


def scale_to_unit_sphere(mesh):
    """Center a mesh and scale it so all vertices fit inside the unit sphere.

    Standard normalization used for Chamfer-distance evaluation
    (same convention as mesh_to_sdf / SENS). Missing from the released
    PASTA code but imported by eval.py.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices = vertices / np.max(distances)
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
