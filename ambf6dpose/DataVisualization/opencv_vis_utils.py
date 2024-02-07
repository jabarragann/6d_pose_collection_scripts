from pathlib import Path
from typing import Dict, List
import numpy as np
import cv2
from ambf6dpose.DataCollection.DatasetSample import DatasetSample, RigidObjectsIds


def load_mesh(mesh_path):
    # open3d is to big of a dependency given that we only use it to read the mesh.
    # TODO: replace with a lighter weight library that focus on ply loading.
    import open3d as o3d

    assert mesh_path.exists()

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd.normals = mesh.vertex_normals

    pts3D = np.asarray(pcd.points)

    return pts3D, mesh


def opencv_gt_vis(vertices: np.ndarray, sample: DatasetSample) -> np.ndarray:
    intrinsic_mat = sample.intrinsic_matrix
    extrinsic_mat = sample.needle_pose
    img = sample.raw_img

    rvec, tvec = cv2.Rodrigues(extrinsic_mat[:3, :3])[0], extrinsic_mat[:3, 3]

    # # "anno['camMat']" is camera intrinsic matrix
    img_points, _ = cv2.projectPoints(
        vertices, rvec, tvec, intrinsic_mat, np.zeros(5, dtype="float32")
    )

    # draw perspective correct point cloud back on the image
    for point in img_points:
        p1, p2 = int(point[0][0]), int(point[0][1])
        img[p2, p1] = (255, 255, 255)

    return img