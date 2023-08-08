from pathlib import Path
import open3d as o3d
import numpy as np
import cv2
from ambf6dpose import DatasetReader
from surgical_robotics_challenge.units_conversion import SimToSI

from ambf6dpose.DataCollection.DatasetBuilder import DatasetSample


def load_mesh():
    mesh_path = Path("./scripts/6d_pose_sample_ds/Needle.ply")
    assert mesh_path.exists()

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd.normals = mesh.vertex_normals

    pts3D = np.asarray(pcd.points)

    # convert from simulation units to SI units
    pts3D = pts3D / SimToSI.linear_factor

    return pts3D, mesh


def open_cv_projection(vertices: np.ndarray, sample: DatasetSample):
    sample = dataset[0]
    intrinsic_mat = sample.intrinsic_matrix
    extrinsic_mat = sample.extrinsic_matrix
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

    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    ds_path = Path("./scripts/6d_pose_sample_ds")
    dataset = DatasetReader(ds_path)

    vertices, mesh = load_mesh()

    # o3d.visualization.draw(mesh, raw_mode=True)

    open_cv_projection(vertices, dataset[0])
