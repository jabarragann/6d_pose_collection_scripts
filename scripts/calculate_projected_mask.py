import copy
from pathlib import Path
import open3d as o3d
import numpy as np
import cv2
from ambf6dpose import YamlDatasetReader
from surgical_robotics_challenge.units_conversion import SimToSI

from ambf6dpose.DataCollection.DatasetSample import DatasetSample


def load_mesh(mesh_path):
    assert mesh_path.exists()

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd.normals = mesh.vertex_normals

    pts3D = np.asarray(pcd.points)

    # convert from simulation units to SI units
    # pts3D = pts3D / SimToSI.linear_factor

    return pts3D, mesh


def opencv_projection(vertices: np.ndarray, sample: DatasetSample):
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

    cv2.imshow("img", img)
    cv2.waitKey(0)


def bop_rendering(model_path: Path, sample: DatasetSample):
    from bop_toolkit_lib.renderer_vispy import RendererVispy
    from bop_toolkit_lib.visualization import draw_rect, write_text_on_image
    from bop_toolkit_lib import misc

    intrinsic_mat = sample.intrinsic_matrix
    extrinsic_mat = sample.needle_pose
    fx, fy, cx, cy = (
        intrinsic_mat[0, 0],
        intrinsic_mat[1, 1],
        intrinsic_mat[0, 2],
        intrinsic_mat[1, 2],
    )
    img = sample.raw_img
    ren_rgb_info = np.zeros_like(img)

    width = 640
    height = 480
    renderer = RendererVispy(
        width, height, mode="rgb", shading="flat", bg_color=(0, 0, 0, 0)
    )

    # Load model
    model_color = [0.0, 0.5, 0.0]
    renderer.add_object(1, model_path, surf_color=model_color)
    # Render
    ren_out = renderer.render_object(
        1, extrinsic_mat[:3, :3], extrinsic_mat[:3, 3], fx, fy, cx, cy
    )
    ren_out = ren_out["rgb"]

    # Draw bounding box and text info
    obj_mask = np.sum(ren_out > 0, axis=2)
    ys, xs = obj_mask.nonzero()
    if len(ys):
        bbox_color = (0.5, 0.5, 0.5)
        text_color = (1.0, 1.0, 1.0)
        text_size = 11

        im_size = (obj_mask.shape[1], obj_mask.shape[0])
        bbox = misc.calc_2d_bbox(xs, ys, im_size)
        ren_rgb_info = draw_rect(ren_rgb_info, bbox, bbox_color)

        # text info
        text_loc = (bbox[0] + 2, bbox[1])
        txt_info = [dict(name="needle", val=0, fmt="")]
        ren_rgb_info = write_text_on_image(
            ren_rgb_info, txt_info, text_loc, color=text_color, size=text_size
        )

    # Combine with raw image
    vis_im_rgb = (
        0.5 * img.astype(np.float32)
        + 0.5 * ren_out.astype(np.float32)
        + 1.0 * ren_rgb_info.astype(np.float32)
    )

    vis_im_rgb[vis_im_rgb > 255] = 255
    vis_im_rgb = vis_im_rgb.astype(np.uint8)

    final = np.hstack((img, vis_im_rgb))
    cv2.imshow("img", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mesh_path = Path("./scripts/6d_pose_sample_ds/Needle_triangle_scaled_mm.ply")
    # ds_path = Path("./scripts/6d_pose_sample_ds")
    ds_path = Path("./test_ds_good")
    dataset = YamlDatasetReader(ds_path)

    vertices, mesh = load_mesh(mesh_path)

    # opencv_projection(vertices, dataset[5])
    bop_rendering(mesh_path, dataset[5])
