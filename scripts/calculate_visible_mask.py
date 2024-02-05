import copy
from pathlib import Path
import open3d as o3d
import numpy as np
import cv2
from ambf6dpose import YamlDatasetReader
from surgical_robotics_challenge.units_conversion import SimToSI
from ambf6dpose.DataCollection.CustomYamlSaver.DatasetBuilder import DatasetSample
from bop_toolkit_lib.renderer_vispy import RendererVispy
from bop_toolkit_lib.visualization import draw_rect, write_text_on_image
from bop_toolkit_lib import misc


def bop_rendering(model_path, sample:DatasetSample):

    intrinsic_mat = sample.intrinsic_matrix
    extrinsic_mat = sample.extrinsic_matrix
    extrinsic_mat[:3,3] = extrinsic_mat[:3,3] # convert from m to mm
    fx, fy, cx, cy = (
        intrinsic_mat[0, 0],
        intrinsic_mat[1, 1],
        intrinsic_mat[0, 2],
        intrinsic_mat[1, 2],
    )
    img = sample.raw_img
    img_depth =  sample.depth_img
    ren_rgb_info = np.zeros_like(img)

    width = 640
    height = 480
    renderer = RendererVispy(width, height, mode="rgb+depth", shading="flat", bg_color=(0, 0, 0, 0))

    # Load model
    model_color = [0.0, 0.5, 0.0]
    renderer.add_object(1, model_path, surf_color=model_color)
    # Render
    ren_out = renderer.render_object(1, extrinsic_mat[:3, :3], extrinsic_mat[:3, 3], fx, fy, cx, cy)
    ren_rgb = ren_out["rgb"]

    # Calculate visible mask
    ren_depth = ren_out["depth"]
    valid_mask = (img_depth>0) * (ren_depth>0)
    depth_diff = valid_mask * abs((ren_depth.astype(np.float32) - img_depth))
    delta = 1
    below_delta = valid_mask * (depth_diff < delta)
    below_delta_vis = (255 * below_delta).astype(np.uint8)

    above_delta = valid_mask * (depth_diff > delta)
    above_delta_vis = (255 * above_delta).astype(np.uint8)

    visible_mask = np.dstack((below_delta_vis, above_delta_vis, below_delta_vis)).astype(np.uint8)

    # Draw bounding box and text info
    obj_mask = np.sum(ren_rgb > 0, axis=2)
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
        + 0.5 * visible_mask.astype(np.float32)
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
    ds_path = Path("./test_ds")
    dataset = YamlDatasetReader(ds_path)

    sample: DatasetSample = dataset[55]
    bop_rendering(mesh_path, sample)