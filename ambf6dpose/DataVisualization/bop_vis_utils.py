from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from ambf6dpose.DataCollection.DatasetSample import DatasetSample, RigidObjectsIds

from bop_toolkit_lib.renderer_vispy import RendererVispy
from bop_toolkit_lib import misc
from bop_toolkit_lib.renderer_vispy import RendererVispy
from bop_toolkit_lib.visualization import draw_rect, write_text_on_image


class BOPRendererWrapper:
    def __init__(self, width=640, height=480):

        self.renderer = RendererVispy(
            width, height, mode="rgb", shading="flat", bg_color=(0, 0, 0, 0)
        )

        self.models_dict = {}

    def add_object(self, obj_id: RigidObjectsIds, model_path, surf_color):
        self.renderer.add_object(obj_id.value, model_path, surf_color=surf_color)
        self.models_dict[obj_id] = model_path

    def get_model_path(self, obj_id: RigidObjectsIds):
        return self.models_dict[obj_id]

    def render_obj(
        self, obj_id: RigidObjectsIds, obj_pose: np.ndarray, sample: DatasetSample
    ):

        intrinsic_mat = sample.intrinsic_matrix
        fx, fy, cx, cy = (
            intrinsic_mat[0, 0],
            intrinsic_mat[1, 1],
            intrinsic_mat[0, 2],
            intrinsic_mat[1, 2],
        )
        ren_out = self.renderer.render_object(
            obj_id.value, obj_pose[:3, :3], obj_pose[:3, 3], fx, fy, cx, cy
        )
        ren_out = ren_out["rgb"]

        return ren_out


@dataclass
class ImageAnnotations:
    img: np.ndarray
    text_size: int = 11
    text_loc_offset: Tuple[int, int] = (2, 0)

    def __post_init__(self):
        self.annotated_img = self.img.copy()
        self.im_size = (self.img.shape[1], self.img.shape[0])

        self.annotations_list: List[Dict[str, np.ndarray]] = []

    def add_annotations(self, obj_name, render_out: np.ndarray):
        annotation_img = np.zeros_like(self.img)

        obj_mask = np.sum(render_out > 0, axis=2)
        ys, xs = obj_mask.nonzero()
        if len(ys):
            bbox = misc.calc_2d_bbox(xs, ys, self.im_size)
            annotation_img = self._add_bbox(annotation_img, bbox)
            annotation_img = self._add_text(annotation_img, bbox, obj_name)

        self.annotations_list.append(
            dict(name=obj_name, annotation=annotation_img, render_out=render_out)
        )

    def _add_text(self, annotation_img, bbox, obj_name: str) -> np.ndarray:
        text_color = (1.0, 1.0, 1.0)
        text_loc = (
            bbox[0] + self.text_loc_offset[0],
            bbox[1] + self.text_loc_offset[1],
        )
        txt_info = [dict(name=obj_name, val=0, fmt="")]

        annotation_img = write_text_on_image(
            annotation_img, txt_info, text_loc, color=text_color, size=self.text_size
        )

        return annotation_img

    def _add_bbox(self, annotation_img, bbox) -> np.ndarray:
        bbox_color = (0.5, 0.5, 0.5)
        annotation_img = draw_rect(annotation_img, bbox, bbox_color)
        return annotation_img

    def combine_annotations(self):

        all_annotations = np.zeros_like(self.img)
        all_renders = np.zeros_like(self.img)
        for annotation in self.annotations_list:
            all_annotations += annotation["annotation"]
            all_renders += annotation["render_out"]
        all_annotations[all_annotations > 255] = 255
        all_renders[all_renders > 255] = 255

        annotated_img = (
            0.5 * all_annotations.astype(np.float32)
            + 0.4 * all_renders.astype(np.float32)
            + 0.6 * self.img.astype(np.float32)
        )

        annotated_img[annotated_img > 255] = 255
        annotated_img = annotated_img.astype(np.uint8)

        return annotated_img


def rendering_gt_single_obj(model_path: Path, sample: DatasetSample) -> np.ndarray:

    intrinsic_mat = sample.intrinsic_matrix
    extrinsic_mat = sample.psm2_toolpitchlink_pose
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
    model_color = [0.0, 0.8, 0.0]
    renderer.add_object(1, model_path, surf_color=model_color)
    ren_out = renderer.render_object(
        1, extrinsic_mat[:3, :3], extrinsic_mat[:3, 3], fx, fy, cx, cy
    )
    ren_out = ren_out["rgb"]

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

    return vis_im_rgb, ren_out
