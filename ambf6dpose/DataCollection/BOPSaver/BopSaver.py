from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import cv2
import numpy as np
from enum import Enum
import yaml
import png
from contextlib import ExitStack
from ambf6dpose.DataCollection.DatasetSample import DatasetSample
from ambf6dpose.DataCollection.ReaderSaverUtils import AbstractSaver
from ambf6dpose.DataCollection.ReaderSaverUtils import ImgDirs, ImageSaver


def get_folder_names():
    folder_names = {
        ImgDirs.RAW: "rgb",
        ImgDirs.SEGMENTED: "segmented",
        ImgDirs.DEPTH: "depth",
        ImgDirs.GT_VISUALIZATION: "gt_vis",
    }
    return folder_names


class GroundTruthFiles(Enum):
    SCENE_CAMERA = "scene_camera.json"
    SCENE_GT = "scene_gt.json"


class SceneCameraKeys(Enum):
    CAM_K = "cam_K"
    DEPTH_SCALE = "depth_scale"
    VIEW_LEVEL = "view_level"


class SceneGtKeys(Enum):
    CAM_R_M2C = "cam_R_m2c"
    CAM_T_M2C = "cam_t_m2c"
    OBJ_ID = "obj_id"


class DatasetConsts(Enum):
    NUM_OF_DIGITS_PER_STEP = 6
    MAX_STEP = math.pow(10, NUM_OF_DIGITS_PER_STEP) - 1
    FMT_STR = f"0{NUM_OF_DIGITS_PER_STEP}d"


@dataclass
class BopSampleSaver(AbstractSaver):
    scene_id: int
    img_saver: ImageSaver = field(default=None)
    json_saver: JsonSaver = field(default=None)

    fmt_str: str = field(default=DatasetConsts.FMT_STR.value, init=False)
    max_step: int = field(default=DatasetConsts.MAX_STEP.value, init=False)

    def __post_init__(self):
        self.__internal_step = 0

        if isinstance(self.scene_id, str):
            self.scene_id = int(self.scene_id)

        self.root = self.root / self.fmt_step(self.scene_id)
        self.root.mkdir(exist_ok=True, parents=True)

        if self.img_saver is None:
            self.img_saver = ImageSaver(self.root, get_folder_names())

        if self.json_saver is None:
            self.json_saver = JsonSaver(self.root)

    def fmt_step(self, step) -> str:
        return f"{step:{self.fmt_str}}"

    def save_sample(self, sample: DatasetSample):
        self.img_saver.save_sample(str_step=self.fmt_step(self.__internal_step), data=sample)
        self.json_saver.save_sample(self.__internal_step, data=sample)
        self.__internal_step += 1

        if self.__internal_step > self.max_step:
            raise ValueError(
                f"Max number of samples reached. "
                "Modify `self.__num_of_digits_per_step` to collect bigger datasets "
                "than {self.max_step}"
            )

    def close(self):
        self.json_saver.close()

    def __enter__(self):
        self.json_saver.open_files()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


@dataclass
class JsonSaver(ABC):
    """Saves data into BOP format. Each image has a single scene_camera value
     and a list gt annotations to account for multiple objects.

    ```
    self.scene_gt: Dict[int, List[Dict[str, Any]]] = {}
    self.scene_camera: Dict[int, Dict[str, Any]] = {}
    ```

    Parameters
    ----------
    ABC : _type_
        _description_
    """

    root: Path
    save_every: int = 4
    scene_camera_name: str = GroundTruthFiles.SCENE_CAMERA.value
    scene_gt_name: str = GroundTruthFiles.SCENE_GT.value
    safe_save: bool = False

    def __post_init__(self):
        self.exit_stack = ExitStack()  # Manage json files opening and closing
        self.scene_camera_path = self.root / self.scene_camera_name
        self.scene_gt_path = self.root / self.scene_gt_name

        if self.scene_gt_path.exists() and self.safe_save:
            msg = (
                f"GT file: {self.scene_gt_path} already exists. Do you want to overwrite it? (y/n) "
            )
            if input(msg) != "y":
                print("exiting ...")
                exit()

        self.scene_gt_file = JsonFileManager(self.scene_gt_path)
        self.scene_camera_file = JsonFileManager(self.scene_camera_path)

        self.scene_gt: Dict[int, List[Dict[str, Any]]] = {}
        self.scene_camera: Dict[int, Dict[str, Any]] = {}

        self.__internal_step = 0

    def __enter__(self):
        self.open_files()
        return self

    def open_files(self):
        # Trigger the __enter__ method of the File managers
        self.exit_stack.enter_context(self.scene_gt_file)
        self.exit_stack.enter_context(self.scene_camera_file)

    def save_sample(self, im_id: int, data: DatasetSample):
        obj_id = 1
        self.add_to_scene_camera(im_id, data.intrinsic_matrix)
        self.add_to_scene_gt(im_id, obj_id, data.extrinsic_matrix)

        if self.__internal_step % self.save_every == 0:
            self.save_scene_camera()
            self.save_scene_gt()

        self.__internal_step += 1

    def add_to_scene_camera(self, im_id: int, intrinsic_matrix: np.ndarray):
        self.scene_camera[im_id] = {
            SceneCameraKeys.CAM_K: intrinsic_matrix,
            SceneCameraKeys.DEPTH_SCALE: 1.0,
        }

    def add_to_scene_gt(self, im_id: int, obj_id: int, extrinsic_matrix: np.ndarray):
        rot_m2c = extrinsic_matrix[:3, :3]
        t_m2c = extrinsic_matrix[:3, 3]

        self.scene_gt[im_id] = [
            {
                SceneGtKeys.CAM_R_M2C: rot_m2c,
                SceneGtKeys.CAM_T_M2C: t_m2c,
                SceneGtKeys.OBJ_ID: int(obj_id),
            }
        ]

    def save_scene_gt(self):
        self.scene_gt = _scene_gt_as_json(self.scene_gt)
        self.scene_gt = _replace_enums_scene_gt(self.scene_gt)
        # print("Saving scene_gt")
        # print(self.scene_gt)
        self.scene_gt_file.save_json(self.scene_gt)

        self.scene_gt = {}

    def save_scene_camera(self):
        """Saves information about the scene camera to a JSON file.

        See docs/bop_datasets_format.md for details.

        :param path: Path to the output JSON file.
        :param scene_camera: Dictionary to save to the JSON file.
        """
        for im_id in sorted(self.scene_camera.keys()):
            self.scene_camera[im_id] = _scene_camera_as_json(self.scene_camera[im_id])
        self.scene_camera = _replace_enums_scene_camera(self.scene_camera)

        # print("Saving scene_camera")
        # print(self.scene_camera)
        self.scene_camera_file.save_json(self.scene_camera)

        # Clear the dictionary.
        self.scene_camera = {}

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def close(self):
        self.save_scene_camera()
        self.save_scene_gt()
        self.exit_stack.close()  # Trigger the __exit__ method of the File managers


@dataclass
class JsonFileManager:
    path: Path
    store_data_as: str = "dict"

    def __post_init__(self):
        assert self.store_data_as in ["dict", "list"], "store_data_as must be either dict or list"
        if self.store_data_as == "dict":
            self.opening_char = "{\n"
            self.closing_char = "\n}"
        elif self.store_data_as == "list":
            self.opening_char = "[\n"
            self.closing_char = "\n]"

    def __enter__(self):
        self.file = open(self.path, "wt", encoding="utf-8")
        self.file.write(self.opening_char)  # Opening bracket

        return self.file

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        The exit method needs to eliminate the last comma. Trailing commas are not allowed in JSON.
        https://geekpython.medium.com/moving-and-locating-the-file-pointer-in-python-fa32758f7054
        """

        self.file.seek(self.file.tell() - 2, 0)  # Remove last comma
        self.file.write(self.closing_char)  # Closing bracket
        self.file.close()

    def save_json(self, content: Union[dict[str, Any], List[Any]]):
        """Saves the provided content to a JSON file.
        Taken from BOP toolkit

        :param content: Dictionary/list to save.
        """

        if self.store_data_as == "dict":
            self._save_dict(content)
        elif self.store_data_as == "list":
            self._save_list(content)

    def _save_dict(self, content: dict[str, Any]):
        assert isinstance(content, dict), "content must be dict when store_data_as is dict"

        content_sorted = sorted(content.items(), key=lambda x: x[0])
        for elem_id, (k, v) in enumerate(content_sorted):
            self.file.write('  "{}": {}'.format(k, json.dumps(v, sort_keys=True)))
            self.file.write(",")
            self.file.write("\n")

    def _save_list(self, content: List[Any]):
        assert isinstance(content, list), "content must be a list when store_data_as is list"
        for elem_id, elem in enumerate(content):
            self.file.write("  {}".format(json.dumps(elem, sort_keys=True)))
            self.file.write(",")
            self.file.write("\n")

    # def replace_enum_keys(self, content: dict[Enum, Any]) -> dict[str, Any]:
    #     new_dict = defaultdict(dict)
    #     print(content)
    #     for img_id, content_dict in content.items():
    #         for k, v in content_dict.items():
    #             if isinstance(k, Enum):
    #                 new_dict[img_id][k.value] = v
    #             else:
    #                 new_dict[img_id][k] = v

    #     return dict(new_dict)


def _scene_gt_as_json(scene_gt):
    """Saves ground-truth annotations to a JSON file.

    See docs/bop_datasets_format.md for details.

    :param path: Path to the output JSON file.
    :param scene_gt: Dictionary to save to the JSON file.
    """

    for im_id in sorted(scene_gt.keys()):
        im_gts = scene_gt[im_id]
        for gt in im_gts:
            if SceneGtKeys.CAM_R_M2C in gt.keys():
                gt[SceneGtKeys.CAM_R_M2C] = gt[SceneGtKeys.CAM_R_M2C].flatten().tolist()
            if SceneGtKeys.CAM_T_M2C in gt.keys():
                gt[SceneGtKeys.CAM_T_M2C] = gt[SceneGtKeys.CAM_T_M2C].flatten().tolist()
            if "obj_bb" in gt.keys():
                gt["obj_bb"] = [int(x) for x in gt["obj_bb"]]

    return scene_gt


def _scene_camera_as_json(camera):
    if SceneCameraKeys.CAM_K in camera.keys():
        camera[SceneCameraKeys.CAM_K] = camera[SceneCameraKeys.CAM_K].flatten().tolist()
    if "cam_R_w2c" in camera.keys():
        camera["cam_R_w2c"] = camera["cam_R_w2c"].flatten().tolist()
    if "cam_t_w2c" in camera.keys():
        camera["cam_t_w2c"] = camera["cam_t_w2c"].flatten().tolist()
    return camera


def _replace_enums_scene_gt(content: dict[int, list[dict[Enum, Any]]]) -> dict[str, Any]:
    new_dict = defaultdict(list)
    for img_id, annotation_list in content.items():
        for anno in annotation_list:
            annotation_dict = {}
            for k, v in anno.items():
                if isinstance(k, Enum):
                    annotation_dict[k.value] = v
                else:
                    annotation_dict[k] = v
            new_dict[img_id].append(annotation_dict)

    return dict(new_dict)


def _replace_enums_scene_camera(content: dict[int, dict[Enum, Any]]) -> dict[str, Any]:
    new_dict = defaultdict(dict)
    for img_id, content_dict in content.items():
        for k, v in content_dict.items():
            if isinstance(k, Enum):
                new_dict[img_id][k.value] = v
            else:
                new_dict[img_id][k] = v

    return dict(new_dict)


## Old save_json func

# def save_json(file, content)
#     """Saves the provided content to a JSON file.
#     Taken from BOP toolkit

#     :param path: Path to the output JSON file.
#     :param content: Dictionary/list to save.
#     """

#     if isinstance(content, dict):
#         file.write("{\n")
#         content_sorted = sorted(content.items(), key=lambda x: x[0])
#         for elem_id, (k, v) in enumerate(content_sorted):
#             file.write('  "{}": {}'.format(k, json.dumps(v, sort_keys=True)))
#             if elem_id != len(content) - 1:
#                 file.write(",")
#             file.write("\n")
#         file.write("}")

#     elif isinstance(content, list):
#         file.write("[\n")
#         for elem_id, elem in enumerate(content):
#             file.write("  {}".format(json.dumps(elem, sort_keys=True)))
#             if elem_id != len(content) - 1:
#                 file.write(",")
#             file.write("\n")
#         file.write("]")
#     else:
#         json.dump(content, file, sort_keys=True)
