
from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from enum import Enum
import yaml
import png
from contextlib import ExitStack
from ambf6dpose.DataCollection.DatasetSample import DatasetSample
from ambf6dpose.DataCollection.ReaderSaverUtils import AbstractSaver
from ambf6dpose.DataCollection.InOut import save_depth
from ambf6dpose.DataCollection.ReaderSaverUtils import ImgDirs, ImageSaver

def get_folder_names():
    folder_names = {
        ImgDirs.RAW: "rgb",
        ImgDirs.SEGMENTED: "segmented",
        ImgDirs.DEPTH: "depth",
        ImgDirs.GT_VISUALIZATION: "gt_vis"
    }
    return folder_names

class GroundTruthFiles(Enum):
    SCENE_CAMERA= "scene_camera.json"
    SCENE_GT= "scene_gt.json"

class SceneCameraKeys(Enum):
    CAM_K= "cam_K"
    DEPTH_SCALE= "depth_scale"
    VIEW_LEVEL= "view_level"

class SceneGtKeys(Enum):
    CAM_R_M2C = "cam_R_m2c"
    CAM_T_M2C= "cam_t_m2c"
    OBJ_ID= "obj_id"

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
    root: Path
    save_every: int = 40

    def __post_init__(self):
        self.exit_stack = ExitStack() # Manage json files opening and closing
        self.scene_camera_path = self.root / GroundTruthFiles.SCENE_CAMERA.value
        self.scene_gt_path = self.root / GroundTruthFiles.SCENE_GT.value

        if self.scene_gt_path.exists():
            msg = f"GT file: {self.scene_gt_path} already exists. Do you want to overwrite it? (y/n) "
            if input(msg) != "y":
                print("exiting ...")
                exit()

        self.scene_gt_file = JsonFileManager(self.scene_gt_path)
        self.scene_camera_file = JsonFileManager(self.scene_camera_path)

        self.scene_gt: Dict[int, List[Dict[str, Any]]] = {}
        self.scene_camera: Dict[int, Dict[str, Any]] = {}

    def open_files(self):
        # Trigger the __enter__ method of the File managers
        self.exit_stack.enter_context(self.scene_gt_file)
        self.exit_stack.enter_context(self.scene_camera_file)

    def save_sample(self, im_id: int, data: DatasetSample):
        obj_id = 1
        self.add_to_scene_camera(im_id, data.intrinsic_matrix)
        self.add_to_scene_gt(im_id, obj_id, data.extrinsic_matrix)

        if im_id % self.save_every == 0:
            self.save_scene_camera()
            self.save_scene_gt()

    def add_to_scene_camera(self,im_id:int, intrinsic_matrix: np.ndarray):
        self.scene_camera[im_id] = {
        'cam_K': intrinsic_matrix,
        'depth_scale': 1.0,
        }

    def add_to_scene_gt(self, im_id:int, obj_id:int, extrinsic_matrix:np.ndarray):    
        rot_m2c = extrinsic_matrix[:3, :3]
        t_m2c = extrinsic_matrix[:3, 3]

        self.scene_gt[im_id] = [{
        'cam_R_m2c': rot_m2c,
        'cam_t_m2c': t_m2c,
        'obj_id': int(obj_id)
        }]

    def save_scene_gt(self):
        self.scene_gt = _scene_gt_as_json(self.scene_gt)
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

        self.scene_camera_file.save_json(self.scene_camera)
        # save_json(self.scene_camera_handle, self.scene_camera)

        # Clear the dictionary.
        self.scene_camera = {}

    def close(self):
        self.save_scene_camera()
        self.save_scene_gt()
        self.exit_stack.close()

        # self.scene_camera_handle.close()
        # self.scene_gt_handle.close()

@dataclass
class JsonFileManager:
    path: Path

    def __enter__(self):
        self.file = open(self.path, "wt", encoding="utf-8")
        self.file.write('{\n') # Opening bracket

        return self.file
      
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        The exit method needs to eliminate the last comma. Trailing commas are not allowed in JSON.
        https://geekpython.medium.com/moving-and-locating-the-file-pointer-in-python-fa32758f7054
        """
        
        self.file.seek(self.file.tell()-2,0) # Remove last comma
        self.file.write('\n}') # Closing bracket
        self.file.close()

    def save_json(self, content):
        """Saves the provided content to a JSON file.
        Taken from BOP toolkit

        :param content: Dictionary/list to save.
        """
        assert isinstance(content, dict)

        content_sorted = sorted(content.items(), key=lambda x: x[0])
        for elem_id, (k, v) in enumerate(content_sorted):
            self.file.write('  \"{}\": {}'.format(k, json.dumps(v, sort_keys=True)))
            self.file.write(',')
            self.file.write('\n')

def save_json(file, content):
    """Saves the provided content to a JSON file.
    Taken from BOP toolkit

    :param path: Path to the output JSON file.
    :param content: Dictionary/list to save.
    """

    if isinstance(content, dict):
        file.write('{\n')
        content_sorted = sorted(content.items(), key=lambda x: x[0])
        for elem_id, (k, v) in enumerate(content_sorted):
            file.write('  \"{}\": {}'.format(k, json.dumps(v, sort_keys=True)))
            if elem_id != len(content) - 1:
                file.write(',')
            file.write('\n')
        file.write('}')

    elif isinstance(content, list):
        file.write('[\n')
        for elem_id, elem in enumerate(content):
            file.write('  {}'.format(json.dumps(elem, sort_keys=True)))
            if elem_id != len(content) - 1:
                file.write(',')
            file.write('\n')
        file.write(']')
    else:
        json.dump(content, file, sort_keys=True)

def _scene_gt_as_json(scene_gt):
    """Saves ground-truth annotations to a JSON file.

    See docs/bop_datasets_format.md for details.

    :param path: Path to the output JSON file.
    :param scene_gt: Dictionary to save to the JSON file.
    """

    for im_id in sorted(scene_gt.keys()):
        im_gts = scene_gt[im_id]
        for gt in im_gts:
            if 'cam_R_m2c' in gt.keys():
                gt['cam_R_m2c'] = gt['cam_R_m2c'].flatten().tolist()
            if 'cam_t_m2c' in gt.keys():
                gt['cam_t_m2c'] = gt['cam_t_m2c'].flatten().tolist()
            if 'obj_bb' in gt.keys():
                gt['obj_bb'] = [int(x) for x in gt['obj_bb']]
    return scene_gt

def _scene_camera_as_json(camera):
  if 'cam_K' in camera.keys():
    camera['cam_K'] = camera['cam_K'].flatten().tolist()
  if 'cam_R_w2c' in camera.keys():
    camera['cam_R_w2c'] = camera['cam_R_w2c'].flatten().tolist()
  if 'cam_t_w2c' in camera.keys():
    camera['cam_t_w2c'] = camera['cam_t_w2c'].flatten().tolist()
  return camera