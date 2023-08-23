from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import yaml
from ambf6dpose.DataCollection.BOPSaver.BopSaver import (
    get_folder_names,
    SceneCameraKeys,
    SceneGtKeys,
    GroundTruthFiles,
    DatasetConsts,
)
from ambf6dpose.DataCollection.DatasetSample import DatasetSample
from ambf6dpose.DataCollection.ReaderSaverUtils import AbstractReader, ImgDirs
from dataclasses import dataclass, field
import cv2
import imageio
from ambf6dpose.DataCollection.ReaderSaverUtils import is_rotation, trnorm
from natsort import natsorted


@dataclass
class BopReader(AbstractReader):
    """
    See bop_toolkit for more info

    Parameters
    ----------
    dataset_split : str, optional
        train, val, or test, by default 'train'
    dataset_split_type : str, optional
        split type, by default None
    scene_id : List[int], optional
        scene id, by default [] which means take all scene_ids

    """

    dataset_split: str = "train"
    dataset_split_type: str = None
    scene_id: List[int] = field(default_factory=list)  # If empty, read all scenes

    def __post_init__(self):
        if self.dataset_split_type is not None:
            self.dataset_split_complete = self.dataset_split + "_" + self.dataset_split_type
        else:
            self.dataset_split_complete = self.dataset_split

        self.root = self.root / self.dataset_split_complete

        self.__scene_gt = {}
        self.__scene_camera = {}
        self.__dict_paths: Dict[ImgDirs, Path] = {}
        self.__idx2imgs: List[str, str] = []  # Store tuples of (scene_id, img_name)
        self.__load_data()
        self.__internal_idx: int = 0
        self.__dataset_size: int = self.calculate_size()

    def __len__(self) -> int:
        return self.__dataset_size

    def calculate_size(self) -> int:
        """Make sure that data was loaded correctly"""
        total_length = 0
        for scene_id in self.scene_id:
            total_length += len(list(self.__dict_paths[scene_id][ImgDirs.RAW].glob("*.png")))

        assert total_length == len(self.__idx2imgs), "Error while loading data"
        return total_length

    def __iter__(self):
        self.__internal_idx = 0
        return self

    def __next__(self):
        if self.__internal_idx < self.__dataset_size:
            sample = self[self.__internal_idx]
            self.__internal_idx += 1
            return sample
        else:
            raise StopIteration

    def __load_data(self):
        folder_names = get_folder_names()
        self.__dict_paths = defaultdict(dict)
        if len(self.scene_id) == 0:
            self.scene_id = [p.name for p in list(self.root.glob("*"))]
        else:
            self.scene_id = [self.format_step(x) for x in self.scene_id]

        for scene_id in self.scene_id:
            for img_dir in ImgDirs:
                self.__dict_paths[scene_id][img_dir] = self.root / scene_id / folder_names[img_dir]

        # Create table of idx to scene_id and img_name
        for scene_id in self.scene_id:
            ids = [
                (scene_id, p.name)
                for p in natsorted(self.__dict_paths[scene_id][ImgDirs.RAW].glob("*.png"))
            ]
            self.__idx2imgs += ids

        self.__dict_paths = dict(self.__dict_paths)

        # Load GT files
        for scene_id in self.scene_id:
            with open(self.root / scene_id / GroundTruthFiles.SCENE_GT.value, "r") as f:
                self.__scene_gt[scene_id] = json.load(f)
            with open(self.root / scene_id / GroundTruthFiles.SCENE_CAMERA.value, "r") as f:
                self.__scene_camera[scene_id] = json.load(f)

    def format_step(self, step: int) -> str:
        return f"{step:{DatasetConsts.FMT_STR.value}}"

    def __getitem__(self, key: int) -> DatasetSample:
        if key > DatasetConsts.MAX_STEP.value or key < 0:
            raise IndexError
        else:
            return self.construct_sample(key)

    def get_metadata(self, key: int) -> Tuple[int, str]:
        scene_id, img_name = self.__idx2imgs[key]
        return scene_id, img_name

    def construct_sample(self, key: int) -> DatasetSample:
        step_str = self.format_step(key)
        scene_id, img_name = self.__idx2imgs[key]

        raw_path = str(self.__dict_paths[scene_id][ImgDirs.RAW] / f"{img_name}")
        seg_path = str(self.__dict_paths[scene_id][ImgDirs.SEGMENTED] / f"{img_name}")
        depth_path = str(self.__dict_paths[scene_id][ImgDirs.DEPTH] / f"{img_name}")

        sample = DatasetSample(
            raw_img=cv2.imread(raw_path),
            segmented_img=cv2.imread(seg_path),
            depth_img=self.load_depth(depth_path),
            extrinsic_matrix=self.get_matrix_from_gt(GroundTruthFiles.SCENE_GT, scene_id, img_name),
            intrinsic_matrix=self.get_matrix_from_gt(
                GroundTruthFiles.SCENE_CAMERA, scene_id, img_name
            ),
        )
        return sample

    def load_depth(self, path):
        d: np.ndarray = imageio.imread(path)
        return d.astype(np.float32)

    def get_matrix_from_gt(
        self, file_type: GroundTruthFiles, scene_id: int, img_name: str
    ) -> np.ndarray:
        if file_type == GroundTruthFiles.SCENE_GT:
            return self.get_extrinsic_matrix(scene_id, img_name)
        elif file_type == GroundTruthFiles.SCENE_CAMERA:
            k = str(int(img_name[:-4]))
            intrinsic = self.__scene_camera[scene_id][k][SceneCameraKeys.CAM_K.value]
            intrinsic = np.array(intrinsic).reshape(3, 3)
            return intrinsic
        else:
            raise ValueError

    def get_extrinsic_matrix(self, scene_id: int, img_name) -> np.ndarray:
        k = str(int(img_name[:-4]))
        rot = self.__scene_gt[scene_id][k][0][SceneGtKeys.CAM_R_M2C.value]
        rot = np.array(rot).reshape(3, 3)

        t = self.__scene_gt[scene_id][k][0][SceneGtKeys.CAM_T_M2C.value]
        t = np.array(t)

        obj_id = self.__scene_gt[scene_id][k][0][SceneGtKeys.OBJ_ID.value]
        rot = self.reorthogonalize(rot)

        return self.create_rigid_transform_matrix(rot, t)

    def create_rigid_transform_matrix(self, rot_str: np.ndarray, t_str: np.ndarray) -> np.ndarray:
        rigid_transform = np.eye(4)
        rigid_transform[:3, :3] = rot_str
        rigid_transform[:3, 3] = t_str
        return rigid_transform

    def str_to_numpy(self, str_array: str, shape: Tuple[int, int]) -> np.ndarray:
        loaded = np.fromstring(str_array, sep=",")
        loaded = loaded.reshape(shape)
        return loaded

    def reorthogonalize(self, rot: np.ndarray) -> np.ndarray:
        if is_rotation(rot):
            return rot
        else:
            rot = trnorm(rot)
            assert is_rotation(rot), "Reortogonalization failed..."
            return trnorm(rot)


if __name__ == "__main__":
    reader = BopReader(
        root=Path("."), scene_id=[0, 1], dataset_split="test", dataset_split_type="ds_bop"
    )
    print(f"Dataset size: {len(reader)}")

    idx = 111
    sample: DatasetSample = reader[idx]
    scene_id, img_name = reader.get_metadata(idx)
    sample.generate_gt_vis()
    cv2.imshow(f"{scene_id}-{img_name}", sample.gt_vis_img)
    cv2.waitKey(0)
