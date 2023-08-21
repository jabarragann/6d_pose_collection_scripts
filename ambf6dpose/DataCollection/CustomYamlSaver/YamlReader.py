from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import yaml
from ambf6dpose.DataCollection.CustomYamlSaver.YamlSaver import YamlFiles, YamlKeys, ImgDirs, DatasetConsts, get_folder_names
from ambf6dpose.DataCollection.DatasetSample import DatasetSample
from ambf6dpose.DataCollection.ReaderSaverUtils import AbstractReader
from dataclasses import dataclass, field
import cv2
import imageio
from ambf6dpose.DataCollection.ReaderSaverUtils import is_rotation, trnorm


@dataclass
class DatasetReader(AbstractReader):

    def __post_init__(self):
        self.__dict_paths: Dict[ImgDirs, Path] = {}
        self.__init_dict_paths()
        self.__internal_idx: int = 0
        self.__dataset_size: int = self.calculate_size()

        with open(self.root / YamlFiles.EXTRINSIC.value, "r") as f:
            self.__extrinsic_yaml = yaml.load(f, Loader=yaml.FullLoader)

        with open(self.root / YamlFiles.INTRINSIC.value, "r") as f:
            self.__intrinsic_yaml = yaml.load(f, Loader=yaml.FullLoader)

    def __len__(self) -> int:
        return self.__dataset_size

    def calculate_size(self) -> int:
        return len(list(self.__dict_paths[ImgDirs.RAW].glob("*.png")))

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

    def __init_dict_paths(self):
        folder_names = get_folder_names()
        for img_dir in ImgDirs:
            self.__dict_paths[img_dir] = self.root / folder_names[img_dir]

    def format_step(step: int) -> str:
        return f"{step:{DatasetConsts.FMT_STR.value}}"

    def __getitem__(self, key: int) -> DatasetSample:
        if key > DatasetConsts.MAX_STEP.value or key < 0:
            raise IndexError
        else:
            return self.construct_sample(key) 

    def construct_sample(self, key: int) -> DatasetSample:
        step_str = DatasetReader.format_step(key)
        raw_path = str(self.__dict_paths[ImgDirs.RAW] / f"{step_str}.png")
        seg_path = str(self.__dict_paths[ImgDirs.SEGMENTED] / f"{step_str}.png")
        depth_path = str(self.__dict_paths[ImgDirs.DEPTH] / f"{step_str}.png")

        sample = DatasetSample(
            raw_img=cv2.imread(raw_path),
            segmented_img=cv2.imread(seg_path),
            depth_img = self.load_depth(depth_path), 
            extrinsic_matrix=self.get_matrix_from_yaml(YamlFiles.EXTRINSIC, step_str),
            intrinsic_matrix=self.get_matrix_from_yaml(YamlFiles.INTRINSIC, step_str),
        )
        return sample

    def load_depth(self, path):
        d:np.ndarray = imageio.imread(path)
        return d.astype(np.float32)

    def get_matrix_from_yaml(self, yaml_type: YamlFiles, key: str) -> np.ndarray:
        if yaml_type == YamlFiles.EXTRINSIC:
            return self.get_extrinsic_matrix(key)
        elif yaml_type == YamlFiles.INTRINSIC:
            intrinsic_str = self.__intrinsic_yaml[YamlKeys.INTRINSIC.value]
            return self.str_to_numpy(intrinsic_str, (3, 3))
        else:
            raise ValueError

    def get_extrinsic_matrix(self, key: str) -> np.ndarray:
        rot = self.str_to_numpy(self.__extrinsic_yaml[key][YamlKeys.ROT_EXT.value], (3, 3))
        rot = self.reorthogonalize(rot)
        t = self.str_to_numpy(self.__extrinsic_yaml[key][YamlKeys.T_EXT.value], (3,))
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
    reader = DatasetReader(Path("./test_ds"))
    sample = reader[0]
    sample.generate_blended_img()
    cv2.imshow("raw", sample.blended_img)
    cv2.waitKey(0)
