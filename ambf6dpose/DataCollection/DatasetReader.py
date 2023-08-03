from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import yaml
from ambf6dpose.DataCollection.DatasetBuilder import YamlFiles, YamlKeys, ImgDirs, DatasetConsts
from ambf6dpose.DataCollection.DatasetBuilder import DatasetSample
from dataclasses import dataclass, field
import cv2


def trnorm(rot: np.ndarray):
    """Convert to proper rotation matrix
    https://petercorke.github.io/spatialmath-python/func_3d.html?highlight=trnorm#spatialmath.base.transforms3d.trnorm

    Parameters
    ----------
    rot : np.ndarray
        3x3 numpy array

    Returns
    -------
    proper_rot
        proper rotation matrix
    """

    unitvec = lambda x: x / np.linalg.norm(x)
    o = rot[:3, 1]
    a = rot[:3, 2]

    n = np.cross(o, a)  # N = O x A
    o = np.cross(a, n)  # (a)];
    new_rot = np.stack((unitvec(n), unitvec(o), unitvec(a)), axis=1)

    return new_rot


def is_rotation(rot: np.ndarray, tol=100):
    """Test if matrix is a proper rotation matrix

    Taken from
    https://petercorke.github.io/spatialmath-python/func_nd.html#spatialmath.base.transformsNd.isR

    Parameters
    ----------
    rot : np.ndarray
        3x3 np.array
    """
    _eps = np.finfo(np.float64).eps
    return (
        np.linalg.norm(rot @ rot.T - np.eye(rot.shape[0])) < tol * _eps
        and np.linalg.det(rot @ rot.T) > 0
    )


@dataclass
class DatasetReader:
    root: Path

    def __post_init__(self):
        self.__dict_paths: Dict[ImgDirs, Path] = {}

        self.__init_dict_paths()

        with open(self.root / YamlFiles.EXTRINSIC.value, "r") as f:
            self.__extrinsic_yaml = yaml.load(f, Loader=yaml.FullLoader)

        with open(self.root / YamlFiles.INTRINSIC.value, "r") as f:
            self.__intrinsic_yaml = yaml.load(f, Loader=yaml.FullLoader)

    def __init_dict_paths(self):
        for img_dir in ImgDirs:
            self.__dict_paths[img_dir] = self.root / img_dir.value

    def format_step(step: int) -> str:
        return f"{step:{DatasetConsts.FMT_STR.value}}"

    def __getitem__(self, key: int) -> DatasetSample:
        if key > DatasetConsts.MAX_STEP.value or key < 0:
            raise IndexError
        else:
            step_str = DatasetReader.format_step(key)
            sample = DatasetSample(
                raw_img=cv2.imread(str(self.__dict_paths[ImgDirs.RAW] / f"{step_str}.png")),
                segmented_img=cv2.imread(
                    str(self.__dict_paths[ImgDirs.SEGMENTED] / f"{step_str}.png")
                ),
                extrinsic_matrix=self.get_matrix_from_yaml(YamlFiles.EXTRINSIC, step_str),
                intrinsic_matrix=self.get_matrix_from_yaml(YamlFiles.INTRINSIC, step_str),
            )
            return sample

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
    reader = DatasetReader(Path("./test_d2"))
    sample = reader[0]
    sample.generate_blended_img()
    cv2.imshow("raw", sample.blended_img)
    cv2.waitKey(0)
