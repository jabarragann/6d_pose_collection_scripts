from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math
from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np
from enum import Enum
import yaml


class ImgDirs(Enum):
    RAW = "raw_img"
    SEGMENTED = "segmented_img"
    BLENDED = "blended_img"


class YamlFiles(Enum):
    EXTRINSIC = "extrinsic.yaml"
    INTRINSIC = "intrinsic.yaml"


class YamlKeys(Enum):
    ROT_EXT = "rot_model2cam"
    T_EXT = "t_model2cam"
    INTRINSIC = "intrinsic"


@dataclass
class DatasetSample:
    raw_img: np.ndarray
    segmented_img: np.ndarray
    extrinsic_matrix: np.ndarray
    intrinsic_matrix: np.ndarray

    def generate_blended_img():
        pass


@dataclass
class DatasetBuilder:
    dataset_saver: SampleSaver


@dataclass
class SampleSaver:
    root: Path
    img_saver: ImageSaver = field(default=None)
    yaml_saver: YamlSaver = field(default=None)

    def __post_init__(self):
        self.__num_of_digits_per_step: int = 5
        self.__max_step = math.pow(10, self.__num_of_digits_per_step) - 1
        self.__internal_step = 0
        self.__fmt = self.init_step_fmt()

        self.root.mkdir(exist_ok=True)

        if self.img_saver is None:
            self.img_saver = ImageSaver(self.root)

        if self.yaml_saver is None:
            self.yaml_saver = YamlSaver(self.root)

    def init_step_fmt(self) -> str:
        """Create a format string for the step number such as `{step:05d}`"""
        return f"0{self.__num_of_digits_per_step}d"

    def get_fmt_step(self) -> str:
        print(self.__fmt)
        return f"{self.__internal_step:{self.__fmt}}"

    def save_sample(self, sample: DatasetSample):
        self.img_saver.save_sample(str_step=self.get_fmt_step(), data=sample)
        self.yaml_saver.save_sample(str_step=self.get_fmt_step(), data=sample)
        self.__internal_step += 1

        if self.__internal_step > self.__max_step:
            raise ValueError(
                f"Max number of samples reached. "
                "Modify `self.__num_of_digits_per_step` to collect bigger datasets "
                "than {self.__max_step}"
            )

    def save_intrinsics(self, intrinsics: np.ndarray):
        pass

    def close(self):
        self.yaml_saver.close()


@dataclass
class ImageSaver:
    root_path: Path

    def __post_init__(self):
        self.root_path.mkdir(exist_ok=True)
        self.dir_dict = self.get_img_dirs()
        self.create_dir()

    def get_img_dirs(self) -> Dict[ImgDirs, Path]:
        return {
            ImgDirs.RAW: self.root_path / ImgDirs.RAW.value,
            ImgDirs.SEGMENTED: self.root_path / ImgDirs.SEGMENTED.value,
        }

    def create_dir(self):
        for dir in self.dir_dict.values():
            dir.mkdir(exist_ok=True)

    def save_sample(self, str_step: str, data: DatasetSample):
        raw_path = str(self.dir_dict[ImgDirs.RAW] / f"{str_step}.png")
        segmented_path = str(self.dir_dict[ImgDirs.SEGMENTED] / f"{str_step}.png")
        cv2.imwrite(raw_path, data.raw_img)
        cv2.imwrite(segmented_path, data.segmented_img)


@dataclass
class NumpyStrFormatter:
    """
    Class to format numpy array into a string before writting the yaml file.
    """

    precision: int = 8
    max_line_width: int = 200

    def convert_to_yaml(self, data: np.ndarray) -> str:
        flat_data = data.ravel()
        str_repr = np.array2string(
            flat_data, separator=",", precision=self.precision, max_line_width=self.max_line_width
        )
        return str_repr[1:-1]

    def convert_from_yaml(self, data: str, shape: Tuple) -> np.ndarray:
        return np.fromstring(data, sep=",").reshape(shape)


@dataclass
class YamlSaver(ABC):
    root: Path

    def __post_init__(self):
        self.__first_sample = True
        self.numpy_fmt = NumpyStrFormatter()
        file_name = self.root / YamlFiles.EXTRINSIC.value
        if file_name.exists():
            msg = f"GT file: {file_name} already exists. Do you want to overwrite it? (y/n)"
            # if (input(msg) != "y"):
            #     print("exiting ...")
            #     exit()

        self.file_handle = open(file_name, "wt", encoding="utf-8")

    def save_intrinsics(self, data: DatasetSample):
        intrinsics_str = self.numpy_fmt.convert_to_yaml(data.intrinsic_matrix)
        data_dict = {YamlKeys.INTRINSIC.value: intrinsics_str}

        with open(self.root / YamlFiles.INTRINSIC.value, "wt", encoding="utf-8") as file_handle:
            yaml.dump(
                data_dict,
                file_handle,
                Dumper=yaml.SafeDumper,
                default_style="",
                line_break=None,
                width=200,
            )

    def save_sample(self, str_step: str, data: DatasetSample):
        if self.__first_sample:
            self.save_intrinsics(data)
            self.__first_sample = False

        rot_str = self.numpy_fmt.convert_to_yaml(data.extrinsic_matrix[:3, :3])
        t_str = self.numpy_fmt.convert_to_yaml(data.extrinsic_matrix[:3, 3])
        data_dict = {f"{str_step}": {YamlKeys.ROT_EXT.value: rot_str, YamlKeys.T_EXT.value: t_str}}

        yaml.dump(
            data_dict,
            self.file_handle,
            Dumper=yaml.SafeDumper,
            default_style="",
            line_break=None,
            width=200,
        )
        self.file_handle.write("\n")
        self.file_handle.flush()

    def close(self):
        self.file_handle.close()
