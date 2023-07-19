from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict
import cv2

import numpy as np
from enum import Enum

class DataTypes(Enum):
    RAW="raw_img"
    SEGMENTED="segmented_img"
    BLENDED="blended_img"
    EXTRINSIC="extrinsic.yaml"

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
    root : Path
    img_saver: ImageSaver = field(default=None) 
    # yaml_saver: YamlSaver = field(default=None) 

    def __post_init__(self):
        self.__internal_step = 0
        self.root.mkdir(exist_ok=True)

        if self.img_saver is None:
            self.img_saver = ImageSaver(self.root)

        # if self.yaml_saver is None:
        #     self.yaml_saver = YamlSaver(self.root)


    def save_sample(self, sample: DatasetSample):
        self.img_saver.save_sample(step=self.__internal_step, data=sample)
        # self.yaml_saver.save_sample(step=self.__internal_step, data=sample)
        self.__internal_step += 1
    
    def save_intrinsics(self, intrinsics: np.ndarray):
        pass 



@dataclass
class ImageSaver():
    root_path: Path 

    def __post_init__(self):
        self.root_path.mkdir(exist_ok=True)
        self.dir_dict = self.dir_structure()
        self.create_dir()

    def dir_structure(self)->Dict[str,Path]:
        return {
            DataTypes.RAW.name: self.root_path / DataTypes.RAW.value,
            DataTypes.SEGMENTED.name: self.root_path / DataTypes.SEGMENTED.value,
        }

    def create_dir(self):
        for dir in self.dir_dict.values():
            dir.mkdir(exist_ok=True) 
    
    def save_sample(self, step:int, data: DatasetSample):
        raw_path =       str(self.dir_dict[DataTypes.RAW.name] / f"{step}.png")
        segmented_path = str(self.dir_dict[DataTypes.SEGMENTED.name] / f"{step}.png")
        cv2.imwrite(raw_path, data.raw_img)
        cv2.imwrite(segmented_path, data.segmented_img)

class YamlSaver(ABC):
    pass