from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple
import cv2

import numpy as np
from enum import Enum

import yaml

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
    yaml_saver: YamlSaver = field(default=None) 

    def __post_init__(self):
        self.__internal_step = 0
        self.root.mkdir(exist_ok=True)

        if self.img_saver is None:
            self.img_saver = ImageSaver(self.root)

        if self.yaml_saver is None:
            self.yaml_saver = YamlSaver(self.root)


    def save_sample(self, sample: DatasetSample):
        self.img_saver.save_sample(step=self.__internal_step, data=sample)
        self.yaml_saver.save_sample(step=self.__internal_step, data=sample)
        self.__internal_step += 1
    
    def save_intrinsics(self, intrinsics: np.ndarray):
        pass 

    def close(self):
        self.yaml_saver.close()
        


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
        raw_path =       str(self.dir_dict[DataTypes.RAW.name] / f"{step:05d}.png")
        segmented_path = str(self.dir_dict[DataTypes.SEGMENTED.name] / f"{step:05d}.png")
        cv2.imwrite(raw_path, data.raw_img)
        cv2.imwrite(segmented_path, data.segmented_img)

@dataclass
class NumpyYamlFormatter:
    """
    Class to format numpy array to yaml and vice versa
    """
    precision:int = 8
    max_line_width:int = 200

    def convert_to_yaml(self, data: np.ndarray)->str:
        flat_data = data.ravel()
        str_repr = np.array2string(flat_data, separator=',', precision=self.precision, max_line_width=self.max_line_width)
        return str_repr[1:-1]
    
    def convert_from_yaml(self, data: str, shape:Tuple)->np.ndarray:
        return np.fromstring(data, sep=',').reshape(shape)

@dataclass
class YamlSaver(ABC):
    root: Path 

    def __post_init__(self):
        self.numpy_fmt = NumpyYamlFormatter()
        file_name = self.root / "gt.yaml"
        if file_name.exists():

            msg = f"GT file: {file_name} already exists. Do you want to overwrite it? (y/n)"
            # if (input(msg) != "y"):
            #     print("exiting ...")
            #     exit() 

        self.file_handle = open(file_name, "wt", encoding="utf-8") 
    
    def save_sample(self, step:int, data: DatasetSample): 
        rot_str = self.numpy_fmt.convert_to_yaml(data.extrinsic_matrix[:3,:3])
        t_str = self.numpy_fmt.convert_to_yaml(data.extrinsic_matrix[:3,3])
        print(f"step: {step}")
        print(rot_str)
        data_dict = {f"{step:05d}":{ "rot_model2cam": rot_str, "t_model2cam": t_str}}
        # yaml.dump(data_dict, self.file_handle,Dumper= yaml.SafeDumper,default_flow_style=False, default_style='\"', width=200)
        # yaml.dump(data_dict, self.file_handle,Dumper= yaml.SafeDumper, line_break=None, width=200)
        yaml.dump(data_dict, self.file_handle,Dumper= yaml.SafeDumper,default_style="", line_break=None, width=200)

        # self.file_handle.write(: \n")
        self.file_handle.flush() 
    
    def close(self):
        self.file_handle.close()