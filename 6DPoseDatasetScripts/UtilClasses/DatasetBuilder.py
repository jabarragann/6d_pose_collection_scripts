from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np

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
    dataset_saver: DatasetSaver 

@dataclass
class DatasetSaver:
    dir_manager: AbstractDirManager 
    yaml_formatter: AbstractYamlFormatter

class AbstractDirManager(ABC):
    pass

class AbstractYamlFormatter(ABC):
    pass