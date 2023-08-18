from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from ambf6dpose.DataCollection.DatasetSample import DatasetSample
import numpy as np
from enum import Enum, auto
import cv2

class ImgDirs(Enum):
    RAW = auto()
    SEGMENTED = auto()
    DEPTH = auto()
    GT_VISUALIZATION = auto()

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
class AbstractSaver(ABC):
    root:Path
        
    @abstractmethod
    def save_sample(self, sample: DatasetSample) -> None:
        pass



@dataclass
class AbstractReader(ABC):
    root: Path
    

@dataclass
class ImageSaver:
    """ Utility to save images into a folder structure defined by
    `folder_names`.  The dict `folder_names` needs to include values for all the
    constants defined in `ImgDirs` enum.
    """

    root_path: Path
    folder_names: Dict[ImgDirs, str] 

    def __post_init__(self):
        self.root_path.mkdir(exist_ok=True)
        self.dir_dict = self.get_img_dirs()
        self.create_dir()

    def get_img_dirs(self) -> Dict[ImgDirs, Path]:
        img_dirs = {}
        for img_dir in ImgDirs:
            img_dirs[img_dir] = self.root_path / self.folder_names[img_dir]
        return img_dirs

    def create_dir(self):
        for dir in self.dir_dict.values():
            dir.mkdir(exist_ok=True)

    def save_sample(self, str_step: str, data: DatasetSample):
        raw_path = str(self.dir_dict[ImgDirs.RAW] / f"{str_step}.png")
        segmented_path = str(self.dir_dict[ImgDirs.SEGMENTED] / f"{str_step}.png")
        cv2.imwrite(raw_path, data.raw_img)
        cv2.imwrite(segmented_path, data.segmented_img)

        # Save depth
        save_depth(self.dir_dict[ImgDirs.DEPTH] / f"{str_step}.png", data.depth_img)

        data.generate_blended_img()
        blended_path = str(self.dir_dict[ImgDirs.GT_VISUALIZATION] / f"{str_step}.png")
        cv2.imwrite(blended_path, data.blended_img)

def save_depth(path:Path, im:np.ndarray):
  """Saves a depth image (16-bit) to a PNG file.

  :param path: Path to the output depth image file.
  :param im: ndarray with the depth image to save.
  """
  path = str(path)
  if path.split('.')[-1].lower() != 'png':
    raise ValueError('Only PNG format is currently supported.')

  im_uint16 = np.round(im).astype(np.uint16)

  # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
  w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
  with open(path, 'wb') as f:
    w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))