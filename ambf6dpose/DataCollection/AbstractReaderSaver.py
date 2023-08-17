from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from ambf6dpose.DataCollection.DatasetSample import DatasetSample
import numpy as np

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
    
