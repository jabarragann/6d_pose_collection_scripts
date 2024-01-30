from enum import Enum, auto

class ErrorRecordHeader(Enum):
    """Headers to build a pandas dataframe with error metrics"""

    # Info
    scene_im_id = auto()
    scene_id = auto()
    im_id = auto()
    obj_id = auto()
    visib_fract = "Visibility fraction"
    dist_to_cam = "Distance to camera (m)"
    # Rotation and translation error
    re = "rotation error (deg)"
    te = "translation error (mm)" 
    # delta x
    dx = auto()
    dy = auto()
    dz = auto()
    # mssd - 
    mssd = "Maximum Symmetry-Aware Surface Distance (mm)"

    @classmethod
    def is_a_error_metric(cls, header):
        error_list = [cls.re, cls.te, cls.dx, cls.dy, cls.dz, cls.mssd]
        return header in error_list