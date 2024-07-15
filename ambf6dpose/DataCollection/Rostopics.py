from __future__ import annotations
from typing import Any, Callable, Dict
from sensor_msgs.msg import Image, PointCloud2
from ambf_msgs.msg import CameraState
from ambf_msgs.msg import RigidBodyState
from enum import Enum
import tf_conversions.posemath as pm
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from cv_bridge import CvBridge, CvBridgeError
import PyKDL
from surgical_robotics_challenge.units_conversion import SimToSI
import numpy as np
import ros_numpy


##############################
# Poses configuration 
##############################

# All topics in RosTopics will stored with the sync recoder
class RosTopics(Enum):
    CAMERA_L_STATE = ("/ambf/env/cameras/cameraL/State", CameraState)
    CAMERA_FRAME = ("/ambf/env/CameraFrame/State", RigidBodyState)
    NEEDLE = ("/ambf/env/Needle_001_001/State", RigidBodyState)
    CAMERA_L_IMAGE = ("/ambf/env/cameras/cameraL/ImageData", Image)
    CAMERA_L_SEG_IMAGE = ("/ambf/env/cameras/cameraR/ImageData", Image)
    # CAMERA_L_DEPTH = ("/ambf/env/cameras/cameraL/DepthData", PointCloud2)
    PSM1_TOOL_PITCH_LINK = ("/ambf/env/psm1/toolpitchlink/State", RigidBodyState)
    PSM2_TOOL_PITCH_LINK = ("/ambf/env/psm2/toolpitchlink/State", RigidBodyState)
    PSM1_TOOL_YAW_LINK = ("/ambf/env/psm1/toolyawlink/State", RigidBodyState)
    PSM2_TOOL_YAW_LINK = ("/ambf/env/psm2/toolyawlink/State", RigidBodyState)
    PSM1_BASE_LINK = ("/ambf/env/psm1/baselink/State", RigidBodyState) # --------------------added by Shuang


# Association between rostopics and the corresponding attribute in RosClients.RawSimulationData
# This dictionary is used to populate a RawSimulationData construction
topic_to_attr_dict = {
    RosTopics.CAMERA_L_STATE: "camera_l_pose",
    RosTopics.CAMERA_FRAME: "camera_frame_pose",
    RosTopics.NEEDLE: "needle_pose",
    RosTopics.CAMERA_L_IMAGE: "camera_l_img",
    RosTopics.CAMERA_L_SEG_IMAGE: "camera_l_seg_img",
    # RosTopics.CAMERA_L_DEPTH: "camera_l_depth",
    RosTopics.PSM1_TOOL_PITCH_LINK: "psm1_toolpitchlink_pose",
    RosTopics.PSM2_TOOL_PITCH_LINK: "psm2_toolpitchlink_pose",
    RosTopics.PSM1_TOOL_YAW_LINK: "psm1_toolyawlink_pose",
    RosTopics.PSM2_TOOL_YAW_LINK: "psm2_toolyawlink_pose",
    RosTopics.PSM1_BASE_LINK: "psm1_baselink_pose", # --------------------added by Shuang
}


def get_topics_processing_cb() -> Dict[RosTopics, Callable[[Any]]]:
    image_processor = get_image_processor()
    point_cloud_processor = get_point_cloud_processor()

    TopicsProcessingCb = {
        RosTopics.CAMERA_L_STATE: processing_pose_data,
        RosTopics.CAMERA_FRAME: processing_pose_data,
        RosTopics.NEEDLE: processing_pose_data,
        RosTopics.CAMERA_L_IMAGE: image_processor,
        RosTopics.CAMERA_L_SEG_IMAGE: image_processor,
        # RosTopics.CAMERA_L_DEPTH: point_cloud_processor,
        RosTopics.PSM1_TOOL_PITCH_LINK: processing_pose_data,
        RosTopics.PSM2_TOOL_PITCH_LINK: processing_pose_data,
        RosTopics.PSM1_TOOL_YAW_LINK: processing_pose_data,
        RosTopics.PSM2_TOOL_YAW_LINK: processing_pose_data,
        RosTopics.PSM1_BASE_LINK: processing_pose_data, # --------------------added by Shuang
    }

    return TopicsProcessingCb

##############################
# Utility functions 
##############################

def convert_units(frame: PyKDL.Frame):
    scaled_frame = PyKDL.Frame(frame.M, frame.p / SimToSI.linear_factor)
    return scaled_frame


def processing_pose_data(msg: RigidBodyState) -> np.ndarray:
    return pm.toMatrix(convert_units(pm.fromMsg(msg.pose)))


def get_image_processor():
    bridge = CvBridge()

    def process_img(msg: Image) -> np.ndarray:
        return bridge.imgmsg_to_cv2(msg, "bgr8")

    return process_img


def get_point_cloud_processor():
    w = 640
    h = 480
    scale = (1 / SimToSI.linear_factor) * 1000  # convert to from simulation units to mm
    extrinsic = np.array(
        [[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
    )  # T_cv_ambf

    def process_point_cloud(msg: PointCloud2) -> np.ndarray:
        xyz_array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        xcol = xyz_array["x"][:, None] * scale
        ycol = xyz_array["y"][:, None] * scale
        zcol = xyz_array["z"][:, None] * scale

        scaled_depth = np.concatenate([xcol, ycol, zcol], axis=-1)
        # # halve precision to save storage
        # scaled_depth = scaled_depth.astype(np.float16)
        # reverse height direction due to AMBF reshaping
        scaled_depth = np.ascontiguousarray(scaled_depth.reshape([h, w, 3])[::-1])
        # convert to cv convention
        scaled_depth = np.einsum("ab,hwb->hwa", extrinsic[:3, :3], scaled_depth)[
            ..., -1
        ]

        # scaled_depth = np.round(scaled_depth).astype(np.uint16)
        # print(scaled_depth.shape)
        # print(scaled_depth.max())
        # print(scaled_depth.min())

        return scaled_depth

    return process_point_cloud
