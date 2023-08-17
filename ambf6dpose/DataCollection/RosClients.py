from __future__ import annotations
import time
from surgical_robotics_challenge.simulation_manager import SimulationManager
from surgical_robotics_challenge.ecm_arm import ECM
from surgical_robotics_challenge.scene import Scene
from surgical_robotics_challenge.camera import Camera
import numpy as np
import tf_conversions.posemath as pm
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError
import rospy
from dataclasses import dataclass, field
from abc import ABC
from ambf_msgs.msg import CameraState
from ambf_msgs.msg import RigidBodyState
import message_filters
from ambf6dpose.DataCollection.Rostopics import RosTopics, get_topics_processing_cb, topic_to_attr_dict



@dataclass
class RawSimulationData:
    camera_l_pose: np.ndarray
    camera_frame_pose: np.ndarray
    needle_pose: np.ndarray
    camera_l_img: np.ndarray
    camera_l_seg_img: np.ndarray
    camera_l_depth: np.ndarray

    def __post_init__(self):
        if self.has_none_members():
            raise ValueError("SimulationData cannot have None members")

    def has_none_members(self) -> bool:
        """Check if any member attribute is initialized to None

        Taken from
        https://www.geeksforgeeks.org/how-to-get-a-list-of-class-attributes-in-python/#
        """
        attrs_values = list(vars(self).values())
        return any([e is None for e in attrs_values])

    @classmethod
    def from_dict(cls:RawSimulationData, data: dict[RosTopics,np.ndarray]):
        #Map the keys to the class attributes
        dict_variables = {}
        for (key,value) in data.items():
            dict_variables[topic_to_attr_dict[key]] = value
        # dict_variable = {cls.topic_to_attr[key]:value for (key,value) in data.items()}

        return cls(**dict_variables)

@dataclass
class AbstractSimulationClient(ABC):
    """
    Abstract ros client for collecting data from the simulation.

    * Derived classes from this abstract class will need default values for the
    attributes in python version less than 3.10.
    * Derived classes need to call super().__post_init__() in its __post_init__()

    https://medium.com/@aniscampos/python-dataclass-inheritance-finally-686eaf60fbb5
    """

    raw_data: RawSimulationData = field(default=None, init=False)
    client_name = "ambf_collection_client"

    def __post_init__(self):
        if "/unnamed" == rospy.get_name():
            rospy.init_node(self.client_name)
            time.sleep(0.2)
        else:
            self._client_name = rospy.get_name()

    def get_data(self) -> RawSimulationData:
        if self.raw_data is None:
            raise ValueError("No data has been received")

        return self.raw_data

    def has_data(self) -> bool:
        return self.raw_data is not None

    def wait_until_first_sample(self, verbose=True, timeout=10) -> None:
        init_time = last_time = time.time()
        while not self.has_data():
            time.sleep(0.1)
            last_time = time.time()
            if last_time - init_time > timeout:
                raise TimeoutError(f"Timeout waiting for first sample. Timeout: {timeout}s")

        if verbose:
            print(f"Time to get first package data: {last_time - init_time:0.3f}s")


@dataclass
class AMBFClientWrapper(AbstractSimulationClient):
    sim_manager: SimulationManager = field(
        default_factory=lambda: SimulationManager("Collect6dpose")
    )

    def __post_init__(self):
        super().__post_init__()
        self.img_subs = ImageSub()

        self.scene = Scene(self.sim_manager)  # Provides access to needle and entry/exit points
        self.ambf_cam_l = Camera(self.sim_manager, "/ambf/env/cameras/cameraL")
        self.ambf_cam_frame = ECM(self.sim_manager, "CameraFrame")

        assert self.ambf_cam_l is not None, "CameraL not found"
        assert self.ambf_cam_frame is not None, "CameraFrame not found"

    def has_data(self) -> bool:
        return True

    def get_data(self) -> RawSimulationData:
        needle_pose = pm.toMatrix(self.scene.needle_measured_cp())  # Needle to world
        camera_l_pose = pm.toMatrix(self.ambf_cam_l.get_T_c_w())  # CamL to CamFrame
        camera_frame_pose = pm.toMatrix(self.ambf_cam_frame.get_T_c_w())  # CamFrame to world
        raw_img_l = self.img_subs.left_frame
        seg_img_l = self.img_subs.seg_left_frame

        self.raw_data = RawSimulationData(
            camera_l_pose, camera_frame_pose, needle_pose, raw_img_l, seg_img_l
        )

        return self.raw_data


class ImageSub:
    def __init__(self):
        self.bridge = CvBridge()
        self.img_subs = rospy.Subscriber(
            RosTopics.CAMERA_L_IMAGE.value[0], Image, self.left_callback
        )
        self.seg_img_subs = rospy.Subscriber(
            RosTopics.CAMERA_L_SEG_IMAGE.value[0], Image, self.seg_left_callback
        )
        self.left_frame = np.zeros((640, 480, 3), dtype=np.uint8)
        self.left_ts = None
        self.seg_left_frame = np.zeros((640, 480, 3), dtype=np.uint8)
        self.seg_left_ts = None

        # Wait a until subscribers and publishers are ready
        rospy.sleep(0.5)

    def left_callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.left_frame = cv2_img
            self.left_ts = msg.header.stamp
        except CvBridgeError as e:
            print(e)

    def seg_left_callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.seg_left_frame = cv2_img
            self.seg_left_ts = msg.header.stamp
        except CvBridgeError as e:
            print(e)




@dataclass
class SyncRosInterface(AbstractSimulationClient):
    def __post_init__(self):
        super().__post_init__()
        self.bridge = CvBridge()
        self.subscribers = []
        self.callback_dict = get_topics_processing_cb()

        for topic in RosTopics:
            self.subscribers.append(message_filters.Subscriber(topic.value[0], topic.value[1]))

        # WARNING: TimeSynchronizer did not work. Use ApproximateTimeSynchronizer instead.
        # self.time_sync = message_filters.TimeSynchronizer(self.subscribers, 10)
        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, queue_size=10, slop=0.05
        )
        self.time_sync.registerCallback(self.cb)

        time.sleep(0.25)

    def cb(self, *inputs):
        raw_data_dict = {}
        for input_msg, topic in zip(inputs, RosTopics):
            raw_data_dict[topic] = self.callback_dict[topic](input_msg)

        self.raw_data = RawSimulationData.from_dict(raw_data_dict)
        #     if topic == RosTopics.CAMERA_L:
        #         # camera_l_pose = pm.toMatrix(convert_units(pm.fromMsg(input_msg.pose)))
        #         camera_l_pose = self.callback_dict[topic](input_msg)
        #     elif topic == RosTopics.CAMERA_FRAME:
        #         camera_frame_pose = pm.toMatrix(convert_units(pm.fromMsg(input_msg.pose)))
        #     elif topic == RosTopics.NEEDLE:
        #         needle_pose = pm.toMatrix(convert_units(pm.fromMsg(input_msg.pose)))
        #     elif topic == RosTopics.CAMERA_L_IMAGE:
        #         camera_l_img = self.bridge.imgmsg_to_cv2(input_msg, "bgr8")
        #     elif topic == RosTopics.CAMERA_L_SEG_IMAGE:
        #         camera_l_seg_img = self.bridge.imgmsg_to_cv2(input_msg, "bgr8")
        #     elif topic == RosTopics.CAMERA_L_DEPTH:
        #         camera_l_depth = self.callback_dict[topic](input_msg)

        # self.raw_data = RawSimulationData(
        #     camera_l_pose=camera_l_pose,
        #     camera_frame_pose=camera_frame_pose,
        #     needle_pose=needle_pose,
        #     camera_l_img=camera_l_img,
        #     camera_l_seg_img=camera_l_seg_img,
        #     camera_l_depth=camera_l_depth,
        # )


@dataclass
class RosInterface(AbstractSimulationClient):
    def __post_init__(self):
        super().__post_init__()
        self.bridge = CvBridge()
        self.camera_l_subs = rospy.Subscriber(
            RosTopics.CAMERA_L.value[0], CameraState, self.camera_l_callback
        )
        self.camera_l_img_subs = rospy.Subscriber(
            RosTopics.CAMERA_L_IMAGE.value[0], Image, self.camera_l_img_callback
        )
        self.camera_l_seg_img_subs = rospy.Subscriber(
            RosTopics.CAMERA_L_SEG_IMAGE.value[0], Image, self.camera_l_seg_img_callback
        )
        self.camera_frame_subs = rospy.Subscriber(
            RosTopics.CAMERA_FRAME.value[0], RigidBodyState, self.camera_frame_callback
        )
        self.needle_subs = rospy.Subscriber(
            RosTopics.NEEDLE.value[0], RigidBodyState, self.needle_callback
        )

        time.sleep(0.2)

    def camera_l_callback(self, msg):
        self.camera_l_pose = pm.toMatrix(convert_units(pm.fromMsg(msg.pose)))

    def camera_l_img_callback(self, msg):
        self.camera_l_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def camera_l_seg_img_callback(self, msg):
        self.camera_l_seg_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def camera_frame_callback(self, msg):
        self.camera_frame_pose = pm.toMatrix(convert_units(pm.fromMsg(msg.pose)))

    def needle_callback(self, msg):
        self.needle_pose = pm.toMatrix(convert_units(pm.fromMsg(msg.pose)))

    def has_data(self) -> bool:
        return (
            self.camera_l_pose is not None
            and self.camera_frame_pose is not None
            and self.needle_pose is not None
        )

    def get_data(self) -> RawSimulationData:
        self.raw_data = RawSimulationData(
            camera_l_pose=self.camera_l_pose,
            camera_frame_pose=self.camera_frame_pose,
            needle_pose=self.needle_pose,
            camera_l_img=self.camera_l_img,
            camera_l_seg_img=self.camera_l_seg_img,
        )

        return self.raw_data
