import time
import rospy
import tf_conversions.posemath as pm
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from surgical_robotics_challenge.units_conversion import SimToSI
from dataclasses import dataclass
import PyKDL
from ambf_msgs.msg import CameraState
from ambf_msgs.msg import RigidBodyState
from enum import Enum
import message_filters
from ambf6dpose.DataCollection.Rostopics import RosTopics

"""
TimeSynchronizer did not work. Cb was never called.
ApproximateTimeSynchronizer works.
"""


def convert_units(frame: PyKDL.Frame):
    scaled_frame = PyKDL.Frame(frame.M, frame.p / SimToSI.linear_factor)
    return scaled_frame


count = 1


@dataclass
class SyncRosInterface:
    def __post_init__(self):
        self.count = 1
        self.last_time = time.time()

        self.bridge = CvBridge()
        self.subscribers = []

        for topic in RosTopics:
            print(topic)
            self.subscribers.append(message_filters.Subscriber(topic.value[0], topic.value[1]))

        # WARNING: TimeSynchronizer did not work. Cb was never called.
        # self.time_sync = message_filters.TimeSynchronizer(self.subscribers, 10)
        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, 10, slop=0.05
        )
        self.time_sync.registerCallback(self.cb)

        time.sleep(0.2)

    def cb(self, *inputs):
        print(f"callback {self.count}")
        assert len(inputs) == len(RosTopics), "error in callback"

        for input_msg, topic in zip(inputs, RosTopics):
            if topic == RosTopics.CAMERA_L:
                camera_l_pose = pm.toMatrix(convert_units(pm.fromMsg(input_msg.pose)))
            elif topic == RosTopics.CAMERA_FRAME:
                camera_frame_pose = pm.toMatrix(convert_units(pm.fromMsg(input_msg.pose)))
            elif topic == RosTopics.NEEDLE:
                needle_pose = pm.toMatrix(convert_units(pm.fromMsg(input_msg.pose)))
            elif topic == RosTopics.CAMERA_L_IMAGE:
                camera_l_img = self.bridge.imgmsg_to_cv2(input_msg, "bgr8")
            elif topic == RosTopics.CAMERA_L_SEG_IMAGE:
                camera_l_seg_img = self.bridge.imgmsg_to_cv2(input_msg, "bgr8")

        self.count += 1

        time_lapse = time.time() - self.last_time
        freq = 1 / time_lapse
        print(f"hz {freq:0.3f} time lapse {time_lapse:0.3f}")
        self.last_time = time.time()


if __name__ == "__main__":
    rospy.init_node("test_sync")
    time.sleep(0.5)
    subscriber = SyncRosInterface()
    rospy.spin()
