from ambf6dpose.DataCollection.DatasetBuilder import DatasetSample
import cv2
import numpy as np
from numpy.linalg import inv
from surgical_robotics_challenge.scene import Scene
from surgical_robotics_challenge.camera import Camera
from ambf_client import Client
import time
import tf_conversions.posemath as pm
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
from surgical_robotics_challenge.simulation_manager import SimulationManager
from surgical_robotics_challenge.ecm_arm import ECM
from dataclasses import dataclass, field


np.set_printoptions(precision=3, suppress=True)

class ImageSub:
    def __init__(self):
        self.bridge = CvBridge()
        self.img_subs = rospy.Subscriber(
            "/ambf/env/cameras/cameraL/ImageData", Image, self.left_callback
        )
        self.seg_img_subs = rospy.Subscriber(
            "/ambf/env/cameras/cameraL2/ImageData", Image, self.seg_left_callback
        )
        self.left_frame = None
        self.left_ts = None
        self.seg_left_frame = None
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
class SimulationInterface:
    sim_manager: SimulationManager = field(default_factory=lambda : SimulationManager("Collect6dpose"))

    def __post_init__(self):
        self.img_subs = ImageSub()

        self.scene = Scene(self.sim_manager)  # Provides access to needle and entry/exit points
        self.ambf_cam_l = Camera(self.sim_manager, "/ambf/env/cameras/cameraL")
        self.ambf_cam_frame = ECM(self.sim_manager, "CameraFrame")

        self.calculate_intrinsics()

        assert self.ambf_cam_l is not None, "CameraL not found"

    def calculate_intrinsics(self):
        # Calculate camera intrinsics in opencv format
        fvg = 1.2
        width = 640
        height = 480
        f = height / (2 * np.tan(fvg / 2))

        self.intrinsic_params = np.zeros((3, 3))
        self.intrinsic_params[0, 0] = f
        self.intrinsic_params[1, 1] = f
        self.intrinsic_params[0, 2] = width / 2
        self.intrinsic_params[1, 2] = height / 2
        self.intrinsic_params[2, 2] = 1.0
    
    def get_intrinsics(self)->np.ndarray:
        return self.intrinsic_params
    
    def get_needle_extrinsics(self) ->np.ndarray:
        T_WN = pm.toMatrix(self.scene.needle_measured_cp())  # Needle to world
        T_FL = pm.toMatrix(self.ambf_cam_l.get_T_c_w())  # CamL to CamFrame
        T_WF = pm.toMatrix(self.ambf_cam_frame.get_T_c_w())  # CamFrame to world

        T_WL = T_WF.dot(T_FL)
        T_LN = inv(T_WL).dot(T_WN) #Needle to CamL

        # Convert AMBF camera axis to Opencv Camera axis
        F = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        T_LN_CV2 = F.dot(T_LN)

        return T_LN_CV2 
    
    def generate_dataset_sample(self) -> DatasetSample:
        img = self.img_subs.left_frame
        seg_img = self.img_subs.seg_left_frame
        # Get extrinsics
        T_LN_CV2 = self.get_needle_extrinsics() # Needle to CamL
        # Get intrinsics
        K = self.get_intrinsics()
        return DatasetSample(img,seg_img, T_LN_CV2, K)


if __name__ == "__main__":
    sim_interface = SimulationInterface()

    img = sim_interface.img_subs.left_frame
    T_LN_CV2 = sim_interface.get_needle_extrinsics() 

    # Project center of the needle with OpenCv
    rvecs, _ = cv2.Rodrigues(T_LN_CV2[:3, :3])
    tvecs = T_LN_CV2[:3, 3]

    # needle_salient points
    theta = np.linspace(np.pi / 3, np.pi, num=8).reshape((-1, 1))
    radius = 0.1018 / 10
    needle_salient = radius * np.hstack((np.cos(theta), np.sin(theta), theta * 0))

    # Project points
    img_pt, _ = cv2.projectPoints(
        needle_salient,
        rvecs,
        tvecs,
        sim_interface.get_intrinsics(),
        np.float32([0, 0, 0, 0, 0]),
    )


    # Display image
    for i in range(img_pt.shape[0]):
        img = cv2.circle(img, (int(img_pt[i, 0, 0]), int(img_pt[i, 0, 1])), 3, (255, 0, 0), -1)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
