from ambf6dpose.DataCollection.DatasetBuilder import DatasetSample
import cv2
import numpy as np
from numpy.linalg import inv
import tf_conversions.posemath as pm
from dataclasses import dataclass
from ambf6dpose import AbstractSimulationClient, RawSimulationData

np.set_printoptions(precision=3, suppress=True)


@dataclass
class SimulatorDataProcessor:
    simulation_client: AbstractSimulationClient

    def __post_init__(self):
        self.intrinsic_params = self.calculate_intrinsics()

    def calculate_intrinsics(self) -> np.ndarray:
        """
        Calculate camera intrinsics in opencv format
        """
        fvg = 1.2
        width = 640
        height = 480
        f = height / (2 * np.tan(fvg / 2))

        intrinsic_params = np.zeros((3, 3))
        intrinsic_params[0, 0] = f
        intrinsic_params[1, 1] = f
        intrinsic_params[0, 2] = width / 2
        intrinsic_params[1, 2] = height / 2
        intrinsic_params[2, 2] = 1.0
        return intrinsic_params

    def get_intrinsics(self) -> np.ndarray:
        return self.intrinsic_params

    def get_needle_extrinsics(self, raw_data: RawSimulationData) -> np.ndarray:
        T_WN = raw_data.needle_pose  # Needle to world
        T_FL = raw_data.camera_l_pose  # CamL to CamFrame
        T_WF = raw_data.camera_frame_pose  # CamFrame to world

        T_WL = T_WF.dot(T_FL)
        T_LN = inv(T_WL).dot(T_WN)  # Needle to CamL

        # Convert AMBF camera axis to Opencv Camera axis
        F = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        T_LN_CV2 = F.dot(T_LN)

        return T_LN_CV2

    def generate_dataset_sample(self) -> DatasetSample:
        raw_data = self.simulation_client.get_data()
        img = raw_data.camera_l_img
        seg_img = raw_data.camera_l_seg_img
        depth_img = raw_data.camera_l_depth

        # Get extrinsics
        T_LN_CV2 = self.get_needle_extrinsics(raw_data)  # Needle(N) to CamL (L) (T_LN)
        # Get intrinsics
        K = self.get_intrinsics()

        return DatasetSample(img, seg_img, depth_img, T_LN_CV2, K)


if __name__ == "__main__":
    from ambf6dpose import RosInterface, SyncRosInterface, AMBFClientWrapper
    import time

    # Client - 3 options: AMBFClientWrapper, RosInterface, SyncRosInterface

    # client = AMBFClientWrapper()
    client = SyncRosInterface()
    # client = RosInterface()

    client.wait_until_first_sample(timeout=10)

    # Collect and process data
    sim_interface = SimulatorDataProcessor(client)
    sample = sim_interface.generate_dataset_sample()
    sample.generate_blended_img()
    cv2.imshow("blended", sample.blended_img)
    cv2.waitKey(0)
