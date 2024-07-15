from ambf6dpose.DataCollection.DatasetSample import DatasetSample
import cv2
import numpy as np
from numpy.linalg import inv
import tf_conversions.posemath as pm
from dataclasses import dataclass
from ambf6dpose import AbstractSimulationClient, RawSimulationData

np.set_printoptions(precision=3, suppress=True)


@dataclass
class SimulatorDataProcessor:
    """
    Express poses in terms of the camera left frame. Also, change units to mm to comply with BOP datasets.
    """

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
        needle_pose_in_caml = self.transform_from_world_to_cam_l(
            raw_data.needle_pose, raw_data.camera_l_pose, raw_data.camera_frame_pose
        )
        return self.convert_pose_to_mm(needle_pose_in_caml)

    def get_tool_pose(
        self, raw_data: RawSimulationData, raw_tool_pose: np.ndarray
    ) -> np.ndarray:
        tool_pose_in_caml = self.transform_from_world_to_cam_l(
            raw_tool_pose, raw_data.camera_l_pose, raw_data.camera_frame_pose
        )
        return self.convert_pose_to_mm(tool_pose_in_caml)

    def transform_from_world_to_cam_l(
        self,
        obj2world_pose: np.ndarray,
        caml2camframe_pose: np.ndarray,
        camframe2world_pose: np.ndarray,
    ) -> np.ndarray:
        """
        Transform objects defined in the world frame to the left camera frame.
        Camera left is attached to a camera frame in the surgical robotics
        assets. Therefore, the location of the left camera and the camera frame
        are needed.
        """
        T_W_OBJ = obj2world_pose
        T_FL = caml2camframe_pose
        T_WF = camframe2world_pose

        T_WL = T_WF @ T_FL  # CamL to world
        T_LW = inv(T_WL)  # World to CamL
        T_L_OBJ = T_LW @ T_W_OBJ  # Needle to CamL

        # Convert AMBF camera axis to Opencv Camera axis
        F = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        T_L_OBJ_CV2 = F.dot(T_L_OBJ)

        return T_L_OBJ_CV2

    def convert_pose_to_mm(self, pose: np.ndarray) -> np.ndarray:
        pose[:3, 3] = pose[:3, 3] * 1000
        return pose

    def generate_dataset_sample(self) -> DatasetSample:
        """Transformations are saved in mm to comply with the BOP standard."""
        raw_data = self.simulation_client.get_data()
        img = raw_data.camera_l_img
        seg_img = raw_data.camera_l_seg_img
        depth_img = np.zeros((300, 300))

        # Process object poses
        needle_in_caml_frame = self.get_needle_extrinsics(raw_data)
        psm1_toolpitchlink_in_caml_frame = self.get_tool_pose(
            raw_data, raw_data.psm1_toolpitchlink_pose
        )
        psm2_toolpitchlink_in_caml_frame = self.get_tool_pose(
            raw_data, raw_data.psm2_toolpitchlink_pose
        )
        psm1_toolyawlink_in_caml_frame = self.get_tool_pose(
            raw_data, raw_data.psm1_toolyawlink_pose
        )
        psm2_toolyawlink_in_caml_frame = self.get_tool_pose(
            raw_data, raw_data.psm2_toolyawlink_pose
        )
        psm1_toolbaselink_in_caml_frame = self.get_tool_pose( # -----------------added by Shuang
            raw_data, raw_data.psm1_baselink_pose
        )

        # Get intrinsics
        K = self.get_intrinsics()

        return DatasetSample(
            img,
            seg_img,
            depth_img,
            needle_in_caml_frame,
            psm1_toolpitchlink_in_caml_frame,
            psm2_toolpitchlink_in_caml_frame,
            psm1_toolyawlink_in_caml_frame,
            psm2_toolyawlink_in_caml_frame,
            psm1_toolbaselink_in_caml_frame, # -----------------added by Shuang
            K,
        )


if __name__ == "__main__":
    from ambf6dpose import SyncRosInterface
    import time

    # Client - 3 options: AMBFClientWrapper, RosInterface, SyncRosInterface

    # client = AMBFClientWrapper()
    client = SyncRosInterface()
    # client = RosInterface()

    client.wait_for_data(timeout=10)

    # Collect and process data
    sim_interface = SimulatorDataProcessor(client)
    sample = sim_interface.generate_dataset_sample()
    sample.generate_gt_vis()
    cv2.imshow("blended", sample.gt_vis_img)
    cv2.waitKey(0)
