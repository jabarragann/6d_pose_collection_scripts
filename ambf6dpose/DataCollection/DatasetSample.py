from dataclasses import dataclass, field
import cv2
import numpy as np
from enum import Enum, auto 

class RigidObjectsIds(Enum):
    needle_pose = 0 
    psm1_toolpitchlink_pose = 1
    psm1_toolyawlink_pose = 2
    psm2_toolpitchlink_pose = 3
    psm2_toolyawlink_pose = 4
    psm1_baselink_pose = 5 # ------------------added by Shuang, the base to left_camera

@dataclass
class DatasetSample:
    """
    Dataset processed samples. All rigid bodies are specified with respect the
    camera left frame.
    """

    raw_img: np.ndarray
    segmented_img: np.ndarray
    depth_img: np.ndarray
    needle_pose: np.ndarray
    psm1_toolpitchlink_pose: np.ndarray
    psm2_toolpitchlink_pose: np.ndarray
    psm1_toolyawlink_pose: np.ndarray
    psm2_toolyawlink_pose: np.ndarray
    psm1_baselink_pose: np.ndarray # ---------------------added by Shuang
    intrinsic_matrix: np.ndarray
    gt_vis_img: np.ndarray = field(default=None, init=False)
    gt_vis_img_right: np.ndarray = field(default=None, init=False)
    toolpitch_to_base_pose: np.ndarray = field(default=None, init=False)

    def project_needle_points(self) -> np.ndarray:
        T_LN_CV2 = self.needle_pose

        # Project center of the needle with OpenCv
        rvecs, _ = cv2.Rodrigues(T_LN_CV2[:3, :3])
        tvecs = T_LN_CV2[:3, 3]

        # needle_salient points
        theta = np.linspace(np.pi / 3, np.pi, num=8).reshape((-1, 1))
        radius = 0.1018 / 10 * 1000
        needle_salient = radius * np.hstack((np.cos(theta), np.sin(theta), theta * 0))

        # Project points
        img_pt, _ = cv2.projectPoints(
            needle_salient,
            rvecs,
            tvecs,
            self.intrinsic_matrix,
            np.float32([0, 0, 0, 0, 0]),
        )

        return img_pt
    

    
    def project_feature_points(self) -> np.ndarray:

        # tool pitch link to left camera
        T_LeftCamera_PitchLink = self.psm1_toolpitchlink_pose 

        # Project feature point1 with OpenCv
        rvecs, _ = cv2.Rodrigues(T_LeftCamera_PitchLink[:3, :3])
        tvecs = T_LeftCamera_PitchLink[:3, 3]

        # feature point1 is just the origin of the featurepoint1 frame
        FeaturePoints = np.array([[6.1, -2.005, -2.949], # Point 1
                                  [6.1, 1.895, -2.951], # Point 2
                                  [0,0,-3.9]], # Point 3
                                 dtype=np.float)

        # Project first two points
        img_pt, _ = cv2.projectPoints(
            FeaturePoints,
            rvecs,
            tvecs,
            self.intrinsic_matrix,
            np.float32([0, 0, 0, 0, 0]),
        )

        return img_pt 

    def project_feature_points_newtool(self) -> np.ndarray:

        # tool pitch link to left camera
        T_LeftCamera_PitchLink = self.psm1_toolpitchlink_pose 

        # Project feature point1 with OpenCv
        rvecs, _ = cv2.Rodrigues(T_LeftCamera_PitchLink[:3, :3])
        tvecs = T_LeftCamera_PitchLink[:3, 3]

        # feature point1 is just the origin of the featurepoint1 frame
        FeaturePoints = np.array([[0.0, 0.0, 4], # Point 1
                                  [1.951, -1.442, 2.71], # Point2
                                  [3.307, -1.629, 3.2], # Point 3
                                  [5.349,-1.676, 2.71], # Point 4
                                  [2.891, -2.975, -0.368], # Point 5
                                  [9.270, -2.7665, 0]], # Point 6
                                 dtype=np.float)

        # Project first two points
        img_pt, _ = cv2.projectPoints(
            FeaturePoints,
            rvecs,
            tvecs,
            self.intrinsic_matrix,
            np.float32([0, 0, 0, 0, 0]),
        )

        return img_pt

    def project_feature_points_right(self) -> np.ndarray:

        # tool pitch link to left camera
        T_LeftCamera_PitchLink = self.psm1_toolpitchlink_pose 

        # Define the left camera location and orientation (AMBF convention)
        left_location = np.array([-0.02, 0.0, -0.5])/10
        left_look_at = np.array([0.0, 0.0, -1.0])  # Negative X-axis direction
        left_look_up = np.array([0.0, 1.0, 0.0])   # Z-axis direction

        # Define the right camera location and orientation (AMBF convention)
        right_location = np.array([0.02, 0.0, -0.5])/10
        right_look_at = np.array([0.0, 0.0, -1.0])  # Negative X-axis direction
        right_look_up = np.array([0.0, 1.0, 0.0])   # Z-axis direction

        # Normalize vectors
        def normalize(v):
            return v / np.linalg.norm(v)

        # Compute camera axes for left camera
        left_X = normalize(-left_look_at)
        left_Z = normalize(left_look_up)
        left_Y = normalize(np.cross(left_Z, left_X))

        # Compute camera axes for right camera
        right_X = normalize(-right_look_at)
        right_Z = normalize(right_look_up)
        right_Y = normalize(np.cross(right_Z, right_X))

        # Rotation matrices (wrt camera frame)
        R_left = np.column_stack((left_X, left_Y, left_Z))
        R_right = np.column_stack((right_X, right_Y, right_Z))

        def create_transformation_matrix(position, orientation):
            T = np.eye(4)
            T[:3, :3] = orientation  # Assuming orientation is a 3x3 rotation matrix
            T[:3, 3] = position
            return T
        
        # left and right respect to camera frame 
        T_Camera_LeftCamera = create_transformation_matrix(left_location, R_left)
        T_Camera_RightCamera = create_transformation_matrix(right_location, R_right)


        # tool pitch link to right camera
        T_opencv_ambf = np.array([[0, 0, -1, 0],
                                 [1, 0, 0, 0],
                                 [0, -1, 0, 0],
                                 [0, 0, 0, 1]])
        T_ambf_opencv = np.linalg.inv(T_opencv_ambf)

        # 
        T_Camera_LeftCamera = np.array([
            [-0.0021, 1.0, 0.0, 0.002],
            [-0.0, -0.0, 1.0, 0.0],
            [1.0, 0.0021, 0.0, -0.05],
            [0.0, 0.0, 0.0, 1.0]
        ])


        T_RightCamera_PitchLink = T_opencv_ambf @ np.linalg.inv(T_Camera_RightCamera) @ T_Camera_LeftCamera  @ T_ambf_opencv @ T_LeftCamera_PitchLink 

        # Project feature point1 with OpenCv
        rvecs, _ = cv2.Rodrigues(T_RightCamera_PitchLink[:3, :3])
        tvecs = T_RightCamera_PitchLink[:3, 3]

        # feature points in tool frame (from blender file)
        FeaturePoints = np.array([[6.1, -2.005, -2.949], # Point 1
                                  [6.1, 1.895, -2.951], # Point 2
                                  [0,0,-3.9]], # Point 3
                                 dtype=np.float)

        # Project first two points
        img_pt, _ = cv2.projectPoints(
            FeaturePoints,
            rvecs,
            tvecs,
            self.intrinsic_matrix,
            np.float32([0, 0, 0, 0, 0]),
        )

        return img_pt
    
    def generate_gt_vis(self) -> None:
        img = self.raw_img.copy()

        # Project feature points
        img_pt = self.project_feature_points()
        for i in range(img_pt.shape[0]):
            if i == 0:
                color = (255, 255, 255)  # White color for the first point
            elif i == 1:
                color = (0, 0, 255)  # Red color for the second point
            elif i == 2:
                color = (255, 0, 0)  # Blue color for the third point
            else:
                color = (0, 255, 0)  # Default to red for additional points
            img = cv2.circle(
                img, (int(img_pt[i, 0, 0]), int(img_pt[i, 0, 1])), 5, color, -1
            )

        self.gt_vis_img = img
    
    # def generate_gt_vis_newtool(self) -> None:
    #     img = self.raw_img.copy()

    #     # Define colors for six points
    #     colors = [
    #         (255, 255, 255),  # White
    #         (0, 0, 255),      # Red
    #         (255, 0, 0),      # Blue
    #         (0, 255, 0),      # Green
    #         (255, 255, 0),    # Yellow
    #         (0, 255, 255)     # Cyan
    #     ]

    #     # Project feature points
    #     img_pt = self.project_feature_points_newtool()
    #     for i in range(img_pt.shape[0]):
    #         color = colors[i] if i < len(colors) else (255, 0, 255)  # Magenta for additional points
    #         img = cv2.circle(
    #             img, (int(img_pt[i, 0, 0]), int(img_pt[i, 0, 1])), 5, color, -1
    #         )

    #     self.gt_vis_img = img

    def generate_gt_vis_newtool(self) -> None:
        img = self.raw_img.copy()

        # Define colors for six points
        colors = [
            (255, 255, 255),  # White
            (0, 0, 255),      # Red
            (255, 0, 0),      # Blue
            (0, 255, 0),      # Green
            (255, 255, 0),    # Yellow
            (0, 255, 255)     # Cyan
        ]

        # Project feature points
        img_pt = self.project_feature_points_newtool()
        for i in range(img_pt.shape[0]):
            if i < len(colors):
                color = colors[i]
            else:
                color = (255, 0, 255)  # Magenta for additional points

            point = (int(img_pt[i, 0, 0]), int(img_pt[i, 0, 1]))
            
            if 0 <= point[0] < img.shape[1] and 0 <= point[1] < img.shape[0]:
                # Draw the point
                img = cv2.circle(img, point, 5, color, -1)

                # Draw the arrow pointing to the point with the number
                arrow_start = (point[0] - 30, point[1] - 30)
                img = cv2.arrowedLine(img, arrow_start, point, color, 1, tipLength=0.3)
                
                # Put the number of the point
                text = f"{i+1}"
                img = cv2.putText(img, text, (arrow_start[0] - 20, arrow_start[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        self.gt_vis_img = img

    def generate_gt_vis_right(self) -> None:
        img = self.segmented_img.copy()

        # Project feature points
        img_pt = self.project_feature_points_right()
        for i in range(img_pt.shape[0]):
            if i == 0:
                color = (255, 255, 255)  # White color for the first point
            elif i == 1:
                color = (0, 0, 255)  # Red color for the second point
            elif i == 2:
                color = (255, 0, 0)  # Blue color for the third point
            else:
                color = (255, 0, 0)  # Default to red for additional points
            img = cv2.circle(
                img, (int(img_pt[i, 0, 0]), int(img_pt[i, 0, 1])), 5, color, -1
            )

        self.gt_vis_img_right = img

    
    def draw_axis(self, img, pose):
        s = 3
        thickness = 2
        R, t = pose[:3, :3], pose[:3, 3]
        K = self.intrinsic_matrix
        # unit is mm
        rotV, _ = cv2.Rodrigues(R)
        points = np.float32([[s, 0, 0], [0, s, 0], [0, 0, s], [0, 0, 0]]).reshape(-1, 3)
        axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
        axisPoints = axisPoints.astype(int)

        img = cv2.line(
            img,
            tuple(axisPoints[3].ravel()),
            tuple(axisPoints[0].ravel()),
            (255, 0, 0),
            thickness,
        )
        img = cv2.line(
            img,
            tuple(axisPoints[3].ravel()),
            tuple(axisPoints[1].ravel()),
            (0, 255, 0),
            thickness,
        )

        img = cv2.line(
            img,
            tuple(axisPoints[3].ravel()),
            tuple(axisPoints[2].ravel()),
            (0, 0, 255),
            thickness,
        )
        return img
    
    def get_intrinsic(self):
        return self.intrinsic_matrix
    
    def get_tool_pitch_pose(self):
        return self.psm1_toolpitchlink_pose
    
    def generate_pitch_to_base(self) -> None:
        self.toolpitch_to_base_pose = np.linalg.inv(self.psm1_baselink_pose) @ self.psm1_toolpitchlink_pose
