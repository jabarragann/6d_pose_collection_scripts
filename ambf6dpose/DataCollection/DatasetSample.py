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
    intrinsic_matrix: np.ndarray
    gt_vis_img: np.ndarray = field(default=None, init=False)

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

    def generate_gt_vis(self) -> None:
        img = self.raw_img.copy()

        # Project needle points on image
        img_pt = self.project_needle_points()
        for i in range(img_pt.shape[0]):
            img = cv2.circle(
                img, (int(img_pt[i, 0, 0]), int(img_pt[i, 0, 1])), 3, (255, 0, 0), -1
            )

        # Draw axis on tool yaw links
        img = self.draw_axis(
            img,
            self.psm1_toolpitchlink_pose,
        )
        img = self.draw_axis(
            img,
            self.psm2_toolpitchlink_pose,
        )
        img = self.draw_axis(
            img,
            self.psm1_toolyawlink_pose,
        )
        img = self.draw_axis(
            img,
            self.psm2_toolyawlink_pose,
        )

        self.gt_vis_img = img

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
