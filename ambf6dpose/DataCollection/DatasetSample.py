from dataclasses import dataclass, field
import cv2
import numpy as np


@dataclass
class DatasetSample:
    raw_img: np.ndarray
    segmented_img: np.ndarray
    depth_img: np.ndarray
    extrinsic_matrix: np.ndarray
    intrinsic_matrix: np.ndarray
    gt_vis_img: np.ndarray = field(default=None, init=False)

    def generate_gt_vis(self) -> None:
        T_LN_CV2 = self.extrinsic_matrix
        img = self.raw_img.copy()

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

        # Display image
        for i in range(img_pt.shape[0]):
            img = cv2.circle(img, (int(img_pt[i, 0, 0]), int(img_pt[i, 0, 1])), 3, (255, 0, 0), -1)

        self.gt_vis_img = img
