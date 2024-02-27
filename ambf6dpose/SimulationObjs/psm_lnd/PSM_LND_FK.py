import numpy as np
from typing import Union
from ambf6dpose.SimulationObjs.psm_lnd.utils import twist2ht, skew


class LNDForwardKinematic:
    def __init__(self):
        self.num_link = 3  # 5 if adding two gripper

        ### the following values are readings from Blender
        self.link_base_main = 0 - (-0.050825)
        self.link_main_roll = -0.050825 - (-0.48263)
        self.link_roll_pitch = -0.48263 - (-0.491617)
        self.screw_axis = self.__get_screw_axis()
        self.M = self.__get_M()
        self.lower_limits = [
            np.deg2rad(-175),
            np.deg2rad(-90),
            np.deg2rad(-85),
            np.deg2rad(-90),
            np.deg2rad(-90),
        ]
        self.upper_limits = [
            np.deg2rad(175),
            np.deg2rad(90),
            np.deg2rad(85),
            np.deg2rad(90),
            np.deg2rad(90),
        ]

    def __get_screw_axis(self) -> np.ndarray:
        """
        Obtain the screw axes for the robot model
        :return: nx3 matrix for screw axes of all n joints
        """
        self.w1 = np.array([1, 0, 0])
        self.w2 = np.array([0, 0, -1])
        self.w3 = np.array([0, -1, 0])
        self.w4 = np.array([0, 1, 0])
        self.w5 = np.array([0, -1, 0])
        self.p1 = np.array([self.link_base_main, 0, 0])
        self.p2 = np.array([self.link_base_main + self.link_main_roll, 0, 0])
        self.p3 = np.array(
            [self.link_base_main + self.link_main_roll + self.link_roll_pitch, 0, 0]
        )
        self.p4 = self.p3
        self.p5 = self.p3

        for i_v in range(self.num_link):
            exec(f"self.v{i_v+1} = -skew(self.w{i_v+1}) @ self.p{i_v+1}")

        screw_axes = np.zeros((self.num_link, 6))
        for i_s in range(self.num_link):
            exec(f"screw_axes[{i_s}, :] = np.hstack((self.w{i_s+1}, self.v{i_s+1}))")

        return screw_axes

    def __get_M(self) -> np.ndarray:
        """
        Find the transformation matrix M between the robot base and the end-effector
        :return: a 4x4 homogeneous transformation matrix
        """
        M = np.array(
            [
                [
                    0,
                    0,
                    -1,
                    self.link_base_main + self.link_main_roll + self.link_roll_pitch,
                ],
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        return M

    def compute_FK(self, joint_val: Union[np.array, list]) -> np.ndarray:
        """
        Compute the forward kinematics for the robot
        :param q: joint values (angle in rad, distance in meter)
        :return: a 4x4 homogeneous transformation matrix
        """
        assert (
            len(joint_val) == self.num_link
        ), "The FK input should have the same length as the number of robot DOF"
        T = np.eye(4)
        for i_axis in range(self.num_link):
            S = self.screw_axis[i_axis, :]
            q = joint_val[i_axis]
            T_i = twist2ht(S, q)
            T = np.dot(T, T_i)
        T = np.dot(T, self.M)
        return T


if __name__ == "__main__":
    lnd = LNDForwardKinematic()
    T_0 = lnd.compute_FK([0, 0, 0])
    print("init pose: \n")
    print(T_0)
