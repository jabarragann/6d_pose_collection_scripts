from ambf_client import Client
import time
from PyKDL import Vector, Frame, Rotation
from ambf6dpose.SimulationObjs.psm_lnd.PSM_LND_FK import LNDForwardKinematic
from threading import Thread, Lock
import numpy as np


class LNDJointMapping:
    def __init__(self):
        # self.idx_to_name = {0: 'maininsertionlink-toolrolllink',
        #                     1: 'toolrolllink-toolpitchlink',
        #                     2: 'toolpitchlink-toolyawlink',
        #                     3: 'toolyawlink-toolgripperleftlink',
        #                     4: 'toolyawlink-toolgripperrightlink'}
        self.idx_to_name = {
            0: "maininsertionlink-toolrolllink",
            1: "toolrolllink-toolpitchlink",
            2: "toolpitchlink-toolyawlink",
        }

        # self.name_to_idx = {'maininsertionlink-toolrolllink': 0,
        #                     'toolrolllink-toolpitchlink': 1,
        #                     'toolpitchlink-toolyawlink': 2,
        #                     'toolyawlink-toolgripperleftlink': 3,
        #                     'toolyawlink-toolgripperrightlink': 4}
        self.name_to_idx = {
            "maininsertionlink-toolrolllink": 0,
            "toolrolllink-toolpitchlink": 1,
            "toolpitchlink-toolyawlink": 2,
        }


def conver_xyz_rpy_to_frame(
    x: float, y: float, z: float, roll: float, pitch: float, yaw: float
) -> Frame:
    rot = Rotation.RPY(roll, pitch, yaw)
    pos = Vector(x, y, z)
    return Frame(rot, pos)


ljm = LNDJointMapping()


class LND:
    def __init__(self, namespace: str, client: Client):
        self.client = client
        self.base = self.client.get_obj_handle(namespace + "tool_main_insert")
        time.sleep(0.5)

        ### base transformation info
        self.T_t_b_home = Frame(Rotation.RPY(0.0, 0.0, 0.0), Vector(0.0, 0.0, 0.0))
        self._kd = LNDForwardKinematic()

        ## init
        self._T_b_w = None
        self._T_w_b = None
        self._base_pose_updated = False
        self._num_joints = 3
        self._ik_solution = np.zeros([self._num_joints])
        self._force_exit_thread = False
        self._thread_lock = Lock()
        self.set_jaw_angle(0.5)
        time.sleep(0.5)

    def set_home_pose(self, pose):
        self.T_t_b_home = pose

    def is_present(self):
        if self.base is None:
            return False
        else:
            return True

    def get_lower_limits(self):
        return self._kd.lower_limits

    def get_upper_limits(self):
        return self._kd.upper_limits

    def get_T_w_b(self):
        self.__update_base_pose()
        return self._T_w_b

    def get_T_b_w(self):
        self.__update_base_pose()
        return self._T_b_w

    def __update_base_pose(self):
        if not self._base_pose_updated:
            self._T_b_w = conver_xyz_rpy_to_frame(*self.base.get_pose())

            self._T_w_b = self._T_b_w.Inverse()
            self._base_pose_updated = True

    def servo_cp(self, T_t_b):
        pass

    def servo_jp(self, jp):
        self.base.set_joint_pos(0, jp[0])
        self.base.set_joint_pos(1, jp[1])
        self.base.set_joint_pos(2, jp[2])

    def servo_jv(self, jv):
        print("Setting Joint Vel: ", jv)
        self.base.set_joint_vel(0, jv[0])
        self.base.set_joint_vel(1, jv[1])
        self.base.set_joint_vel(2, jv[2])

    def set_jaw_angle(self, jaw_angle):
        self.base.set_joint_pos(3, -jaw_angle)
        self.base.set_joint_pos(4, -jaw_angle)

    def measured_jp(self):
        j0 = self.base.get_joint_pos(0)
        j1 = self.base.get_joint_pos(1)
        j2 = self.base.get_joint_pos(2)
        q = [j0, j1, j2]
        return q

    def measured_jv(self):
        j0 = self.base.get_joint_vel(0)
        j1 = self.base.get_joint_vel(1)
        j2 = self.base.get_joint_vel(2)
        return [j0, j1, j2]

    def get_joint_names(self):
        return self.base.get_joint_names()


if __name__ == "__main__":

    client = Client("LNDSimulation")
    client.connect()
    lnd = LND("/new_psm1/", client)

    lnd.servo_jp([0, 1, -1])
    lnd.set_jaw_angle(0.0)
    time.sleep(1.0)

    T_b_w_1 = lnd.get_T_b_w()

    for i in range(1, 4):
        pos = T_b_w_1.p
        pos = [pos.x(), pos.y(), pos.z()]
        lnd.base.set_pos(pos[0] + 0.005 * i, pos[1], pos[2])
        time.sleep(1)

    print("returning")
    lnd.base.set_pos(T_b_w_1.p.x(), T_b_w_1.p.y(), T_b_w_1.p.z())
    lnd.servo_jp([0, 0, 0])
    lnd.set_jaw_angle(0.0)
    time.sleep(2)

    print("Test Done")
