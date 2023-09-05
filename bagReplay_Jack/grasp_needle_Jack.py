import PyKDL
from PyKDL import Vector, Rotation, Frame
from scipy.spatial.transform import Rotation as R
from surgical_robotics_challenge.utils.utilities import cartesian_interpolate_step_num
import numpy as np
import time
import rospy
import sys
from surgical_robotics_challenge.psm_arm import PSM
from surgical_robotics_challenge.ecm_arm import ECM
from surgical_robotics_challenge.simulation_manager import SimulationManager
from surgical_robotics_challenge.utils.utilities import cartesian_interpolate_step
from surgical_robotics_challenge.utils import coordinate_frames
from surgical_robotics_challenge.kinematics.psmFK import *
from surgical_robotics_challenge.kinematics.psmIK import *
from surgical_robotics_challenge.utils.utilities import convert_mat_to_frame
from surgical_robotics_challenge.utils.utilities import convert_frame_to_mat

def pykdl_to_np(T_pykdl):
    rot_des = R.from_quat(T_pykdl.M.GetQuaternion()).as_matrix()
    pos_des = np.array([T_pykdl.p.x(), T_pykdl.p.y(), T_pykdl.p.z()])
    T_np = np.eye(4)
    T_np[0:3, 0:3] = rot_des
    T_np[0:3, 3] = pos_des
    return T_np

def np_to_pykdl(T_np):
    rot_des = np.squeeze(np.array(T_np[0:3, 0:3].reshape(-1, 1))).tolist()
    pos_des = np.squeeze(np.array(T_np[0:3, 3])).tolist()
    v = Vector(pos_des[0], pos_des[1], pos_des[2])
    r = Rotation(rot_des[0], rot_des[1], rot_des[2],
                 rot_des[3], rot_des[4], rot_des[5],
                 rot_des[6], rot_des[7], rot_des[8])
    T_pydkl = Frame(r, v)
    return T_pydkl

class NeedleInitialization:
    def __init__(self, simulation_manager):
        self.T_needle_psmtip = coordinate_frames.Needle.T_center_psmtip
        self.T_needle_psmtip_far = self.T_needle_psmtip * Frame(Rotation.RPY(0., 0., 0.), Vector(0., 0., -0.010))

        self.needle = simulation_manager.get_obj_handle('Needle')
        time.sleep(1.0)
        self._release = False
        self._reached = False

    def get_tip_to_needle_offset(self):
        return self.T_needle_psmtip

    def move_to(self, psm_tip):
        print('Moving Needle to PSM 2 Tip')
        self._release = False
        if psm_tip is None:
            print('Not a valid link, returning')
            return
        T_nINw = self.needle.get_pose()
        T_tINw = psm_tip.get_pose()
        # First reach the farther point
        self._reached = False
        done = False
        while not done:
            T_nINw_cmd = T_tINw * self.T_needle_psmtip_far
            T_delta, done = cartesian_interpolate_step(T_nINw, T_nINw_cmd, 0.01, 0.005)
            r_delta = T_delta.M.GetRPY()
            # print(error_max)
            T_cmd = Frame()
            T_cmd.p = T_nINw.p + T_delta.p
            T_cmd.M = T_nINw.M * Rotation.RPY(r_delta[0], r_delta[1], r_delta[2])
            T_nINw = T_cmd
            self.needle.set_pose(T_cmd)
            time.sleep(0.01)

        time.sleep(0.5)
        done = False
        T_nINw = self.needle.get_pose()
        T_tINw = psm_tip.get_pose()
        while not done:
            T_nINw_cmd = T_tINw * self.T_needle_psmtip
            T_delta, done = cartesian_interpolate_step(T_nINw, T_nINw_cmd, 0.01, 0.005)
            r_delta = T_delta.M.GetRPY()
            T_cmd = Frame()
            T_cmd.p = T_nINw.p + T_delta.p
            T_cmd.M = T_nINw.M * Rotation.RPY(r_delta[0], r_delta[1], r_delta[2])
            T_nINw = T_cmd
            self.needle.set_pose(T_cmd)
            time.sleep(0.01)

        self._reached = True

    def release(self):
        print('Releasing Needle')
        self._release = True
        self.needle.set_force(Vector(0, 0, 0))
        self.needle.set_torque(Vector(0, 0, 0))

    def has_reached(self):
        return self._reached


if __name__ == "__main__":
    simulation_manager = SimulationManager('grasp_needle')
    time.sleep(0.2)
    w = simulation_manager.get_world_handle()
    time.sleep(0.2)
    w.reset_bodies()
    time.sleep(0.2)
    cam = ECM(simulation_manager, "CameraFrame")
    cam.servo_jp([0.0, 0.05, -0.01, 0.0])
    time.sleep(0.2)
    psm1 = PSM(simulation_manager, "psm1", add_joint_errors=False)
    time.sleep(0.2)
    psm2 = PSM(simulation_manager, "psm2", add_joint_errors=False)
    time.sleep(0.2)
    needle = simulation_manager.get_obj_handle('Needle')
    time.sleep(0.2)

    # init joint position from the recording of old phantom
    psm1_init = [0.30780306382205863, -0.22222915389237488, 0.1423643360325034,
                 -1.3613186165319513,0.5750600725456388, -0.8399263308008617]
    psm2_init = [-0.46695894800579796, -0.17860657808832947, 0.15012366098379068,
                 -1.0873261421084663, 0.7172512403887915, 0.48780102579228307]
    # # init joint position from the recording of 3d med phantom
    # psm1_init = [0.458790917435521, -0.02294350433827583, 0.13121405953236356,
    #              -1.5200111591037573, 0.7426201096608415, -1.3439108486545193]
    # psm2_init = [-0.445682016262167, -0.01655865108279217, 0.1294770684636348,
    #              1.6979817454682469, 0.6866264040948467, 1.064089891395347]

    ## offset of PSM
    offset_psm1 = PyKDL.Frame(Rotation.RPY(-np.pi / 2., 0., np.pi),
                              Vector(-0.009973019361495972, -0.005215135216712952, 0.003237169608473778))

    offset_psm2 = PyKDL.Frame(Rotation.RPY(-np.pi / 2., 0., 0.),
                              Vector(0.009973019361495972, -0.005215135216712952, 0.003237169608473778))
    psm1.move_jp(psm1_init)
    psm1.set_jaw_angle(0.7)
    psm2.move_jp(psm2_init)
    psm2.set_jaw_angle(0.7)
    time.sleep(3.0)

    needle_topic = NeedleInitialization(simulation_manager)
    psm2_tip = simulation_manager.get_obj_handle('psm2/toolyawlink')
    # Sanity sleep
    time.sleep(0.5)
    # This method will automatically start moving the needle to be with the PSM2's jaws
    needle_topic.move_to(psm2_tip)
    time.sleep(0.5)
    for i in range(30):
        # Close the jaws to grasp the needle
        # Calling it repeatedly a few times so that the needle is forced
        # between the gripper tips and grasped properly
        psm2.set_jaw_angle(0.0)
        time.sleep(0.01)
    time.sleep(0.5)
    # Don't forget to release the needle control loop to move it freely.
    needle_topic.release()
    time.sleep(2.0)
    # Open the jaws to let go of the needle from grasp
    psm2.set_jaw_angle(0.0)
    time.sleep(2.0)

    T_tINw = psm2_tip.get_pose()
    T_des = np.zeros((4, 4))
    rot_des = R.from_quat(T_tINw.M.GetQuaternion()).as_matrix()
    pos_des = np.array([T_tINw.p.x(), T_tINw.p.y(), T_tINw.p.z()])
    # T_des[0:3, 0:3] = T_tINw.M
    # T_des[0:3, 3] = T_tINw.p

    T_p_b = psm2.get_T_b_w()  # from base to psm
    T_b_p = psm2.get_T_w_b()  # from psm to base
    psm2_pose_cp = psm2.measured_cp()
    psm2_pose = psm2.measured_jp()
    psm2_pose.append(0.0)
    s1 = compute_FK(psm2_pose, 7)
    s2 = compute_FK(psm2_pose, 6)
    L_yaw2ctrlpnt = 0.0106
    T_psmtip_offset = Frame(Rotation.RPY(0, 0, 0),
                     L_yaw2ctrlpnt * Vector(0.0, 0.0, 1.0)) ## (0,0,-1)
    T_PinchJoint_7 = Frame(Rotation.RPY(0, 0, 0),
                           L_yaw2ctrlpnt * Vector(0.0, 0.0, -1.0))
    # Pinch Joint in Origin
    # T_PinchJoint_0 = T_7_0 * T_PinchJoint_7

    T_psm_yaw = convert_mat_to_frame(s2)
    T_test = T_p_b * T_psm_yaw ## == TtInw
    test_ik = compute_IK(T_psm_yaw*T_psmtip_offset)
