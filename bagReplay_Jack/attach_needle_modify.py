#!/usr/bin/env python3
# //==============================================================================
# /*
#     Software License Agreement (BSD License)
#     Copyright (c) 2020-2021 Johns Hopkins University (JHU), Worcester Polytechnic Institute (WPI) All Rights Reserved.


#     All rights reserved.

#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions
#     are met:

#     * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.

#     * Neither the name of authors nor the names of its contributors may
#     be used to endorse or promote products derived from this software
#     without specific prior written permission.

#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#     "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#     LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#     FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#     COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#     INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#     BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#     ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#     POSSIBILITY OF SUCH DAMAGE.


#     \author    <amunawar@jhu.edu>
#     \author    Adnan Munawar
#     \version   1.0
# */
# //==============================================================================

from surgical_robotics_challenge.simulation_manager import SimulationManager
import PyKDL
from PyKDL import Vector, Rotation
from surgical_robotics_challenge.utils.utilities import cartesian_interpolate_step_num
import numpy as np
import time
import rospy
import sys
from std_msgs.msg import Int64
if sys.version_info[0] >= 3:
    from tkinter import *
else:
    from Tkinter import *


# class NeedleOffsets:
#     TnINt1 = PyKDL.Frame(Rotation.RPY(-np.pi/2., 0., np.pi),
#                    Vector(-0.009973019361495972, -0.005215135216712952, 0.003237169608473778))
#     TnINt2 = PyKDL.Frame(Rotation.RPY(-np.pi/2., 0., 0.),
#                    Vector(0.009973019361495972, -0.005215135216712952, 0.003237169608473778))
#

# def attach_needle(needle, link, T_offset):
#     reached = False
#     if link is None:
#         print('Not a valid link, returning')
#         return
#     T_nINw = needle.get_pose()
#     print('run attach')
#     while not reached and not rospy.is_shutdown():
#         T_tINw = link.get_pose()
#         T_nINw_cmd = T_tINw * T_offset
#
#         # print('needle: ', T_nINw)
#         #
#         # print('link: ', T_nINw_cmd)
#
#         T_delta, error_max = cartesian_interpolate_step_num(T_nINw, T_nINw_cmd, 0.01, 0.005)
#         r_delta = T_delta.M.GetRPY()
#         print(T_delta.p)
#         # print(T_delta.M)
#         if error_max < 0.01:
#             reached = True
#             print('done!')
#             break
#
#         T_cmd = Frame()
#         T_cmd.p = T_nINw.p + T_delta.p
#         T_cmd.M = T_nINw.M * Rotation.RPY(r_delta[0], r_delta[1], r_delta[2])
#         T_nINw = T_cmd
#         # print(T_cmd)
#         needle.set_pose(T_cmd)
#         time.sleep(0.001)
#         # T_nINw = get_obj_trans(needle)
#
#     # Wait for the needle to get there
#     time.sleep(5.0)
#
#     # You should see the needle in the center of the two fingers.
#     # If the gripper is not already closed, you shall have to manually
#     # close it to grasp the needle. You should probably automate this in the testIK script.
#
#     # Don't forget to release the pose command from the needle. We can
#     # do so by calling:
#     needle.set_force([0, 0, 0])
#     needle.set_torque([0, 0, 0])
class AttachNeedle:
    def __init__(self, needle, link1, link2):
        self.needle = needle
        self.psm1 = link1
        self.psm2 = link2
        self.offset_psm1 = PyKDL.Frame(Rotation.RPY(-np.pi/2., 0., np.pi),
                                       Vector(-0.009973019361495972, -0.005215135216712952, 0.003237169608473778))
        self.offset_psm2 = PyKDL.Frame(Rotation.RPY(-np.pi/2., 0., 0.),
                                       Vector(0.009973019361495972, -0.005215135216712952, 0.003237169608473778))
        self._pub_attach = rospy.Publisher('/Attach_needle', Int64, queue_size=1)
        self._pub_attach.publish(0)

    @staticmethod
    def attach_needle(needle, link, T_offset):
        reached = False
        if link is None:
            print('Not a valid link, returning')
            return
        T_nINw = needle.get_pose()
        print('run attach')
        while not reached and not rospy.is_shutdown():
            T_tINw = link.get_pose()
            T_nINw_cmd = T_tINw * T_offset

            # print('needle: ', T_nINw)
            #
            # print('link: ', T_nINw_cmd)

            T_delta, error_max = cartesian_interpolate_step_num(T_nINw, T_nINw_cmd, 0.01, 0.005)
            r_delta = T_delta.M.GetRPY()
            # print(T_delta.p)
            # print(T_delta.M)
            if error_max < 0.01:
                reached = True
                print('done!')
                break

            T_cmd = Frame()
            T_cmd.p = T_nINw.p + T_delta.p
            T_cmd.M = T_nINw.M * Rotation.RPY(r_delta[0], r_delta[1], r_delta[2])
            T_nINw = T_cmd
            # print(T_cmd)
            needle.set_pose(T_cmd)
            time.sleep(0.001)
            # T_nINw = get_obj_trans(needle)

        # Wait for the needle to get there
        time.sleep(4.0)

        # You should see the needle in the center of the two fingers.
        # If the gripper is not already closed, you shall have to manually
        # close it to grasp the needle. You should probably automate this in the testIK script.

        # Don't forget to release the pose command from the needle. We can
        # do so by calling:
        needle.set_force([0, 0, 0])
        needle.set_torque([0, 0, 0])

    def psm1_btn_cb(self):
        self._pub_attach.publish(1)
        self.attach_needle(self.needle, self.psm1, self.offset_psm1)
        time.sleep(0.5)
        self._pub_attach.publish(0)

    def psm2_btn_cb(self):
        self._pub_attach.publish(2)
        self.attach_needle(self.needle, self.psm2, self.offset_psm2)
        time.sleep(0.5)
        self._pub_attach.publish(0)


if __name__ == "__main__":
    simulation_manager = SimulationManager('attach_needle')
    # psm_name =
    needle = simulation_manager.get_obj_handle('Needle')
    link1 = simulation_manager.get_obj_handle('psm1' + '/toolyawlink')
    link2 = simulation_manager.get_obj_handle('psm2' + '/toolyawlink')
    time.sleep(0.5)
    attachmotion = AttachNeedle(needle, link1, link2)

    tk = Tk()
    tk.title("Attache Needle")
    tk.geometry("250x150")
    link1_button = Button(tk, text="PSM 1", command=attachmotion.psm1_btn_cb,
                          height=3, width=50, bg="red")
    link2_button = Button(tk, text="PSM 2", command=attachmotion.psm2_btn_cb,
                          height=3, width=50, bg="yellow")

    link1_button.pack()
    link2_button.pack()

    tk.mainloop()

    # rate = rospy.Rate(200)
    # pub_topic = rospy.Publisher('/Attach_needle', Int64, queue_size=1)
    #
    # try:
    #     while not rospy.is_shutdown():
    #         tk.mainloop()
    #         pub_topic.publish(0)
    #         rate.sleep()
    # except:
    #     print('Exception! Goodbye')

