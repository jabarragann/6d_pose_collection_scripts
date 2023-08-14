import time
import os
import sys
from glob import glob
import rosbag
import gc
# from surgical_robotics_challenge.utils.task3_init import NeedleInitialization
from surgical_robotics_challenge.psm_arm import PSM
from surgical_robotics_challenge.ecm_arm import ECM
from surgical_robotics_challenge.simulation_manager import SimulationManager
from attach_needle_modify import AttachNeedle
from surgical_robotics_challenge.utils import coordinate_frames
dynamic_path = os.path.abspath(__file__ + "/../")
# data_path = os.path.abspath(__file__+"/../../../../")
# print(dynamic_path)
sys.path.append(dynamic_path)

if __name__ == '__main__':
    # data_folder = os.path.join(dynamic_path, 'data_test')
    # rosbag_name_list = glob(os.path.join(data_folder, '*.bag'))
    # rosbag_name = rosbag_name_list[0]
    # rosbag_name = '/home/zhyjack/dVRK_LfD_simulation/data/test_4.bag'
    # rosbag_name = '/ssd/test_new_1.bag'
    # rosbag_name = '/ssd/test_new_2.bag'
    rosbag_name='/home/jackzhy/user_study_latest_data/user_dave_01.bag'
    # output_folder = os.path.join(dynamic_path, 'test_image')
    #
    # if not os.path.exists(output_folder):
    #     print('Create Output Folder')
    #     os.makedirs(output_folder)

    bag = rosbag.Bag(rosbag_name)
    topics = list(bag.get_type_and_topic_info()[1].keys())
    types = [val[0] for val in bag.get_type_and_topic_info()[1].values()]

    count = 0
    topics_name = []
    psm1_pos = []
    psm2_pos = []
    t_psm1 = []
    t_psm2 = []
    psm1_jaw = []
    psm2_jaw = []
    ecm_pos = []

    ### old bag replay
    # for topic, msg, t in bag.read_messages(topics=topics[16]):
    #     assert topic == '/psm1/setpoint_js', 'load incorrect topics for psm 1 jp'
    #     # psm1_pos_temp = msg.joint_positions[0:6]
    #     psm1_pos_temp = list(msg.data)
    #     psm1_pos.append(psm1_pos_temp)
    #     count += 1
    # print('psm 1 record count: ', count)
    # count = 0
    #
    # for topic, msg, t in bag.read_messages(topics=topics[15]):
    #     assert topic == '/psm1/jaw/setpoint_js', 'load incorrect topics for psm 1 jaw'
    #     psm1_jaw_temp = msg.data
    #     psm1_jaw.append(psm1_jaw_temp)
    #     count += 1
    # print('psm 1 jaw record count: ', count)
    # count = 0
    #
    # for topic, msg, t in bag.read_messages(topics=topics[18]):
    #     assert topic == '/psm2/setpoint_js', 'load incorrect topics for psm 2 jp'
    #     # psm1_pos_temp = msg.joint_positions[0:6]
    #     psm2_pos_temp = list(msg.data)
    #     psm2_pos.append(psm2_pos_temp)
    #     count += 1
    # print('psm 2 record count: ', count)
    # count = 0
    #
    # for topic, msg, t in bag.read_messages(topics=topics[17]):
    #     assert topic == '/psm2/jaw/setpoint_js', 'load incorrect topics for psm 2 jaw'
    #     psm2_jaw_temp = msg.data
    #     psm2_jaw.append(psm2_jaw_temp)
    #     count += 1
    # print('psm 2 jaw record count: ', count)
    # count = 0
    #
    # for topic, msg, t in bag.read_messages(topics=topics[14]):
    #     assert topic == '/ecm/setpoint_js', 'load incorrect topics for ecm jp'
    #     ecm_pos_temp = msg.data
    #     ecm_pos.append(ecm_pos_temp)
    #     count += 1
    # print('ecm record count: ', count)
    # count = 0

    ### new bag replay
    for topic, msg, t in bag.read_messages(topics=topics[18]):
        assert topic == '/psm1/setpoint_js', 'load incorrect topics for psm 1 jp'
        # psm1_pos_temp = msg.joint_positions[0:6]
        psm1_pos_temp = list(msg.data)
        psm1_pos.append(psm1_pos_temp)
        count += 1
    print('psm 1 record count: ', count)
    count = 0

    for topic, msg, t in bag.read_messages(topics=topics[17]):
        assert topic == '/psm1/jaw/setpoint_js', 'load incorrect topics for psm 1 jaw'
        psm1_jaw_temp = msg.data
        psm1_jaw.append(psm1_jaw_temp)
        count += 1
    print('psm 1 jaw record count: ', count)
    count = 0

    for topic, msg, t in bag.read_messages(topics=topics[20]):
        assert topic == '/psm2/setpoint_js', 'load incorrect topics for psm 2 jp'
        # psm1_pos_temp = msg.joint_positions[0:6]
        psm2_pos_temp = list(msg.data)
        psm2_pos.append(psm2_pos_temp)
        count += 1
    print('psm 2 record count: ', count)
    count = 0

    for topic, msg, t in bag.read_messages(topics=topics[19]):
        assert topic == '/psm2/jaw/setpoint_js', 'load incorrect topics for psm 2 jaw'
        psm2_jaw_temp = msg.data
        psm2_jaw.append(psm2_jaw_temp)
        count += 1
    print('psm 2 jaw record count: ', count)
    count = 0

    for topic, msg, t in bag.read_messages(topics=topics[16]):
        assert topic == '/ecm/setpoint_js', 'load incorrect topics for ecm jp'
        ecm_pos_temp = msg.data
        ecm_pos.append(ecm_pos_temp)
        count += 1
    print('ecm record count: ', count)
    count = 0

    ### ambf raw replay
    # for topic, msg, t in bag.read_messages(topics=topics[12]):
    #     assert topic == '/ambf/env/psm1/baselink/State', 'load incorrect topics'
    #     psm1_pos_temp = [msg.joint_positions[0],
    #                      msg.joint_positions[1],
    #                      msg.joint_positions[2] / 10.,
    #                      msg.joint_positions[3],
    #                      msg.joint_positions[4],
    #                      msg.joint_positions[5]]
    #     psm1_pos.append(psm1_pos_temp)
    #     psm1_jaw_temp = (msg.joint_positions[-2] + msg.joint_positions[-1]) / 2.
    #     psm1_jaw.append(psm1_jaw_temp)
    #     t_psm2.append(t)
    #     count += 1
    # print('psm1 record count: ', count)
    # count = 0
    #
    # for topic, msg, t in bag.read_messages(topics=topics[14]):
    #     assert topic == '/ambf/env/psm2/baselink/State', 'load incorrect topics'
    #     psm2_pos_temp = [msg.joint_positions[0],
    #                      msg.joint_positions[1],
    #                      msg.joint_positions[2] / 10.,
    #                      msg.joint_positions[3],
    #                      msg.joint_positions[4],
    #                      msg.joint_positions[5]]
    #     psm2_pos.append(psm2_pos_temp)
    #     psm2_jaw_temp = (msg.joint_positions[-2] + msg.joint_positions[-1]) / 2.
    #     psm2_jaw.append(psm2_jaw_temp)
    #     t_psm2.append(t)
    #     count += 1
    # print('psm2 record count: ', count)
    # count = 0
    gc.collect()

    simulation_manager = SimulationManager('record_test')
    time.sleep(0.5)
    w = simulation_manager.get_world_handle()
    time.sleep(0.2)
    w.reset_bodies()
    time.sleep(0.2)
    cam = ECM(simulation_manager, 'CameraFrame')
    # cam.servo_jp([0.0, 0.05, -0.01, 0.0])
    time.sleep(0.5)
    psm1 = PSM(simulation_manager, 'psm1', add_joint_errors=False)
    time.sleep(0.5)
    # if psm1.is_present():
    #     print('psm1 run')
    #     T_psmtip_c = coordinate_frames.PSM1.T_tip_cam
    #     T_psmtip_b = psm1.get_T_w_b() * cam.get_T_c_w() * T_psmtip_c
    #     psm1.set_home_pose(T_psmtip_b)
    #     time.sleep(1.0)
    psm2 = PSM(simulation_manager, 'psm2', add_joint_errors=False)
    time.sleep(0.5)
    # if psm2.is_present():
    #     print('psm2 run')
    #     T_psmtip_c = coordinate_frames.PSM2.T_tip_cam
    #     T_psmtip_b = psm2.get_T_w_b() * cam.get_T_c_w() * T_psmtip_c
    #     psm2.set_home_pose(T_psmtip_b)
    #     time.sleep(1.0)

    # assert len(psm1_pos) == len(psm2_pos), 'psm1 and psm2 run time not equal'
    #
    # cam.servo_jp(ecm_pos[0])


    ### Add needle attachment code
    # needle = simulation_manager.get_obj_handle('Needle')
    # link1 = simulation_manager.get_obj_handle('psm1' + '/toolyawlink')
    # link2 = simulation_manager.get_obj_handle('psm2' + '/toolyawlink')

    # atn = AttachNeedle(needle, link1, link2)

    for i in range(len(psm1_pos)-1):
        cam.servo_jp(ecm_pos[i])
        psm1.servo_jp(psm1_pos[i])
        # psm1.set_jaw_angle(psm1_jaw[i] - 0.1)
        psm1.set_jaw_angle(psm1_jaw[i])
        psm2.servo_jp(psm2_pos[i])
        psm2.set_jaw_angle(psm2_jaw[i])
        time.sleep(0.01)

        # if (i > 2000) and (i < 4000):
        #     atn.attach_needle(needle, link2, atn.offset_psm2)

        count += 1
        print(count)

    # # First we shall move the PSM to its initial pose using joint commands OR pose command
    # psm2.servo_jp([-0.4, -0.22, 0.139, -1.64, -0.37, -0.11])
    # # Open the Jaws
    # psm2.set_jaw_angle(0.8)
    # # Sleep to achieve the target pose and jaw angle
    # time.sleep(1.0)
    # # Instantiate the needle initialization class
    # needle = NeedleInitialization(simulation_manager)
    # psm2_tip = simulation_manager.get_obj_handle('psm2/toolyawlink')
    # # Sanity sleep
    # time.sleep(0.5)
    # # This method will automatically start moving the needle to be with the PSM2's jaws
    # needle.move_to(psm2_tip)
    # time.sleep(0.5)
    # for i in range(30):
    #    # Close the jaws to grasp the needle
    #    # Calling it repeatedly a few times so that the needle is forced
    #    # between the gripper tips and grasped properly
    #    psm2.set_jaw_angle(0.0)
    #    time.sleep(0.01)
    # time.sleep(0.5)
    # # Don't forget to release the needle control loop to move it freely.
    # needle.release()
    # time.sleep(2.0)
    # # Open the jaws to let go of the needle from grasp
    # psm2.set_jaw_angle(0.8)
    # time.sleep(2.0)
