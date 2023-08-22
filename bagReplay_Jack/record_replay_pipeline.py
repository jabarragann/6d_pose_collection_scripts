import time
import os
import sys
from glob import glob
import rosbag
import gc
import argparse
import subprocess
from surgical_robotics_challenge.psm_arm import PSM
from surgical_robotics_challenge.ecm_arm import ECM
from surgical_robotics_challenge.simulation_manager import SimulationManager
from surgical_robotics_challenge.utils import coordinate_frames
from threading import Thread

dynamic_path = os.path.abspath(__file__ + "/../../")
# data_path = os.path.abspath(__file__+"/../../../../")
print(dynamic_path)
sys.path.append(dynamic_path)


class ThreadWithReturn(Thread):
    def __init__(self, *args, **kwargs):
        super(ThreadWithReturn, self).__init__(*args, **kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args, **kwargs):
        super(ThreadWithReturn, self).join(*args, **kwargs)
        return self._return


def read_rosbag(rosbag_name):
    bag = rosbag.Bag(rosbag_name)
    topics = list(bag.get_type_and_topic_info()[1].keys())
    types = [val[0] for val in bag.get_type_and_topic_info()[1].values()]

    count = 0
    psm1_pos = []
    psm2_pos = []
    psm1_jaw = []
    psm2_jaw = []
    ecm_pos = []

    ### new bag replay
    for topic, msg, t in bag.read_messages(topics=topics[18]):
        assert topic == "/psm1/setpoint_js", "load incorrect topics for psm 1 jp"
        # psm1_pos_temp = msg.joint_positions[0:6]
        psm1_pos_temp = list(msg.data)
        psm1_pos.append(psm1_pos_temp)
        count += 1
    print("psm 1 record count: ", count)
    count = 0

    for topic, msg, t in bag.read_messages(topics=topics[17]):
        assert topic == "/psm1/jaw/setpoint_js", "load incorrect topics for psm 1 jaw"
        psm1_jaw_temp = msg.data
        psm1_jaw.append(psm1_jaw_temp)
        count += 1
    print("psm 1 jaw record count: ", count)
    count = 0

    for topic, msg, t in bag.read_messages(topics=topics[20]):
        assert topic == "/psm2/setpoint_js", "load incorrect topics for psm 2 jp"
        # psm1_pos_temp = msg.joint_positions[0:6]
        psm2_pos_temp = list(msg.data)
        psm2_pos.append(psm2_pos_temp)
        count += 1
    print("psm 2 record count: ", count)
    count = 0

    for topic, msg, t in bag.read_messages(topics=topics[19]):
        assert topic == "/psm2/jaw/setpoint_js", "load incorrect topics for psm 2 jaw"
        psm2_jaw_temp = msg.data
        psm2_jaw.append(psm2_jaw_temp)
        count += 1
    print("psm 2 jaw record count: ", count)
    count = 0

    for topic, msg, t in bag.read_messages(topics=topics[16]):
        assert topic == "/ecm/setpoint_js", "load incorrect topics for ecm jp"
        ecm_pos_temp = msg.data
        ecm_pos.append(ecm_pos_temp)
        count += 1
    print("ecm record count: ", count)
    return ecm_pos, psm1_pos, psm2_pos, psm1_jaw, psm2_jaw


def run_replay(psm1_pos, psm1_jaw, psm2_pos, psm2_jaw):
    count = 0
    total_num = min(len(psm1_pos), len(psm2_pos), len(psm1_jaw), len(psm2_jaw))
    for i in range(total_num):
        psm1.servo_jp(psm1_pos[i])
        psm1.set_jaw_angle(psm1_jaw[i])
        psm2.servo_jp(psm2_pos[i])
        psm2.set_jaw_angle(psm2_jaw[i])
        time.sleep(0.01)
        count += 1
        sys.stdout.write(f"\r Run Progress: {count} / {total_num}")
        sys.stdout.flush()


def run_record(idx_i, idx_j, num_ecm):
    idx = idx_i * num_ecm + idx_j + 1
    command_record = (
        f"python3 {os.path.join(dynamic_path, 'scripts', 'collect_data.py')} "
        f"--path {os.path.join(save_folder)} "
        f"--scene_id {idx}"
    )
    process_record = subprocess.Popen(command_record.split(" "))
    return process_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    data_folder = os.path.join(dynamic_path, "test_replay")  ## add rosbags here!
    save_folder = os.path.join(dynamic_path, "test_record")  ## folder to save images
    file_list = glob(os.path.join(data_folder, "*.bag"))

    rosbag_name = file_list[0]

    # rosbag_name='/home/jackzhy/user_study_latest_data/user_jack_03.bag'
    # output_folder = os.path.join(dynamic_path, 'test_image')
    #
    if not os.path.exists(save_folder):
        print("Create Save Folder")
        os.makedirs(save_folder)

    # bag = rosbag.Bag(rosbag_name)
    # topics = list(bag.get_type_and_topic_info()[1].keys())
    # types = [val[0] for val in bag.get_type_and_topic_info()[1].values()]

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
    # gc.collect()

    simulation_manager = SimulationManager("record_test")
    time.sleep(0.2)
    w = simulation_manager.get_world_handle()
    time.sleep(0.2)
    w.reset_bodies()
    time.sleep(0.2)
    cam = ECM(simulation_manager, "CameraFrame")
    # cam.servo_jp([0.0, 0.05, -0.01, 0.0])
    time.sleep(0.2)
    psm1 = PSM(simulation_manager, "psm1", add_joint_errors=False)
    time.sleep(0.2)
    psm2 = PSM(simulation_manager, "psm2", add_joint_errors=False)
    time.sleep(0.2)

    ### preset ECM pose in joint space
    ecm_list = []
    ecm_list.append([0.0, 0.0, 0.0, 0.0])  # 0
    ecm_list.append([0.0, 0.05, -0.01, 0.0])  # 1
    ecm_list.append([0.0, 0.05, -0.01, 0.4])  # 2
    ecm_list.append([0.0, 0.05, -0.01, -0.4])  # 3
    ecm_list.append([0.2, 0.05, -0.01, 0.0])  # 4
    ecm_list.append([-0.2, 0.05, -0.01, 0.0])  # 5
    ecm_list.append([0.0, 0.15, -0.01, 0.0])  # 6
    ecm_list.append([0.0, -0.05, -0.01, 0.0])  # 7
    ecm_list.append([0.0, 0.05, -0.05, 0.0])  # 8
    ecm_list.append([0.0, 0.05, 0.03, 0.0])  # 9
    ecm_list.append([0.1, 0.05, -0.01, 0.2])  # 10
    ecm_list.append([-0.1, 0.05, -0.01, -0.2])  # 11
    ecm_list.append([0.1, 0.10, -0.01, 0.0])  # 12
    ecm_list.append([-0.1, 0.0, -0.01, 0.0])  # 13
    ecm_list.append([0.0, 0.10, -0.01, 0.3])  # 14
    ecm_list.append([0.0, 0.0, -0.01, 0.3])  # 15
    ecm_list.append([0.1, 0.10, -0.01, 0.1])  # 16
    ecm_list.append([-0.1, 0.0, -0.01, -0.1])  # 17
    ecm_list.append([0.1, 0.10, -0.04, -0.1])  # 18
    ecm_list.append([-0.1, 0.0, 0.02, 0.1])  # 19

    num_ecm = len(ecm_list)
    #
    # ### test ECM poses
    # psm1_jp = [0.30780306382205863, -0.22222915389237488, 0.1423643360325034,
    #            -1.3613186165319513,0.5750600725456388, -0.8399263308008617]
    # psm2_jp = [-0.46695894800579796, -0.17860657808832947, 0.15012366098379068,
    #            -1.0873261421084663, 0.7172512403887915, 0.48780102579228307]
    # psm1.servo_jp(psm1_jp)
    # psm1.set_jaw_angle(0.7)
    # psm2.servo_jp(psm2_jp)
    # psm2.set_jaw_angle(0.7)
    # time.sleep(0.5)
    # for idx in range(num_ecm):
    #     cam.servo_jp(ecm_list[idx])
    #     input('Press Enter to continue ...')

    for i in range(len(file_list)):
        rosbag_name = file_list[i]
        ecm_pos, psm1_pos, psm2_pos, psm1_jaw, psm2_jaw = read_rosbag(rosbag_name)
        gc.collect()
        for j in range(num_ecm):
            t_replay = Thread(target=run_replay, args=(psm1_pos, psm1_jaw, psm2_pos, psm2_jaw))
            t_record = ThreadWithReturn(target=run_record, args=(i, j, num_ecm))
            cam.servo_jp(ecm_list[j])
            time.sleep(0.5)
            print(f"\n Move to Camera Position {str(j).zfill(3)} ... \n")
            t_replay.start()
            t_record.start()
            t_replay.join()
            process_record = t_record.join()
            process_record.terminate()
            time.sleep(0.5)
            w.reset_bodies()
            time.sleep(1.0)
