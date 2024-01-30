import time
import os
import sys
from glob import glob
from typing import List
import rosbag
import gc
import argparse
import subprocess
from surgical_robotics_challenge.psm_arm import PSM
from surgical_robotics_challenge.ecm_arm import ECM
from surgical_robotics_challenge.simulation_manager import SimulationManager
from surgical_robotics_challenge.utils import coordinate_frames
from threading import Thread


def read_rosbag(rosbag_name):
    bag = rosbag.Bag(str(rosbag_name))
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


class RosbagReplayer:
    def __init__(self):
        self.init_src_objects()
        self.run = True

    def init_src_objects(self):
        """Init surgical robotics challenge objects"""

        self.simulation_manager = SimulationManager("record_test")
        time.sleep(0.2)
        self.w = self.simulation_manager.get_world_handle()
        time.sleep(0.2)
        self.w.reset_bodies()
        time.sleep(0.2)
        self.cam = ECM(self.simulation_manager, "CameraFrame")
        # self.cam.servo_jp([0.0, 0.05, -0.01, 0.0])
        time.sleep(0.2)
        self.psm1 = PSM(self.simulation_manager, "psm1", add_joint_errors=False)
        time.sleep(0.2)
        self.psm2 = PSM(self.simulation_manager, "psm2", add_joint_errors=False)
        time.sleep(0.2)

    def move_cam(self, ecm_jp: List[float]):
        # ECM servo jp will automatically interpolate
        self.cam.servo_jp(ecm_jp)

    def reset_bodies(self):
        self.w.reset_bodies()

    def run_replay(self, psm1_pos, psm1_jaw, psm2_pos, psm2_jaw, percent_to_replay: float = 1.0):
        """Replay motions and reset bodies at the end. Ensure the arms have
        been previously moved to the starting position

        """

        assert (
            percent_to_replay <= 1.0 and percent_to_replay >= 0.0
        ), "percent_to_replay must be between 0 and 1"
        self.run = True
        count = 0
        total_num = min(len(psm1_pos), len(psm2_pos), len(psm1_jaw), len(psm2_jaw))
        total_num = int(total_num * percent_to_replay)

        for i in range(total_num):
            if not self.run:
                break
            self.psm1.servo_jp(psm1_pos[i])
            self.psm1.set_jaw_angle(psm1_jaw[i])
            self.psm2.servo_jp(psm2_pos[i])
            self.psm2.set_jaw_angle(psm2_jaw[i])
            time.sleep(0.01)
            count += 1
            sys.stdout.write(f"\r Run Progress: {count} / {total_num}")
            sys.stdout.flush()

        # self.return_psm_to_home()
        # self.reset_bodies()
        # time.sleep(0.2)

    def stop_replay(self):
        self.run = False

    def move_psm_to_start(self, psm1_pos, psm2_pos):
        self.psm1.move_jp(psm1_pos, execute_time=1.0)
        self.psm2.move_jp(psm2_pos, execute_time=1.0)
        time.sleep(2.0)

    def return_psm_to_home(self):
        # Let needle go
        self.psm1.set_jaw_angle(0.8)
        self.psm2.set_jaw_angle(0.8)
        time.sleep(2.2)
        # Return to home
        self.psm1.move_jp([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], execute_time=0.8)
        self.psm1.set_jaw_angle(0.0)
        self.psm2.move_jp([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], execute_time=0.8)
        self.psm2.set_jaw_angle(0.0)
        time.sleep(2.2)
