import os
import sys
from glob import glob
import bagpy
import rosbag
from bagpy import bagreader
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import seaborn as sns
# import matplotlib
# matplotlib.use('Qt5Agg')


dynamic_path = os.path.abspath(__file__+"/../")
# data_path = os.path.abspath(__file__+"/../../../../")
# print(dynamic_path)
sys.path.append(dynamic_path)


def read_bagpy(rosbag_name):
    b = bagreader(rosbag_name)

    all_data = []
    for t in b.topics:
        data_temp = b.message_by_topic(t)
        all_data.append(data_temp)

    data_read = pd.read_csv(all_data[0])

    return all_data, data_read


if __name__ == '__main__':
    # data_folder = os.path.join(dynamic_path, 'data_test')
    # rosbag_name_list = glob(os.path.join(data_folder, '*.bag'))
    # rosbag_name = rosbag_name_list[0]
    rosbag_name = '/home/zhyjack/dVRK_LfD_simulation/data/test_1.bag'
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
    for topic, msg, t in bag.read_messages(topics=topics[11]):
        # if count <= 10:
        assert topic == '/ambf/env/psm1/baselink/State', 'load incorrect topics'
        psm1_pos_temp = msg.joint_positions[0:6]
        psm1_pos.append(psm1_pos_temp)
        psm1_jaw_temp = (msg.joint_positions[-2] + msg.joint_positions[-1]) / 2.
        psm1_jaw.append(psm1_jaw_temp)
        t_psm1.append(t)
        count += 1

    for topic, msg, t in bag.read_messages(topics=topics[13]):
        # if count <= 10:
        assert topic == '/ambf/env/psm2/baselink/State', 'load incorrect topics'
        psm2_pos_temp = msg.joint_positions[0:6]
        psm2_pos.append(psm2_pos_temp)
        psm2_jaw_temp = (msg.joint_positions[-2] + msg.joint_positions[-1]) / 2.
        psm2_jaw.append(psm2_jaw_temp)
        t_psm2.append(t)
        count += 1

    gc.collect()
