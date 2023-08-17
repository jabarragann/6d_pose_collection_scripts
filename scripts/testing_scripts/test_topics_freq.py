"""
Test frequency of topics needed for dataset
"""

import time
import rospy
import rostopic
import rospy, rostopic
from ambf6dpose.DataCollection.Rostopics import RosTopics


class TestRosTopics:
    def __init__(self):
        self.freq_measurer = rostopic.ROSTopicHz(50)
        self.sub_list = []
        for topic in RosTopics:
            print(f"Subscribing to {topic.name}")

            s = rospy.Subscriber(
                topic.value[0],
                topic.value[1],
                callback=self.freq_measurer.callback_hz,
                callback_args=topic.value[0],
            )

            self.sub_list.append(s)

        print("collecting messages...")
        time.sleep(2.5)

    def print_hz(self):
        for topic in RosTopics:
            print(f"Topic: {topic.name}")
            # print_hz needs a list of topics
            self.freq_measurer.print_hz([topic.value[0]])

    def common_cb(self, msg, args):
        topic_type: RosTopics = args
        print(f"Topic: {topic_type.name}")
        # print(f"{msg}")
        pass


if __name__ == "__main__":
    rospy.init_node("test_freq")
    t = TestRosTopics()
    t.print_hz()
