"""
Subscribe to multiple topics with single callback function
https://stackoverflow.com/questions/74026635/ros-subscribe-to-multiple-topics-with-single-function
"""

import time
import rospy
import rostopic
from ambf6dpose.DataCollection.Rostopics import RosTopics

selected_topics = [RosTopics.CAMERA_L_IMAGE, RosTopics.CAMERA_L_SEG_IMAGE]
selected_topics += [RosTopics.PSM1_TOOL_PITCH_LINK, RosTopics.PSM2_TOOL_PITCH_LINK]


class TestRosClient:
    def __init__(self):
        self.r = rostopic.ROSTopicHz(50)
        for topic in RosTopics:
            if topic in selected_topics:
                print(f"Subscribing to {topic.name}")
                rospy.Subscriber(
                    topic.value[0],
                    topic.value[1],
                    callback=self.common_cb,
                    callback_args=topic,
                )

        time.sleep(0.5)

    def common_cb(self, msg, args):
        topic_type: RosTopics = args
        print(f"Topic: {topic_type.name}")


if __name__ == "__main__":
    rospy.init_node("test_ros_client")
    client = TestRosClient()
    rospy.spin()
