import time
import rospy, rostopic
import message_filters
import rostopic
from ambf6dpose.DataCollection.Rostopics import RosTopics
import click


selected_topics = [
    RosTopics.CAMERA_L_IMAGE,
    RosTopics.CAMERA_FRAME,
    RosTopics.NEEDLE,
    RosTopics.CAMERA_L_IMAGE,
    RosTopics.CAMERA_L_SEG_IMAGE,
    # RosTopics.CAMERA_L_DEPTH,
    RosTopics.PSM1_TOOL_PITCH_LINK,
    RosTopics.PSM2_TOOL_PITCH_LINK,
    RosTopics.PSM1_TOOL_YAW_LINK,
    RosTopics.PSM2_TOOL_YAW_LINK,
    RosTopics.PSM1_BASE_LINK, #----------------added by Shuang
]


class TestRosSyncClient:
    def __init__(self, slop):
        self.subscribers = []
        for topic in selected_topics:
            self.subscribers.append(
                message_filters.Subscriber(topic.value[0], topic.value[1])
            )

        # WARNING: TimeSynchronizer did not work. Use ApproximateTimeSynchronizer instead.
        # self.time_sync = message_filters.TimeSynchronizer(self.subscribers, 10)
        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, queue_size=10, slop=slop
        )

        self.last_time = time.time()
        self.time_sync.registerCallback(self.common_cb)
        time.sleep(0.25)

    def common_cb(self, *inputs):
        print(
            f"Time from last message {(time.time() - self.last_time)*1000:0.3f}ms",
            end="\r",
        )
        self.last_time = time.time()


@click.command()
@click.option("--slop", default=0.05, type=float)
def test_sync_client(slop: float):
    """Test approximate syncronize message filter"""

    rospy.init_node("test_ros_client")
    client = TestRosSyncClient(slop)
    rospy.spin()


if __name__ == "__main__":
    test_sync_client()
