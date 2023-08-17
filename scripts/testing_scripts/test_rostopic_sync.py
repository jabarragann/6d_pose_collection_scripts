import time
import rospy, rostopic
import message_filters
import rostopic
from ambf6dpose.DataCollection.Rostopics import RosTopics

class TestRosSyncClient:
    def __init__(self):
        self.subscribers = []
        for topic in RosTopics:
            self.subscribers.append(message_filters.Subscriber(topic.value[0], topic.value[1]))

        # WARNING: TimeSynchronizer did not work. Use ApproximateTimeSynchronizer instead.
        # self.time_sync = message_filters.TimeSynchronizer(self.subscribers, 10)
        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, queue_size=10, slop=0.05
        )

        self.last_time = time.time()
        self.time_sync.registerCallback(self.common_cb)
        time.sleep(0.25)

    def common_cb(self, *inputs):
        print(f"Time from last message {(time.time() - self.last_time)*1000:0.3f}")
        print(f"message received")
        self.last_time = time.time()


if __name__ == "__main__":
    rospy.init_node("test_ros_client")
    client = TestRosSyncClient()
    rospy.spin()
