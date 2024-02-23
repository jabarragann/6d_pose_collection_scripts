from dataclasses import dataclass
import signal
import time
import ambf6dpose.RosBagReplay.RosbagUtils as rosbagutils
from ambf6dpose.DataCollection.Rostopics import get_image_processor, RosTopics
import click
from pathlib import Path
from click_params import FloatListParamType
import rospy
import cv2

default_path = "/home/juan1995/research/accelnet_grant/6d_pose_collection_scripts/test_replay/src_env2_v1.1.3_rec03_jack.bag"


@dataclass
class SimpleImgSubs:
    output_path: Path

    def __post_init__(self):
        assert self.output_path.exists()
        self.img_processor = get_image_processor()
        self.img_subs = rospy.Subscriber(
            RosTopics.CAMERA_L_IMAGE.value[0],
            RosTopics.CAMERA_L_IMAGE.value[1],
            callback=self.topic_cb,
        )
        self.count = 0
        self.saving_mode: bool = False

    def start_saving(self):
        self.saving_mode = True

    def stop_saving(self):
        self.saving_mode = False

    def save_frame(self, img):
        out_path = str(self.output_path / f"frame_{self.count:05d}.png")
        cv2.imwrite(out_path, img)
        self.count += 1

    def topic_cb(self, msg):
        img = self.img_processor(msg)
        if self.saving_mode:
            self.save_frame(img)

    def __enter__(self):
        self.start_saving()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_saving()


def setup_sigint_handler(bag_player: rosbagutils.RosbagReplayer):
    """Function to ensure that the bag player will stop when the user presses ctrl+c
    https://stackoverflow.com/questions/12371361/using-variables-in-signal-handler-require-global
    """

    def signal_handler(sig, frame):
        bag_player.run = False
        print("\nClosing player")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)  # ctrl+c handler


@click.command()
@click.option(
    "--bag_path",
    default=default_path,
    help="Path to bag file",
    type=click.Path(exists=True),
)
@click.option("--percent_to_replay", default=1.0, help="Path to bag file")
@click.option(
    "--ecm_pos",
    default="",
    help="ECM joint position, specifid as a string of 4 float separated with a space, \
    e.g., '1.0 1.0 1.0 1.0'. If not provide current camera pose will be used.",
    type=FloatListParamType(" ", ignore_empty=True),
)
@click.option("-r", "record_im", is_flag=True, default=False, help="Record images")
@click.option(
    "-o",
    "--output_p",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Only required if record flag (-r) is set.",
)
def single_replay(bag_path, percent_to_replay, ecm_pos, record_im, output_p: Path):
    assert len(ecm_pos) == 4 or len(ecm_pos) == 0, "ecm jp needs 4 values"

    if record_im:
        assert output_p is not None, "output_p needs to be set to record data"
        output_p.mkdir(parents=True, exist_ok=True)
        output_p.resolve()
        img_recorder = SimpleImgSubs(output_p)

    bag_path = Path(bag_path)

    # Init ambf objects
    bag_player = rosbagutils.RosbagReplayer()
    if len(ecm_pos) == 4:
        bag_player.move_cam(ecm_pos)
    setup_sigint_handler(bag_player)

    # Load cmds from bag
    ecm_pos, psm1_pos, psm2_pos, psm1_jaw, psm2_jaw = rosbagutils.read_rosbag(bag_path)

    # run replay
    if record_im:
        with img_recorder:
            bag_player.run_replay(
                psm1_pos,
                psm1_jaw,
                psm2_pos,
                psm2_jaw,
                percent_to_replay=percent_to_replay,
            )
    else:
        bag_player.run_replay(
            psm1_pos, psm1_jaw, psm2_pos, psm2_jaw, percent_to_replay=percent_to_replay
        )

    input("Press enter to return to home ")
    bag_player.return_psm_to_home()
    bag_player.reset_bodies()


@click.command()
@click.option("--bag_path", default=default_path, help="Path to bag file")
@click.option("--n_replays", default=2, help="Path to bag file")
@click.option("--percent_to_replay", default=1.0, help="Path to bag file")
def loop_replay(bag_path, n_replays, percent_to_replay):
    """
    Replay motions from bag file N times.
    """
    # Init ambf objects
    bag_player = rosbagutils.RosbagReplayer()

    setup_sigint_handler(bag_player)

    # Load cmds from bag
    ecm_pos, psm1_pos, psm2_pos, psm1_jaw, psm2_jaw = rosbagutils.read_rosbag(bag_path)

    try:
        for i in range(n_replays):
            print(f"Replay {i}")
            bag_player.move_psm_to_start(psm1_pos[0], psm2_pos[0])
            bag_player.run_replay(
                psm1_pos,
                psm1_jaw,
                psm2_pos,
                psm2_jaw,
                percent_to_replay=percent_to_replay,
            )
            time.sleep(0.3)

            bag_player.return_psm_to_home()
            bag_player.reset_bodies()
    except KeyboardInterrupt:
        print("finishing")


@click.group(help="Replay motions from rosbag")
def main():
    pass


if __name__ == "__main__":
    main.add_command(single_replay)
    main.add_command(loop_replay)
    main()
