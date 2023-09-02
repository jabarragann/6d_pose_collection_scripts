"""
Script to automatically generate a dataset in BOP format with surgical robotics assets.

TODO:
* Monitor if the collection script fails during collection and restart it

"""
import subprocess
import os
import signal
import time
from typing import List
import ambf6dpose.RosBagReplay.RosbagUtils as rosbagutils
import click
from pathlib import Path
from click_params import IntListParamType

root_path = Path(__file__ + "/../../").resolve()


def get_camera_positions() -> List[List[float]]:
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
    return ecm_list


def setup_sigint_handler(bag_player: rosbagutils.RosbagReplayer):
    """Function to ensure that the bag player will stop when the user presses ctrl+c
    https://stackoverflow.com/questions/12371361/using-variables-in-signal-handler-require-global
    """

    def signal_handler(sig, frame):
        bag_player.run = False
        print("\nClosing player")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)  # ctrl+c handler


class RecorderManager:
    def __init__(self):
        pass

    def run_record(self, scene_id, save_folder):
        command_record = (
            f"python3 {os.path.join(root_path, 'scripts', 'collect_data.py')} "
            f"--path {os.path.join(save_folder)} "
            f"--scene_id {scene_id}"
        )
        self.record_pid = subprocess.Popen(command_record.split(" "))

        return self

    def __enter__(self):
        print("entering")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.record_pid.send_signal(signal.SIGINT)
        print("closing recorder...")
        time.sleep(1.0)


@click.command()
@click.option(
    "--save_folder",
    default="test_record",
    help="Save folder relative to project root. Default: test_record",
)
@click.option(
    "--bag_folder",
    default="test_replay",
    help="Record path relative to project root. Default:  test_replay",
)
@click.option(
    "--percent_to_replay",
    default=1.0,
    help="Percentage of the recording to replay",
    type=click.FloatRange(0.0, 1.0),
)
@click.option(
    "--ecm_positions",
    default="",
    help="Index of ECM positions, specifid as a string of ints separated with a space, \
    e.g., '1 2 3'. Default value: all available positions",
    type=IntListParamType(" ", ignore_empty=True),
)
def generate_dataset(save_folder, bag_folder, percent_to_replay, ecm_positions):
    """
    Collect 6d pose data from multiple view points
    """
    # Get ecm positions
    ecm_full_list = get_camera_positions()
    if len(ecm_positions) == 0:
        ecm_positions = list(range(0, len(ecm_full_list)))
    print(ecm_positions)
    assert all([i >= 0 for i in ecm_positions]), "ECM positions must be positive integers"
    assert all(
        [i < len(ecm_full_list) for i in ecm_positions]
    ), f"Only {len(ecm_full_list)} ECM positions are available"
    scene_ids = [i + 1 for i in ecm_positions]
    ecm_jp_list = [ecm_full_list[i] for i in ecm_positions]

    # Setup
    save_folder = root_path / save_folder
    bag_folder = root_path / bag_folder
    [p.mkdir(parents=True, exist_ok=True) for p in [save_folder, bag_folder]]
    bag_folder = Path(bag_folder)

    run_generate_dataset(save_folder, bag_folder, percent_to_replay, ecm_jp_list, scene_ids)


def run_generate_dataset(
    save_folder, bag_folder, percent_to_replay, ecm_jp_list: List[List[float]], scene_ids: List[int]
):
    """
    Collect data using the rosbag in bag_folder for every ecm position in ecm_positions. Ecm positions are specified
    in joint space. Scene_id are used to separate data from each view point in independent folders.
    """
    file_list = list(bag_folder.glob("*.bag"))

    # Setup recorder and bag player
    recorder_manager = RecorderManager()
    bag_player = rosbagutils.RosbagReplayer()
    setup_sigint_handler(bag_player)

    for bag_path in file_list:
        ecm_pos, psm1_pos, psm2_pos, psm1_jaw, psm2_jaw = rosbagutils.read_rosbag(str(bag_path))

        for id, ecm_pos in zip(scene_ids, ecm_jp_list):
            bag_player.cam.servo_jp(ecm_pos)
            time.sleep(0.5)
            bag_player.move_psm_to_start(psm1_pos[0], psm2_pos[0])

            with recorder_manager.run_record(id, save_folder):
                try:
                    bag_player.run_replay(
                        psm1_pos, psm1_jaw, psm2_pos, psm2_jaw, percent_to_replay=percent_to_replay
                    )
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    break

            bag_player.return_psm_to_home()
            bag_player.reset_bodies()
            time.sleep(0.2)


if __name__ == "__main__":
    generate_dataset()
