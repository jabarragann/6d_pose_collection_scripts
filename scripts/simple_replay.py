import signal
import time
import ambf6dpose.RosBagReplay.RosbagUtils as rosbagutils
import click
from pathlib import Path

def_path = "/home/juan1995/research_juan/accelnet_grant/6d_pose_dataset_collection/test_replay/src_env2_v1.1.3_rec1_jack.bag"


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
@click.option("--bag_path", default=def_path, help="Path to bag file")
@click.option("--percent_to_replay", default=1.0, help="Path to bag file")
def replay_motions(bag_path, percent_to_replay):
    bag_path = Path(bag_path)
    assert bag_path.exists(), f"Path {bag_path} does not exist"

    # Init ambf objects
    bag_player = rosbagutils.RosbagReplayer()
    setup_sigint_handler(bag_player)

    # Load cmds from bag
    ecm_pos, psm1_pos, psm2_pos, psm1_jaw, psm2_jaw = rosbagutils.read_rosbag(bag_path)

    # run replay
    bag_player.run_replay(
        psm1_pos, psm1_jaw, psm2_pos, psm2_jaw, percent_to_replay=percent_to_replay
    )

    bag_player.return_psm_to_home()
    bag_player.reset_bodies()


@click.command()
@click.option("--bag_path", default=def_path, help="Path to bag file")
@click.option("--percent_to_replay", default=1.0, help="Path to bag file")
def loop_replay(bag_path, percent_to_replay):
    # Init ambf objects
    bag_player = rosbagutils.RosbagReplayer()

    setup_sigint_handler(bag_player)

    # Load cmds from bag
    ecm_pos, psm1_pos, psm2_pos, psm1_jaw, psm2_jaw = rosbagutils.read_rosbag(bag_path)

    try:
        for i in range(6):
            print(f"Replay {i}")
            bag_player.move_psm_to_start(psm1_pos[0], psm2_pos[0])
            bag_player.run_replay(
                psm1_pos, psm1_jaw, psm2_pos, psm2_jaw, percent_to_replay=percent_to_replay
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
    main.add_command(replay_motions)
    main.add_command(loop_replay)
    main()
